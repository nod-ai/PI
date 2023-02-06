import inspect
import itertools
import warnings
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Callable, Union

from ..._tensor import Tensor
from ...types_ import dtype as pi_dtype
from ..parameter import (
    Parameter,
    UninitializedParameter,
    UninitializedBuffer,
    is_uninitialized,
)
from ...utils import hooks
from ...utils.hooks import RemovableHandle


class Module:
    _parameters: Dict[str, Optional[Union[Parameter, UninitializedParameter]]]
    _buffers: Dict[str, Optional[Union[Tensor, UninitializedBuffer]]]
    _modules: Dict[str, Optional["Module"]]
    _forward_pre_hooks: OrderedDict[str, Callable]
    _forward_post_hooks: OrderedDict[str, Callable]
    __forward: Callable
    training = False

    def __init__(self):
        _set = super().__setattr__
        _get = super().__getattribute__

        _set("_parameters", {})
        _set("_buffers", {})
        _set("_modules", {})
        _set("_forward_pre_hooks", OrderedDict())
        _set("_forward_post_hooks", OrderedDict())
        _set(
            "_initialize_hook",
            _get("register_forward_pre_hook")(_get("_initialize")),
        )

        if "forward" in dir(self):
            orig_forward = _get("forward")
            # super attr is Module.__call__ d'oh
            call = self.__call__
            # name mangling means trying to get __forward actually tries
            # to get this
            _set("_Module__forward", orig_forward)
            _set("forward", call)
            # TODO(max): checks here
            if hasattr(orig_forward, "__placeholders__"):
                call.__dict__["__placeholders__"] = orig_forward.__placeholders__

        super(Module, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, *args, **kwargs):
        for hook_id, hook in self._forward_pre_hooks.items():
            result = hook(self, *args, **kwargs)  # type: ignore[misc]
            if result is not None:
                if isinstance(result, tuple) and len(result) == 2:
                    args, kwargs = result
                else:
                    raise RuntimeError(
                        "forward pre-hook must return None or a tuple "
                        f"of (new_args, new_kwargs), but got {result}."
                    )
        result = self.__forward(*args, **kwargs)
        for hook_id, hook in self._forward_post_hooks.items():
            result = hook(self, result, *args, **kwargs)

        return result

    def _register_hook(
        self,
        hook: Callable[..., None],
        hook_dict: OrderedDict[str, Callable],
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        hook_name = hook.__func__.__name__ if inspect.ismethod(hook) else hook.__name__
        handle = hooks.RemovableHandle(hook_dict, name=hook_name)
        hook_dict[handle.id] = hook
        if prepend:
            hook_dict.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    def register_forward_pre_hook(
        self,
        hook: Callable[..., None],
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        return self._register_hook(hook, self._forward_pre_hooks, prepend=prepend)

    def register_forward_post_hook(
        self,
        hook: Callable[..., None],
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        return self._register_hook(hook, self._forward_post_hooks, prepend=prepend)

    def __getattribute__(self, item):
        return super(Module, self).__getattribute__(item)

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        if value.__class__.__name__ == UninitializedParameter.__name__:
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
            )
            assert isinstance(
                value, UninitializedParameter
            ), f"class comparison failed {type(value)} {UninitializedParameter}"
            self.register_parameter(name, value)
        elif name in self._parameters:
            assert value is None or (
                isinstance(self._parameters[name], UninitializedParameter)
                and isinstance(value, Tensor)
            ), f"{name}:{type(value).__module__}.{type(value).__name__} cannot override parameter {name}:{type(self._parameters[name]).__module__}.{type(self._parameters[name]).__name__}"
            self.register_parameter(name, value)
        else:
            if isinstance(value, Module):
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                )
                self.register_module(name, value)
            elif name in self._modules:
                assert value is None, f"{type(value)} cannot override module {name}"
                self.register_module(name, value)
            else:
                if name in self._buffers:
                    warnings.warn(
                        f"{type(value)} overriding buffer {name} in {self.__class__.__name__}"
                    )
                    self.register_buffer(name, value)
                else:
                    super().__setattr__(name, value)

    def __getattr__(
        self, name: str
    ) -> Union[Tensor, "Module", UninitializedParameter, UninitializedBuffer]:
        _parameters = self.__dict__["_parameters"]
        if name in _parameters:
            return _parameters[name]
        _buffers = self.__dict__["_buffers"]
        if name in _buffers:
            return _buffers[name]
        modules = self.__dict__["_modules"]
        if name in modules:
            return modules[name]
        raise AttributeError(f"{type(self).__name__} object has no attribute {name}")

    def register_buffer(
        self,
        name: str,
        tensor: Optional[Union[Tensor, UninitializedBuffer]],
        persistent: bool = True,
    ) -> None:
        self._buffers[name] = tensor

    def register_parameter(
        self, name: str, param: Optional[Union[Parameter, UninitializedParameter]]
    ) -> None:
        self._parameters[name] = param

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{module} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {name}")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif "." in name:
            raise KeyError('module name can\'t contain ".", got: {}'.format(name))
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')
        self._modules[name] = module

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        self.add_module(name, module)

    def initialize_parameters(self, *_args, **_kwargs):
        parameters = self.__dict__["_parameters"]
        for name, param in sorted(parameters.items()):
            if isinstance(param, UninitializedParameter):
                parameters[name] = param()

    def initialize_buffers(self, *_args, **_kwargs):
        buffers = self.__dict__["_buffers"]
        for name, buffer in sorted(buffers.items()):
            if isinstance(buffer, UninitializedBuffer):
                buffers[name] = buffer()

    def initialize(self, *args, **kwargs):
        self.initialize_buffers(*args, **kwargs)
        self.initialize_parameters(*args, **kwargs)

    def has_uninitialized(self):
        params = self._parameters.items()
        buffers = self._buffers.items()
        for param_name, param in itertools.chain(params, buffers):
            if is_uninitialized(param):
                return param_name, param
        return None

    def not_uninitialized(self):
        params = self._parameters.values()
        buffers = self._buffers.values()

        for param in itertools.chain(params, buffers):
            if not is_uninitialized(param):
                return param
        return None

    def _initialize(self, _self, *args, **kwargs):
        for child in self.all_children():
            child.initialize(*args, **kwargs)
        for i, child in enumerate(self.all_children()):
            if uninitialized := child.has_uninitialized():
                raise RuntimeError(
                    f"module {i} {child.__class__.__name__} has not been fully initialized; {uninitialized}"
                )
        self._initialize_hook.remove()
        delattr(self, "_initialize_hook")

    def to(self, dtype: pi_dtype):
        if initialized_param := self.not_uninitialized():
            raise RuntimeError(
                f"module {self.__class__.__name__} has already been initialized; {initialized_param}"
            )

        for name, param in self._parameters.items():
            assert is_uninitialized(param), f"{param} already initialized"
            self._parameters[name] = UninitializedParameter(*param.size, dtype=dtype)

        for name, buffer in self._buffers.items():
            assert is_uninitialized(buffer), f"{buffer} already initialized"
            self._buffers[name] = UninitializedBuffer(*buffer.size, dtype=dtype)

        return self

    def all_children(self):
        def get_children(module):
            children = list(module._modules.values())
            flat_children = []
            if not children:
                return [module]
            else:
                for child in children:
                    children = get_children(child)
                    flat_children.extend(children)
            return flat_children

        return get_children(self)

    modules = all_children
