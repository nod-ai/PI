from __future__ import annotations

import inspect
import itertools
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Callable, Union

from shark import Tensor
from shark.nn.parameter import (
    Parameter,
    UninitializedParameter,
    UninitializedBuffer,
    is_uninitialized,
)
from shark.utils import hooks
from shark.utils.hooks import RemovableHandle


class Module:
    _parameters: Dict[str, Optional[Union[Parameter, UninitializedParameter]]]
    _buffers: Dict[str, Optional[Union[Tensor, UninitializedBuffer]]]
    _modules: Dict[str, Optional[Module]]
    _forward_pre_hooks: OrderedDict[str, Callable]
    _forward: Callable

    def __init__(self):
        _set = super().__setattr__
        _get = super().__getattribute__

        _set("_parameters", {})
        _set("_buffers", {})
        _set("_modules", {})
        _set("_forward_pre_hooks", OrderedDict())
        _set(
            "_initialize_hook",
            _get("register_forward_pre_hook")(_get("_infer_parameters")),
        )

        if "forward" in dir(self):
            orig_forward = _get("forward")
            # super attr is Module.__call__ d'oh
            call = self.__call__
            _set("_forward", orig_forward)
            _set("forward", call)
            # TODO(max): checks here
            if hasattr(orig_forward, "__placeholders__"):
                # setattr(call, "__annotations__", orig_forward.__annotations__)
                call.__dict__["__placeholders__"] = orig_forward.__placeholders__

        super(Module, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

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
        return self._forward(*args, **kwargs)

    def register_forward_pre_hook(
        self,
        hook: Callable[..., None],
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        handle = hooks.RemovableHandle(self._forward_pre_hooks)
        hook_name = hook.__func__.__name__ if inspect.ismethod(hook) else hook.__name__
        hook_id = f"{hook_name}_{handle.id}"
        self._forward_pre_hooks[hook_id] = hook
        if prepend:
            self._forward_pre_hooks.move_to_end(hook_id, last=False)  # type: ignore[attr-defined]
        return handle

    def __getattribute__(self, item):
        return super(Module, self).__getattribute__(item)

    def __setattr__(self, name: str, value: Union[Tensor, Module]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        if isinstance(value, (Parameter, UninitializedParameter)):
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
            )
            self.register_parameter(name, value)
        elif name in self._parameters:
            assert value is None or (
                isinstance(self._parameters[name], UninitializedParameter)
                and isinstance(value, Tensor)
            ), f"{type(value)} cannot override parameter {name}"
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
                    assert value is None, f"{type(value)} cannot override buffer {name}"
                    self.register_buffer(name, value)
                else:
                    super().__setattr__(name, value)

    def __getattr__(
        self, name: str
    ) -> Union[Tensor, Module, UninitializedParameter, UninitializedBuffer]:
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

    def register_module(self, name: str, module: Optional[Module]) -> None:
        self._modules[name] = module

    def initialize_parameters(self, *_args, **_kwargs):
        parameters = self.__dict__["_parameters"]
        for name, param in sorted(parameters.items()):
            if isinstance(param, UninitializedParameter):
                parameters[name] = param()

    def has_uninitialized_params(self):
        params = self._parameters.values()
        buffers = self._buffers.values()
        for param in itertools.chain(params, buffers):
            if is_uninitialized(param):
                return param
        return None

    def _infer_parameters(self, _self, *args, **kwargs):
        self.initialize_parameters(*args, **kwargs)
        if uninitialized_param := self.has_uninitialized_params():
            raise RuntimeError(
                f"module {self.__class__.__name__} has not been fully initialized; {uninitialized_param}"
            )
            self._initialize_hook.remove()
            delattr(self, "_initialize_hook")
