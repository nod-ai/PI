import types

from .dialects import _torch_wrappers

all_ops = {o: _torch_wrappers.__dict__[o] for o in _torch_wrappers.__all__}


class _OpNamespace(types.ModuleType):
    def __init__(self, name):
        super(_OpNamespace, self).__init__("pi." + name)
        self.name = name
        self._dir = []

    def __iter__(self):
        return iter(self._dir)

    def __getattr__(self, op_name):
        # It is not a valid op_name when __file__ is passed in
        # if op_name == "__file__":
        #     return "pi.ops"
        # elif op_name == "__origin__":
        #     raise AttributeError()

        # namespace_name = self.name
        # qualified_op_name = "{}::{}".format(namespace_name, op_name)
        op_name = op_name.split(".")[-1]

        if op_name in all_ops:
            return all_ops[op_name]
        else:
            return _OpNamespace(op_name)

    def __call__(self, *args, **kwargs):
        if self.name in all_ops:
            return all_ops[self.name](*args, **kwargs)
        else:
            raise NotImplementedError(self.name)

        # TODO(max): resolve overloads correctly here
        # Get the op `my_namespace::my_op` if available. This will also check
        # for overloads and raise an exception if there are more than one.
        # try:
        #     op, overload_names = pi._C._jit_get_operation(qualified_op_name)
        # except RuntimeError as e:
        #     # Turn this into AttributeError so getattr(obj, key, default)
        #     # works (this is called by TorchScript with __origin__)
        #     raise AttributeError(
        #         f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'"
        #     ) from e
        #
        # # let the script frontend know that op is identical to the builtin op
        # # with qualified_op_name
        # pi.jit._builtins._register_builtin(op, qualified_op_name)
        # op.__module__ = self.__module__ + "." + namespace_name
        # opoverloadpacket = OpOverloadPacket(
        #     qualified_op_name, op_name, op, overload_names
        # )
        # opoverloadpacket.__module__ = self.__module__ + "." + namespace_name
        # # cache the opoverloadpacket to ensure that each op corresponds to
        # # a unique OpOverloadPacket object
        # setattr(self, op_name, opoverloadpacket)
        # self._dir.append(op_name)
        # return opoverloadpacket


# class _PyOpNamespace(_OpNamespace):
#     def __init__(self):
#         super(_PyOpNamespace, self).__init__("pi.ops")
#         self.pyop_namespace = all_ops


# class _Ops(types.ModuleType):
#     def __init__(self, name):
#         super(_Ops, self).__init__(name)
#
#     def __getattr__(self, name):
#         # Check if the name is a pyop
#         if name in self.pyops.pyop_namespace:
#             return self.pyops.pyop_namespace[name]
#
#         namespace = _OpNamespace(name)
#         setattr(self, name, namespace)
#         return namespace
