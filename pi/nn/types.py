import builtins
import weakref
from enum import Enum
from typing import Tuple
from typing import Union, List


Size = size = Union[List[builtins.int], Tuple[builtins.int, ...]]


class BroadcastingListCls(object):
    def __getitem__(self, types):
        return


BroadcastingList1 = BroadcastingList2 = BroadcastingList3 = BroadcastingListCls()

# Wrapper functions that can call either of 2 functions depending on a boolean
# argument
boolean_dispatched: "weakref.WeakKeyDictionary[Callable, Dict[str, Callable]]" = (
    weakref.WeakKeyDictionary()
)  # noqa: T484


def boolean_dispatch(
    arg_name, arg_index, default, if_true, if_false, module_name, func_name
):
    def fn(*args, **kwargs):
        dispatch_flag = False
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]

        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)

    if module_name is not None:
        fn.__module__ = module_name
    if func_name is not None:
        fn.__name__ = func_name

    boolean_dispatched[fn] = {
        "if_true": if_true,
        "if_false": if_false,
        "index": arg_index,
        "default": default,
        "arg_name": arg_name,
    }
    return fn


