import builtins
import weakref
from enum import Enum
from typing import Tuple
from typing import Union, List

from pi import dtype

bfloat16 = dtype.bfloat16
bool = dtype.bool
complex32 = dtype.complex32
complex64 = dtype.complex64
half = float16 = dtype.float16
float = float32 = dtype.float32
double = float64 = dtype.float64
int8 = dtype.int8
int16 = dtype.int16
int32 = dtype.int32
long = int64 = dtype.int64
qint8 = dtype.qint8
quint8 = dtype.quint8
uint8 = dtype.uint8

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


class layout(Enum):
    strided = 1
    sparse_coo = 2
    sparse_csr = 3
    sparse_csc = 4
    sparse_bsr = 5
    sparse_bsc = 6
    _mkldnn = 7


strided: layout = layout.strided
sparse_coo: layout = layout.sparse_coo
sparse_csr: layout = layout.sparse_csr
sparse_csc: layout = layout.sparse_csc
sparse_bsr: layout = layout.sparse_bsr
sparse_bsc: layout = layout.sparse_bsc
_mkldnn: layout = layout._mkldnn


class memory_format(Enum):
    contiguous_format = 0
    preserve_format = 1
    channels_last = 2
    channels_last_3d = 3


contiguous_format = memory_format.contiguous_format
preserve_format = memory_format.preserve_format
channels_last = memory_format.channels_last
channels_last_3d = memory_format.channels_last_3d
