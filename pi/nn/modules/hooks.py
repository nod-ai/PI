from collections import OrderedDict
import weakref
from typing import Any

__all__ = [
    "RemovableHandle",
]


class RemovableHandle(object):
    id: str
    next_id: int = 0

    def __init__(self, hooks_dict: Any, name: str) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = f"{name}_{RemovableHandle.next_id}"
        RemovableHandle.next_id += 1

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __getstate__(self):
        return self.hooks_dict_ref(), self.id

    def __setstate__(self, state) -> None:
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self.remove()
