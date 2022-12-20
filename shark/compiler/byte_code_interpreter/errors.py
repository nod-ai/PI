class PyVMError(Exception):
    """For raising errors in the operation of the VM."""

    pass


class PyVMRuntimeError(Exception):
    """RuntimeError in operation of PyVM."""

    pass


class PyVMUncaughtException(Exception):
    """Uncaught RuntimeError in operation of PyVM."""

    def __init__(self, name, args, traceback=None):
        self.__name__ = name
        self.traceback = traceback
        self.args = args

    def __getattr__(self, name):
        if name == "__traceback__":
            return self.traceback
        else:
            return super().__getattr__(name)

    def __getitem__(self, i):
        assert 0 <= i <= 2, "Exception index should be in range 0..2 was %d" % i
        if i == 0:
            return self.__name__
        elif i == 1:
            return self.args
        else:
            return self.traceback

    @classmethod
    def from_tuple(cls, exception):
        assert (
            len(exception) == 3
        ), "Expecting exception tuple to have 3 args: type, args, traceback"
        return cls(*exception)

    pass


class NoSourceError(Exception):
    """For raising errors when we can't find source code."""

    pass
