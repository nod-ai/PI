import sys

import logging
import threading
from contextlib import contextmanager
from importlib.abc import MetaPathFinder
from importlib.machinery import SourceFileLoader, ModuleSpec
from importlib.util import find_spec, spec_from_loader
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


OVERLOADS = {
    "torch_mlir.dialects._arith_ops_ext": str(
        Path(__file__).parent / "_arith_ops_ext.py"
    ),
    "torch_mlir.dialects._memref_ops_ext": str(
        Path(__file__).parent / "_memref_ops_ext.py"
    ),
    "torch_mlir.dialects._torch_ops_ext_custom": str(
        Path(__file__).parent / "_torch_ops_ext_custom.py"
    ),
    "torch_mlir.dialects._torch_ops_ext": str(
        Path(__file__).parent / "_torch_ops_ext.py"
    ),
    # TODO(max): upstream to get rid of this hack
    "pyccolo.trace_events": str(
        Path(__file__).parent.parent / "compiler" / "tracing" / "trace_events.py"
    ),
}


# this is based on the birdseye finder (which uses import hooks based on MacroPy's):
# https://github.com/alexmojaki/birdseye/blob/9974af715b1801f9dd99fef93ff133d0ab5223af/birdseye/import_hook.py
class Overloader(MetaPathFinder):
    def __init__(self) -> None:
        self.tracers = None
        self._thread = threading.current_thread()

    @contextmanager
    def _clear_preceding_finders(self) -> Generator[None, None, None]:
        """
        Clear all preceding finders from sys.meta_path, and restore them afterwards.
        """
        orig_finders = sys.meta_path
        try:
            sys.meta_path = sys.meta_path[sys.meta_path.index(self) + 1 :]  # noqa: E203
            yield
        finally:
            sys.meta_path = orig_finders

    def _find_plain_spec(self, fullname, path, target):
        """Try to find the original module using all the
        remaining meta_path finders."""
        spec = None
        self_seen = False
        for finder in sys.meta_path:
            if finder is self:
                self_seen = True
                continue
            elif not self_seen or "pytest" in finder.__module__:
                # when testing with pytest, it installs a finder that for
                # some yet unknown reasons makes birdseye
                # fail. For now it will just avoid using it and pass to
                # the next one
                continue
            if hasattr(finder, "find_spec"):
                spec = finder.find_spec(fullname, path, target=target)
            elif hasattr(finder, "load_module"):
                spec = spec_from_loader(fullname, finder)

            if spec is not None and spec.origin != "builtin":
                return spec

    def find_spec(self, fullname, path=None, target=None):
        if threading.current_thread() is not self._thread:
            return None
        if target is None:
            with self._clear_preceding_finders():
                spec = find_spec(fullname, path)
        else:
            spec = self._find_plain_spec(fullname, path, target)

        if fullname not in OVERLOADS:
            if spec is None or not (
                hasattr(spec.loader, "get_source") and callable(spec.loader.get_source)
            ):  # noqa: E128
                if fullname != "org":
                    # stdlib pickle.py at line 94 contains a ``from
                    # org.python.core for Jython which is always failing,
                    # of course
                    logger.debug("Failed finding spec for %s", fullname)
                return None

            if not isinstance(spec.loader, SourceFileLoader):
                return None
            return spec

        logger.debug("patching spec for %s", fullname)
        new_path = OVERLOADS[fullname]
        source_file_loader = SourceFileLoader(fullname, new_path)
        spec = ModuleSpec(
            name=fullname,
            loader=source_file_loader,
            origin=new_path,
            is_package=False,
        )
        spec.has_location = True
        return spec


if len(sys.meta_path) > 0 and isinstance(sys.meta_path[0], Overloader):
    orig_meta_path_entry = sys.meta_path[0]
    sys.meta_path[0] = Overloader()
else:
    sys.meta_path.insert(0, Overloader())
