"""Root conftest — bridge between pytest absolute imports and source relative imports.

Problem: nodes/*.py use ``from ..lib.X import ...`` which requires a parent
package. Tests use ``from lib.X import ...`` (absolute). Without intervention,
Python loads two separate module objects for the same source file — one via
``lib.recipe`` and one via ``_ecaj.lib.recipe`` — causing isinstance() failures.

Fix: A single MetaPathFinder intercepts all imports of ``lib.*``, ``nodes.*``,
and ``_ecaj.*`` and ensures exactly ONE module object per source file, loaded
under a canonical ``_ecaj.*`` name and aliased under the short (``lib.*`` /
``nodes.*``) name.

- ``lib.*`` relative imports stay within ``lib/`` (no parent needed).
- ``nodes.*`` relative imports go ``from ..lib.X`` → ``_ecaj.lib.X``.
- Everything resolves to a single module object per file.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

collect_ignore = ["__init__.py"]

_ROOT = Path(__file__).resolve().parent
_PKG = "_ecaj"
_SUBS = frozenset(("lib", "nodes"))


class _UnifiedFinder(importlib.abc.MetaPathFinder):
    """Intercept lib.*, nodes.*, and _ecaj.* imports.

    Every module is loaded exactly once under its canonical name (_ecaj.X.Y)
    and aliased under the short name (X.Y).
    """

    def find_spec(self, fullname, path, target=None):
        parts = fullname.split(".")

        # Determine short and canonical names.
        if parts[0] in _SUBS:
            short, canonical = fullname, f"{_PKG}.{fullname}"
        elif parts[0] == _PKG and len(parts) > 1 and parts[1] in _SUBS:
            canonical, short = fullname, ".".join(parts[1:])
        else:
            return None

        # Already loaded — return the existing module for whichever name
        # was requested, ensuring both aliases exist.
        # If one alias was removed (e.g. a test force-deleted sys.modules
        # entry to trigger re-import), remove the other alias too and reload.
        if canonical in sys.modules:
            if short in sys.modules:
                mod = sys.modules[canonical]
                return _existing_spec(fullname, mod)
            # Short was removed — honour the re-import request.
            del sys.modules[canonical]
        elif short in sys.modules:
            if canonical not in sys.modules:
                # Only short exists — register canonical alias too.
                mod = sys.modules[short]
                sys.modules[canonical] = mod
                return _existing_spec(fullname, mod)

        # Ensure all parent packages are loaded first.
        short_parts = short.split(".")
        for i in range(1, len(short_parts)):
            parent_short = ".".join(short_parts[:i])
            parent_canonical = f"{_PKG}.{parent_short}"
            if parent_canonical not in sys.modules:
                # Recursive: triggers this finder for the parent.
                importlib.import_module(parent_short)

        # Parent loading may have loaded us as a side effect.
        if canonical in sys.modules:
            mod = sys.modules[canonical]
            sys.modules.setdefault(short, mod)
            return _existing_spec(fullname, mod)

        # Resolve source file on disk.
        rel = Path(*short_parts)
        base = _ROOT / rel
        if (base / "__init__.py").exists():
            fp = str(base / "__init__.py")
            search = [str(base)]
        elif base.with_suffix(".py").exists():
            fp = str(base.with_suffix(".py"))
            search = None
        else:
            return None

        spec = importlib.util.spec_from_file_location(
            fullname, fp, submodule_search_locations=search,
        )
        if spec is None:
            return None

        is_nodes = short_parts[0] == "nodes"
        spec.loader = _DualLoader(spec.loader, short, canonical, is_nodes)
        return spec


class _DualLoader(importlib.abc.Loader):
    """Load a module once and register under both short and canonical names."""

    def __init__(self, real, short, canonical, is_nodes):
        self._real = real
        self._short = short
        self._canonical = canonical
        self._is_nodes = is_nodes

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        is_pkg = hasattr(module, "__path__")

        # nodes/* need __package__ in the _ecaj namespace so that
        # ``from ..lib.X`` resolves to ``_ecaj.lib.X``.
        # lib/* keep __package__ in the short namespace (their relative
        # imports stay within lib/).
        if self._is_nodes:
            pkg = self._canonical if is_pkg else self._canonical.rsplit(".", 1)[0]
        else:
            pkg = self._short if is_pkg else self._short.rsplit(".", 1)[0]
        module.__package__ = pkg

        # Fix __spec__ so __spec__.parent matches __package__.
        # Python 3.12 warns on mismatch; 3.13+ will use __spec__.parent.
        # (module.__name__ is left alone — SourceFileLoader.exec_module
        # checks it against the loader's name, so it must stay as-is.)
        spec_name = self._canonical if self._is_nodes else self._short
        old = module.__spec__
        if old is not None and old.name != spec_name:
            new_spec = importlib.machinery.ModuleSpec(
                spec_name, old.loader, is_package=is_pkg, origin=old.origin,
            )
            if is_pkg and old.submodule_search_locations is not None:
                new_spec.submodule_search_locations = list(
                    old.submodule_search_locations
                )
            new_spec.has_location = getattr(old, "has_location", False)
            module.__spec__ = new_spec

        # Register under both names BEFORE exec to handle nested imports.
        sys.modules[self._canonical] = module
        sys.modules[self._short] = module

        # Set as attribute on parent package.
        if "." in self._canonical:
            parent_cn, attr = self._canonical.rsplit(".", 1)
            if parent_cn in sys.modules:
                setattr(sys.modules[parent_cn], attr, module)

        self._real.exec_module(module)


# -- helpers ----------------------------------------------------------------


def _existing_spec(name, mod):
    """Create a ModuleSpec that returns an already-loaded module."""
    is_pkg = hasattr(mod, "__path__")
    spec = importlib.machinery.ModuleSpec(name, _NoopLoader(mod), is_package=is_pkg)
    if is_pkg:
        spec.submodule_search_locations = list(mod.__path__)
    spec.origin = getattr(mod, "__file__", None)
    return spec


class _NoopLoader(importlib.abc.Loader):
    """Loader that returns an already-loaded module without re-executing."""

    def __init__(self, mod):
        self._mod = mod

    def create_module(self, spec):
        return self._mod

    def exec_module(self, module):
        pass


# -- bootstrap --------------------------------------------------------------

# Create synthetic root package so _ecaj.lib.* / _ecaj.nodes.* can resolve.
_root = ModuleType(_PKG)
_root.__path__ = [str(_ROOT)]
_root.__package__ = _PKG
sys.modules[_PKG] = _root

# Install before all default finders.
sys.meta_path.insert(0, _UnifiedFinder())
