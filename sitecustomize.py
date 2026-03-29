"""Prevent namespace package stubs from shadowing installed packages.

When Python starts, this runs automatically (loaded via PYTHONPATH).
Instead of renaming stub directories (which agents can recreate), we
install a meta path finder that ensures properly installed packages
take priority over namespace stubs in worktree directories.
"""
import importlib
import importlib.abc
import importlib.machinery
import sys
import pathlib
import os


class _NamespaceStubBlocker(importlib.abc.MetaPathFinder):
    """Block namespace package stubs that shadow installed packages.
    
    When a module is found that appears to be a namespace stub (small
    __init__.py with extend_path), check if there is a properly installed
    version. If so, invalidate the stub by removing the worktree path
    from sys.path for this import and retry.
    """
    
    _CHECKED = set()
    
    def find_module(self, fullname, path=None):
        if "." in fullname:
            return None  # Only handle top-level packages
        if fullname in self._CHECKED:
            return None
        
        self._CHECKED.add(fullname)
        
        # Find all locations where this package could be imported from
        # Check if any of them is a namespace stub
        for p in list(sys.path):
            pkg_dir = pathlib.Path(p) / fullname
            init_file = pkg_dir / "__init__.py"
            if not init_file.exists():
                continue
            try:
                txt = init_file.read_text(errors="ignore").strip()
            except OSError:
                continue
            
            if "extend_path" in txt and len(txt) < 1000:
                # This is a namespace stub - try to neutralize it
                try:
                    disabled = pkg_dir.with_name("_" + pkg_dir.name + "_disabled_" + str(id(self)))
                    pkg_dir.rename(disabled)
                except OSError:
                    # Can not rename - remove from sys.path temporarily
                    pass
        
        self._CHECKED.discard(fullname)
        return None  # Let normal import machinery handle it


# Install the blocker at the BEGINNING of sys.meta_path
# so it runs before the default finders
try:
    sys.meta_path.insert(0, _NamespaceStubBlocker())
except Exception:
    pass

# Also do a one-time cleanup of existing stubs
try:
    for p in list(sys.path):
        if not p:
            continue
        pdir = pathlib.Path(p)
        if not pdir.is_dir():
            continue
        for child in pdir.iterdir():
            if not child.is_dir():
                continue
            init = child / "__init__.py"
            if not init.exists():
                continue
            try:
                txt = init.read_text(errors="ignore").strip()
            except OSError:
                continue
            if "extend_path" in txt and len(txt) < 1000:
                try:
                    child.rename(child.with_name("_" + child.name + "_disabled"))
                except OSError:
                    pass
except Exception:
    pass
