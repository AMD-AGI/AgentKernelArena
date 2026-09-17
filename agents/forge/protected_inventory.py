"""Linear protected-file inventory for the reviewed KernelForge release.

The upstream walker re-resolves every exact protected path for every file it
visits. Arena protects complete source snapshots, making that quadratic work
large enough to consume the model's budget before its first call. Resolve the
exact set once per inventory; retain upstream name policy and filesystem rules.
No resolved paths or file contents are cached between inventory calls.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
import sys


def protected_path_inventory(workspace, *, exact_paths=(), extra_globs=()):
    from kernelforge.llm.workspace_policy import is_protected_path

    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise OSError(f"protected inventory workspace is not a directory: {root}")
    globs = tuple(extra_globs)
    exact = set()
    for item in exact_paths:
        if not str(item or "").strip():
            continue
        path = Path(item).expanduser()
        exact.add((path if path.is_absolute() else root / path).resolve())
    # Missing exact paths must stay in the inventory so later creation is
    # detected. Keep both link identities and resolved destinations as upstream
    # does; do not follow directory links during traversal.
    inventory = set(exact)

    def raise_walk_error(error):
        raise error

    for directory, dirnames, filenames in os.walk(
        root, topdown=True, followlinks=False, onerror=raise_walk_error
    ):
        dirnames[:] = [name for name in dirnames if name != ".git"]
        parent = Path(directory)
        entries = list(filenames)
        for dirname in dirnames:
            if stat.S_ISLNK((parent / dirname).lstat().st_mode):
                entries.append(dirname)
        for name in entries:
            candidate = parent / name
            metadata = candidate.lstat()
            if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
                continue
            if candidate.resolve() in exact or is_protected_path(
                candidate, workspace=root, extra_globs=globs
            ):
                inventory.add(Path(os.path.abspath(candidate)))
    return tuple(sorted(inventory, key=str))


def install():
    """Called only after upstream.probe verifies all affected module hashes."""
    from kernelforge.llm import workspace_policy

    original = workspace_policy.protected_path_inventory
    if original is protected_path_inventory:
        return
    # Guard and in-session gate capture direct imports. Update those references
    # as well as the defining module, without modifying the installed package.
    for module in tuple(sys.modules.values()):
        if module and getattr(module, "__name__", "").startswith("kernelforge."):
            if getattr(module, "protected_path_inventory", None) is original:
                module.protected_path_inventory = protected_path_inventory
