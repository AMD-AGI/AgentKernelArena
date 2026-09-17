"""Align the reviewed in-session gate with its existing SDK target policy.

Task-declared implementation files may have test-like names or live alongside
their harness. Explicit protected paths still win; Arena's assessment bridge
checks symbol-scoped statements before it executes any candidate.
"""
from pathlib import Path


def install():
    """Install only after the upstream source/signature probe has passed."""
    from kernelforge.loop import insession_gate

    original = insession_gate.InSessionGate
    if getattr(original, "_arena_target_policy", False):
        return

    class ArenaInSessionGate(original):
        _arena_target_policy = True

        def _target_exempt(self):
            explicit = {Path(path).resolve() for path in self.protected_abs}
            return {Path(path).resolve() for path in self.target_abs} - explicit

        def _is_protected(self, fp):
            if fp:
                path = Path(fp)
                if not path.is_absolute() and self.workspace_root:
                    path = Path(self.workspace_root) / path
                if path.resolve() in self._target_exempt():
                    return False
            return super()._is_protected(fp)

        def _iter_snapshot_paths(self):
            exempt = self._target_exempt()
            return [path for path in super()._iter_snapshot_paths()
                    if path.resolve() not in exempt]

    insession_gate.InSessionGate = ArenaInSessionGate
