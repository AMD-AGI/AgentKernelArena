"""Isolated image-owned entry point for trusted rocJITsu capsule replay."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


_ENTRYPOINT = Path(__file__).resolve(strict=True)
_FRAMEWORK_ROOT = _ENTRYPOINT.parents[3]
_EXPECTED_HELPER = (
    _FRAMEWORK_ROOT / "src" / "eval_tools" / "adapters" / "rocjitsu_replay.py"
).resolve(strict=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Load the replay implementation only from the entry point's image tree."""

    # The plugin invokes this file with Python ``-I``.  Add only the framework
    # tree containing this absolute entry point; the candidate workspace and
    # candidate-controlled PYTHONPATH must never participate in module lookup.
    sys.path.insert(0, str(_FRAMEWORK_ROOT))
    from src.eval_tools.adapters import rocjitsu_replay

    helper_path = Path(rocjitsu_replay.__file__).resolve(strict=True)
    if helper_path != _EXPECTED_HELPER:
        raise RuntimeError(
            "rocJITsu replay helper resolved outside the image-owned framework: "
            f"{helper_path} != {_EXPECTED_HELPER}"
        )
    return rocjitsu_replay.main(argv)


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests.
    raise SystemExit(main())
