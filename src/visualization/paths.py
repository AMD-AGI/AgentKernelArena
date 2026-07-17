"""Filesystem locations used by the visualization module."""

from __future__ import annotations

import os
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parent


def _configured_path(environment_name: str, default: Path) -> Path:
    configured = os.environ.get(environment_name)
    return Path(configured).expanduser().resolve() if configured else default.resolve()


PROJECT_ROOT = _configured_path("AKA_PROJECT_ROOT", MODULE_ROOT.parents[1])
RUNTIME_ROOT = _configured_path(
    "AKA_VISUALIZATION_RUNTIME_ROOT", PROJECT_ROOT / ".visualization"
)
FRONTEND_ROOT = MODULE_ROOT / "frontend"
DATA_ROOT = RUNTIME_ROOT / "dashboard"
REPORTS_ROOT = RUNTIME_ROOT / "reports"
