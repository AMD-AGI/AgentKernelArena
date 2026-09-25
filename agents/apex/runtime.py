"""Resolve an explicitly provisioned Apex checkout; never fetch at run time."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


APEX_REVISION = "cbe5f9965f4f900468cfce94db735ceb33fa1dd8"


def runtime_environment() -> dict[str, str]:
    value = os.environ.get("APEX_ROOT")
    if not value:
        raise ValueError("Set APEX_ROOT to the pinned recovery-integration checkout; see agents/apex/README.md")
    root = Path(value).resolve(strict=True)
    # The explicitly supplied checkout is read-only in Docker.
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    revision = subprocess.run([*git, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    if revision != APEX_REVISION:
        raise ValueError("Apex checkout does not match the recovery-integration pin in agents/apex/runtime.py")
    changed = subprocess.run([*git, "status", "--porcelain", "--untracked-files=all"],
                             capture_output=True, text=True, check=True, timeout=10).stdout
    if changed.strip() or not (root / "src/apex/cli/__main__.py").is_file():
        raise ValueError("Apex requires a clean pinned source checkout")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def check_runtime(*, gpu: bool = False) -> None:
    environment = runtime_environment()
    probe = """
import os
from pathlib import Path
from apex.bootstrap import build_application
from apex.execution import SubprocessSupervisor
import mcp
result = SubprocessSupervisor().run(
    ['/bin/true'], cwd=Path.cwd(), environment=os.environ,
    timeout_seconds=10,
)
if result.exit_code != 0 or not result.cleanup_succeeded:
    raise RuntimeError('Apex process preflight failed')
"""
    if gpu:
        probe += """
from apex.runtime.assigned_gpu import probe_assigned_gpus
probe_assigned_gpus()
"""
    subprocess.run([sys.executable, "-c", probe], env=environment, check=True, timeout=60)
    print(f"Apex recovery-integration revision {APEX_REVISION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Run a small operation on assigned GPUs")
    check_runtime(gpu=parser.parse_args().gpu)
