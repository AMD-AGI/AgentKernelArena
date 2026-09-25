"""Resolve an explicitly provisioned Apex checkout; never fetch at run time."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


APEX_REVISION = "c9f5765b10959bc30a80e10455789f5f0158e0ca"


def runtime_environment() -> dict[str, str]:
    value = os.environ.get("APEX_ROOT")
    if not value:
        raise ValueError("Set APEX_ROOT to the pinned recovery-integration checkout; see agents/apex/README.md")
    root = Path(value).resolve(strict=True)
    # The explicitly supplied checkout is read-only in Docker and may be owned
    # by the host user even when the caller selects a root container.
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
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise ValueError("Apex requires bubblewrap in the GPU image; see agents/apex/README.md")
    help_text = subprocess.run([bwrap, "--help"], capture_output=True, text=True,
                               check=True, timeout=10).stdout
    if not all(flag in help_text for flag in ("--json-status-fd", "--block-fd", "--unshare-pid")):
        raise ValueError("The GPU image's bubblewrap lacks required Apex containment capabilities")
    environment = runtime_environment()
    probe = """
import os
from pathlib import Path
from apex.bootstrap import build_application
from apex.execution import SubprocessSupervisor
import mcp
result = SubprocessSupervisor().run(
    ['/bin/true'], cwd=Path.cwd(), environment=os.environ,
    timeout_seconds=10, require_pid_namespace=True,
)
if result.exit_code != 0 or not result.cleanup_succeeded:
    raise RuntimeError('Apex process namespace preflight failed')
"""
    if gpu:
        probe += """
import torch
from apex.runtime.gpu import resolve_gpu_device_scope
from apex.runtime.gpu_ownership import RocmSmiGpuOwnershipInspector
# Launch a real GPU operation, then retain its allocation and context. Merely
# reserving memory need not create a KFD queue visible to the ownership API.
allocation = torch.ones(1, device='cuda')
torch.cuda.synchronize()
try:
    receipt = RocmSmiGpuOwnershipInspector().inspect(
        resolve_gpu_device_scope(), allowed_pids=(os.getpid(),),
    )
    if not any(owner.pid == os.getpid() for owner in receipt.allowed_owners):
        raise RuntimeError('KFD inventory did not identify the GPU preflight process')
except Exception as error:
    raise RuntimeError(
        'Apex GPU ownership preflight failed; the container must resolve KFD '
        'process identities. See agents/apex/README.md for host requirements.'
    ) from error
"""
    subprocess.run([sys.executable, "-c", probe], env=environment, check=True, timeout=60)
    print(f"Apex recovery-integration revision {APEX_REVISION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Verify live GPU process ownership")
    check_runtime(gpu=parser.parse_args().gpu)
