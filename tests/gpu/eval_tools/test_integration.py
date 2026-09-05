"""Opt-in smoke tests for real GPU sanitizer installations.

Run with ``AKA_RUN_GPU_EVAL_TOOLS=1``.  rocJITsu additionally requires
``AKA_ROCJITSU_BIN`` and ``AKA_ROCJITSU_CONFIG``.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.eval_tools.probes import PROBE_ROOT


pytestmark = pytest.mark.skipif(
    os.environ.get("AKA_RUN_GPU_EVAL_TOOLS") != "1",
    reason="set AKA_RUN_GPU_EVAL_TOOLS=1 to run GPU integration probes",
)


def run(command, *, env=None):
    return subprocess.run(command, env={**os.environ, **(env or {})}, text=True, capture_output=True, timeout=180)


def test_gpu_asan_safe_and_oob(tmp_path):
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    binary = tmp_path / "gpu_asan_probe"
    compile_result = run(
        [hipcc, "-O2", "-fsanitize=address", "-shared-libsan", "--offload-arch=gfx950:xnack+", str(PROBE_ROOT / "gpu_asan_probe.hip"), "-o", str(binary)]
    )
    assert compile_result.returncode == 0, compile_result.stderr
    env = {"HSA_XNACK": "1", "HSA_DISABLE_FRAGMENT_ALLOCATOR": "1"}
    assert run([str(binary)], env=env).returncode == 0
    oob = run([str(binary), "oob"], env=env)
    assert "AddressSanitizer" in oob.stderr
    assert "buffer-overflow" in oob.stderr


def test_rocjitsu_safe_and_racy(tmp_path):
    binary_path = os.environ.get("AKA_ROCJITSU_BIN")
    config_path = os.environ.get("AKA_ROCJITSU_CONFIG")
    if not binary_path or not config_path:
        pytest.skip("AKA_ROCJITSU_BIN and AKA_ROCJITSU_CONFIG are required")
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    probe = tmp_path / "race_probe"
    compiled = run([hipcc, "-O2", "--offload-arch=gfx950", str(PROBE_ROOT / "rocjitsu_race_probe.hip"), "-o", str(probe)])
    assert compiled.returncode == 0, compiled.stderr
    env = {"RJ_RACE": "1", "RJ_LOG": "1"}
    safe = run([binary_path, "--config", config_path, "--", str(probe)], env=env)
    assert "RACE type=" not in safe.stdout + safe.stderr
    racy = run([binary_path, "--config", config_path, "--", str(probe), "racy"], env=env)
    assert "RACE type=LDS" in racy.stdout + racy.stderr
