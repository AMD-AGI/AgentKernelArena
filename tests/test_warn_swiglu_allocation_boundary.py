"""A candidate must not hoist allocation out of the measured public wrapper."""
from pathlib import Path
import shutil

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_swiglustep_and_mul"


def workspace(tmp_path):
    root = tmp_path / "task"
    shutil.copytree(TASK, root, ignore=shutil.ignore_patterns("__pycache__"))
    return root, root / "source/triton_swiglustep_and_mul.py"


def test_cached_output_wrapper_is_rejected(tmp_path):
    root, source = workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    original = source.read_text()
    allocation = "    output = torch.empty((b, d), dtype=input.dtype, device=input.device)"
    assert allocation in original
    source.write_text(original.replace(allocation, "\n".join([
        "    key = (b, d, input.dtype, input.device)",
        "    if key not in _output_cache:",
        "        _output_cache[key] = torch.empty((b, d), dtype=input.dtype, device=input.device)",
        "    output = _output_cache[key]",
    ])) + "\n_output_cache = {}\n")
    with pytest.raises(RuntimeError, match="triton_swiglustep_and_mul.py"):
        verify_workspace_harness(before)


def test_kernel_change_and_ordinary_helper_remain_editable(tmp_path):
    root, source = workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    original = source.read_text()
    assert "result = gate_clamped * up_clamped" in original
    source.write_text(original.replace("result = gate_clamped * up_clamped",
                                       "result = up_clamped * gate_clamped")
                      + "\ndef implementation_helper(value):\n    return value\n")
    verify_workspace_harness(before)
