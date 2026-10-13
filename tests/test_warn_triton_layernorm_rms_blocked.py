"""Side-output and blocked-stride controls for scored ROCmBench paths."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1] / "tasks"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure_times(*args, **kwargs):
    raise AssertionError("The adapter did not bind the measured-output collector")




@pytest.mark.parametrize("fault", (None, "ignore_input_stride", "ignore_output_stride", "skip_rsigma"))
def test_blocked_rmsnorm_full_width_checks_independent_row_strides(fault):
    task = ROOT / "triton2triton/rocmbench/medium/rmsnorm_fwd"
    reference = load(task / "_arena_reference.py", "_blocked_rms_reference")
    cols = 65536
    x = torch.randn((2, cols), dtype=torch.float16) * .1
    g = torch.rand(cols, dtype=torch.float16)
    context = {
        "x": x, "g": g, "y_buffer": torch.empty_like(x),
        "USE_BLOCKED_fwd": True, "blk_size_fwd": 32768,
        "NUM_PRGMS_fwd": 2, "ZERO_CENTERED_GAMMA": False, "eps": 1e-5,
    }
    seen = []

    def operator(x, g, y, rsigma, *args):
        rows, width, centered, block_size, blocked, programs, eps = args[-7:]
        seen.append((rows, width, block_size, blocked, x.stride(0), y.stride(0)))
        assert blocked is True and width == cols and rows == 3
        xf = x.float()
        if fault == "ignore_input_stride":
            xf = torch.as_strided(x, x.shape, (cols, 1)).float()
        expected_rs = torch.rsqrt((xf * xf).mean(-1) + eps)
        expected_y = (xf * expected_rs[:, None] * g.float()).to(y.dtype)
        if fault == "ignore_output_stride":
            torch.as_strided(y, y.shape, (cols, 1)).copy_(expected_y)
        else:
            y.copy_(expected_y)
        if fault != "skip_rsigma":
            rsigma.copy_(expected_rs)

    module = SimpleNamespace(rmsnorm=operator)
    if fault is None:
        reference.check_blocked_stride_control(context, module)
        assert len(seen) == 1
        assert seen[0][4] == cols + 16 and seen[0][5] == cols + 32
    else:
        with pytest.raises((AssertionError, ValueError)):
            reference.check_blocked_stride_control(context, module)
    context["USE_BLOCKED_fwd"] = False
    seen.clear()
    reference.check_blocked_stride_control(context, module)
    assert seen == []
