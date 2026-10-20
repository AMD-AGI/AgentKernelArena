"""Legal widths must not be accepted from power-of-two-only candidate checks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_swiglustep_and_mul"


def checks():
    spec = importlib.util.spec_from_file_location("_swiglu_width_checks", TASK / "_arena_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(data, limit):
    gate, up = data.float().chunk(2, dim=-1)
    return ((gate * torch.sigmoid(gate)).clamp(max=limit)
            * up.clamp(-limit, limit)).to(data.dtype)


@pytest.mark.parametrize("bad_width", [None, 1, 3, 33, 129, 257, 513, 1025, 2049])
def test_width_specific_wrong_candidate_is_rejected(bad_width):
    seen = set()

    def candidate(data, limit=7.0):
        width = data.shape[-1] // 2
        seen.add(width)
        result = reference(data, limit)
        return torch.zeros_like(result) if width == bad_width else result

    module = SimpleNamespace(swiglustep_and_mul=candidate)
    harness = SimpleNamespace(load_module=lambda: module, reference_swiglustep_and_mul=reference)
    with checks().checked_modules(harness):
        loaded = harness.load_module()
        data = torch.linspace(-20, 20, 3 * 256).reshape(3, 256).half()
        if bad_width is None:
            torch.testing.assert_close(loaded.swiglustep_and_mul(data), reference(data, 7.0))
            assert {1, 3, 33, 129, 257, 513, 1025, 1031, 2049} <= seen
        else:
            with pytest.raises(AssertionError):
                loaded.swiglustep_and_mul(data)
            assert bad_width in seen


def test_controls_preserve_row_stride_and_original_tail_case():
    controls = list(checks().clamp_controls("cpu", torch.float16))
    assert len(controls) == 32
    assert {(x.shape[-1] // 2, limit) for x, limit in controls} >= {(1031, 7.0), (1031, .1), (257, 7.0)}
    for data, limit in controls:
        assert data.shape[0] == 3 and data.stride(0) == 2 * data.shape[-1]
        assert data.stride(1) == 1 and data.dtype == torch.float16
        assert torch.isfinite(reference(data, limit)).all()
