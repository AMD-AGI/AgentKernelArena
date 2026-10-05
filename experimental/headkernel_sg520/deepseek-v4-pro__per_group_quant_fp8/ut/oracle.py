"""Independent PyTorch FP8 quantization oracle for the observed frozen ABI."""

import json
from pathlib import Path
import re


def load_cases(path, allow_provisional=False):
    manifest = json.loads(Path(path).read_text())
    if not manifest.get("confirmed_corrected_1024") and not allow_provisional:
        raise ValueError("corrected 1024-token trace confirmation is pending; provisional checks require explicit opt-in")
    if manifest.get("confirmed_corrected_1024") and not re.fullmatch(r"[a-f0-9]{64}", manifest.get("corrected_trace_sha256") or ""):
        raise ValueError("corrected trace confirmation requires its SHA-256")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("no observed cases")
    for case in cases:
        fixed = {"input_dtype": "bfloat16", "quant_dtype": "float8_e4m3fn", "group_size": 128,
                 "transpose_scale": True, "scale": None, "num_rows": None,
                 "num_rows_factor": 1, "scale_type": "float32", "scale_ub_at_native_call": None}
        if any(case.get(key) != value for key, value in fixed.items()):
            raise ValueError("case differs from the traced native ABI")
        m, n = case["shape"]
        if type(m) is not int or type(n) is not int or min(m, n) <= 0 or n % 128:
            raise ValueError("invalid observed shape")
        if case["input_strides"] != [n, 1] or not case.get("sample_event_evidence"):
            raise ValueError("contiguous input layout and source event evidence required")
    return manifest, cases


def generate(case, seed, device):
    import torch

    m, n = case["shape"]
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, n), generator=generator, device=device, dtype=torch.float32)
    groups = x.reshape(m, n // 128, 128)
    exponent = torch.randint(-8, 9, (m, n // 128, 1), generator=generator, device=device)
    groups.mul_(torch.exp2(exponent.float()))
    # Legal finite corner groups at the real full shape: the zero-scale floor,
    # positive/negative constants, and a small-amplitude group below that floor.
    flat = groups.reshape(-1, 128)
    flat[0].zero_()
    if flat.shape[0] > 1:
        flat[1].fill_(1.0)
    if flat.shape[0] > 2:
        flat[2].fill_(-1.0)
    if flat.shape[0] > 3:
        flat[3].fill_(1e-12)
    return x.to(torch.bfloat16)


def invoke(callable_, x):
    import torch

    return callable_(x, scale=None, quant_dtype=torch.float8_e4m3fn,
                     group_size=128, transpose_scale=True, num_rows=None,
                     num_rows_factor=1, scale_type=torch.float32)


def reference(x):
    import torch

    m, n = x.shape
    groups = x.float().reshape(m, n // 128, 128)
    amax = groups.abs().amax(dim=-1).clamp_min(1e-10)
    scale = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=x.device)
    normalized = (groups * scale.reciprocal().unsqueeze(-1)).clamp(-448.0, 448.0)
    quantized = normalized.to(torch.float8_e4m3fn).reshape(m, n)
    # Native shuffle_scale=True transposes storage, not the returned Tensor's
    # shape or strides. Its public view remains contiguous [M, N/128].
    physical_scale = scale.t().contiguous().reshape(m, n // 128)
    return quantized, physical_scale


def compare(actual, expected, x):
    import torch

    if not isinstance(actual, tuple) or len(actual) != 2:
        raise AssertionError("expected (quantized, scales)")
    for value, reference_value in zip(actual, expected):
        if value.shape != reference_value.shape or value.dtype != reference_value.dtype:
            raise AssertionError("output shape/dtype differs")
        if value.device != x.device or value.stride() != reference_value.stride():
            raise AssertionError("output device/physical layout differs")
        if value.data_ptr() == x.data_ptr():
            raise AssertionError("output aliases the input")
    quantized, scale = actual
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise AssertionError("invalid dynamic scale")
    torch.testing.assert_close(scale, expected[1], rtol=2e-7, atol=0)
    if not torch.equal(quantized.view(torch.uint8), expected[0].view(torch.uint8)):
        raise AssertionError("FP8 encoding differs from the independent reference")
