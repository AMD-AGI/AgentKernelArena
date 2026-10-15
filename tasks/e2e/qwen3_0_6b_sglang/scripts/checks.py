"""Task-owned RMSNorm numerical oracle, independent of candidate kernels."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def source_policy():
    """Check the backend-neutral compute bundle; this is not a security sandbox."""
    import json
    from _aka_serving_bundle import Bundle
    lock = json.loads((ROOT / 'runtime.lock.json').read_text())
    return Bundle(ROOT, lock).check_sources()


def operator_checks(lock):
    source_policy()
    import torch
    torch.manual_seed(lock['correctness']['seed'])
    from types import SimpleNamespace
    from _aka_serving_bundle import effective_function
    module = SimpleNamespace(**{name: effective_function(ROOT, lock,
        'sglang.srt.layers.layernorm', name) for name in ('rms_norm', 'fused_add_rms_norm')})
    evidence = []
    for rows in (1, 7, 8, 64, 128, 1024, 2048, 16384):
        for width in (128, 1024):
            x = torch.randn((rows, width), device='cuda', dtype=torch.float16)
            residual = torch.randn_like(x)
            weight = torch.randn(width, device='cuda', dtype=x.dtype)
            for eps in (1e-6, 1e-5):
                def reference(value):
                    value = value.float()
                    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
                original, wcopy = x.clone(), weight.clone()
                expected = reference(x)
                actual = module.rms_norm(x, weight, eps)
                torch.testing.assert_close(weight, wcopy, rtol=0, atol=0)
                torch.testing.assert_close(actual, expected, rtol=lock['correctness']['rtol'], atol=lock['correctness']['atol'])
                torch.testing.assert_close(x, original, rtol=0, atol=0)
                output, added = torch.empty_like(x), torch.empty_like(x)
                rcopy = residual.clone()
                # Construct the oracle before calling editable code.
                expected_added = (x.float() + residual.float()).to(x.dtype)
                expected_norm = reference(expected_added)
                module.fused_add_rms_norm(output, x, residual, added, weight, eps)
                torch.testing.assert_close(weight, wcopy, rtol=0, atol=0)
                torch.testing.assert_close(added, expected_added, rtol=0, atol=0)
                torch.testing.assert_close(output, expected_norm, rtol=lock['correctness']['rtol'], atol=lock['correctness']['atol'])
                torch.testing.assert_close(residual, rcopy, rtol=0, atol=0)
                torch.testing.assert_close(x, original, rtol=0, atol=0)
                evidence.append(dict(shape=[rows, width], eps=eps))
    torch.cuda.synchronize()
    return evidence
