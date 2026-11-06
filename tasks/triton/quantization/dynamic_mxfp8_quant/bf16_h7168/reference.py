"""MXFP8 (E4M3) per-1x32 dynamic quantization reference used only for correctness."""
import torch

_BLOCK = 32
_CHUNK_ROWS = 4096


def reference(*, x):
    m, n = x.shape
    y = torch.empty((m, n), dtype=torch.float32, device=x.device)
    scales = torch.empty((m, n // _BLOCK), dtype=torch.uint8, device=x.device)
    for start in range(0, m, _CHUNK_ROWS):
        rows = x[start:start + _CHUNK_ROWS].double().reshape(-1, n // _BLOCK, _BLOCK)
        amax = rows.abs().amax(dim=-1)
        # Round amax to a power of two by adding a quarter ulp of the mantissa field (mantissa >= 1.75
        # rounds up), then subtract 8 so that the rounded amax maps to 2**8 (E4M3 maximum is 448).
        mantissa, exponent = torch.frexp(amax)                     # amax = mantissa * 2**exponent
        exponent = exponent - 1 + (mantissa >= 0.875).to(exponent.dtype)
        unbiased = (exponent - 8).clamp(-127, 127).double()
        unbiased = torch.where(amax == 0, torch.full_like(unbiased, -127.0), unbiased)
        scales[start:start + _CHUNK_ROWS] = (unbiased + 127).to(torch.uint8)
        # Exact in float32 for the rounded values; the evaluator casts it to float8_e4m3fn with
        # round-to-nearest-even.
        y[start:start + _CHUNK_ROWS] = (rows * torch.exp2(-unbiased)[..., None]).reshape(-1, n).float()
    return {"y": y, "scale": scales}
