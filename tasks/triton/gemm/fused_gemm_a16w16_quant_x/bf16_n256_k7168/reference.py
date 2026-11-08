"""Float64 reference: y = x @ w^T, and the MXFP8 (E4M3) per-1x32 dynamic quantization of x (codes before the final cast, and E8M0 scale bytes)."""
import torch

_BLOCK = 32
_ROW_CHUNK = 4096


def reference(*, x, w):
    m, k = x.shape
    w64 = w.double()
    y = torch.empty(m, w.shape[0], dtype=torch.float64, device=x.device)
    x_quant = torch.empty(m, k, dtype=torch.float32, device=x.device)
    x_scale = torch.empty(m, k // _BLOCK, dtype=torch.uint8, device=x.device)
    for start in range(0, m, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, m)
        rows64 = x[start:stop].double()
        y[start:stop] = rows64 @ w64.T
        rows = rows64.reshape(-1, k // _BLOCK, _BLOCK)
        amax = rows.abs().amax(dim=-1)
        # Round amax to a power of two by adding a quarter of the mantissa field (mantissa >= 1.75 rounds up),
        # then subtract 8 so that the rounded amax maps to 2**8 (the E4M3 maximum is 448).
        mantissa, exponent = torch.frexp(amax)                     # amax = mantissa * 2**exponent
        exponent = exponent - 1 + (mantissa >= 0.875).to(exponent.dtype)
        unbiased = (exponent - 8).clamp(-127, 127).double()
        unbiased = torch.where(amax == 0, torch.full_like(unbiased, -127.0), unbiased)
        x_scale[start:stop] = (unbiased + 127).to(torch.uint8)
        # Exact in float32 for the rounded values; the evaluator casts it to float8_e4m3fn with round-to-nearest-even.
        x_quant[start:stop] = (rows * torch.exp2(-unbiased)[..., None]).reshape(-1, k).float()
    return {"y": y, "x_quant": x_quant, "x_scale": x_scale}
