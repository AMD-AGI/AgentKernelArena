"""MXFP4 (E2M1) per-1x32 dynamic quantization reference used only for correctness."""
import torch

_BLOCK = 32
_CHUNK_ROWS = 4096


def _e2m1_magnitude_code(magnitude):
    """Round-to-nearest-even onto the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}; ties go to the even code."""
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    # (threshold, inclusive): the code steps up once the magnitude passes the threshold.
    for threshold, inclusive in ((0.25, False), (0.75, True), (1.25, False), (1.75, True),
                                 (2.5, False), (3.5, True), (5.0, False)):
        code += ((magnitude >= threshold) if inclusive else (magnitude > threshold)).to(torch.uint8)
    return code


def reference(*, x):
    m, n = x.shape
    packed = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((m, n // _BLOCK), dtype=torch.uint8, device=x.device)
    for start in range(0, m, _CHUNK_ROWS):
        rows = x[start:start + _CHUNK_ROWS].double().reshape(-1, n // _BLOCK, _BLOCK)
        amax = rows.abs().amax(dim=-1)
        # "even" scale rule: round amax to a power of two by adding a quarter ulp of the mantissa
        # field (mantissa >= 1.75 rounds up), then subtract the E2M1 maximum exponent (2).
        mantissa, exponent = torch.frexp(amax)                     # amax = mantissa * 2**exponent
        exponent = exponent - 1 + (mantissa >= 0.875).to(exponent.dtype)
        unbiased = (exponent - 2).clamp(-127, 127).double()
        unbiased = torch.where(amax == 0, torch.full_like(unbiased, -127.0), unbiased)
        scales[start:start + _CHUNK_ROWS] = (unbiased + 127).to(torch.uint8)
        scaled = rows * torch.exp2(-unbiased)[..., None]            # exact: a power-of-two multiple
        code = _e2m1_magnitude_code(scaled.abs()) | (torch.signbit(scaled).to(torch.uint8) << 3)
        code = code.reshape(-1, n)
        packed[start:start + _CHUNK_ROWS] = code[:, 0::2] | (code[:, 1::2] << 4)
    return {"x_fp4": packed, "blockscale_e8m0": scales}
