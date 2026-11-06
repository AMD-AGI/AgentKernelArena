# Fused QK RMSNorm + GPT-J RoPE, bfloat16, 16 query heads, head size 512, rotary width 64

For each token the kernel normalizes every 512-wide query head and the single shared key/value row
by their root mean square (epsilon 1e-6), scales the kv row by a per-channel `kv_weight` (queries
have no weight), and applies GPT-J (pair-interleaved) rotary embedding to the last 64 channels with
cos/sin rows gathered by `positions[t]`. The first 448 channels pass through unrotated. This is the
q/kv preparation of DeepSeek-V4-style attention (head size 512, rotary tail 64) with 16 query heads
per device. The reference works in float64 and is independent of the baseline.

Baseline: the FlyDSL kernel of AITER (`aiter/ops/flydsl/kernels/qk_norm_rope_quant.py`, MIT),
one wave64 per (token, head) row, restricted to its bfloat16 path. The FP8 quantization, the
optional q_weight and the scale outputs are removed (the operator has no quantized output in this
scenario); the 128-head variant is the sibling scenario `bf16_h128_d512_rd64`. The runtime image's FlyDSL no longer
ships `flydsl.expr.buffer_ops` or `flydsl.expr.vector`, so the used subset of those modules (and
their `meta` helper) is vendored next to the kernel as `buffer_ops.py`, `vector.py` and `meta.py`
from ROCm/FlyDSL commit `28a18d328b4882c999864b2df2f8f9fe3fcc8b47` (Apache-2.0; both licenses are in `baseline/LICENSE`). The kernel body follows the upstream kernel but is written against the FlyDSL helper API of those vendored files rather than aiter's own `buffer_ops`, `GTensor` and `Vector` helpers, which the vendored copies replace.
To satisfy the candidate rules the vendored helpers drop their `os` environment lookup
(`FLYDSL_GPU_ARCH`), the `traceback`-based debug locations, a numpy stride computation and star
imports; kernel arithmetic is unchanged. The kernel launches at most 65535 tokens per grid (hardware
limit on grid Y), so the largest cases use several launches. These changes have been checked on a CPU
only for syntax and the static candidate rules; GPU validation is pending.

Inputs: `q` and `kv` are contiguous `[T, 16, 512]` and `[T, 512]`. Production passes kv as a strided
slice of a wider projection; the kernel supports a row stride and the task declares the dense
layout. cos/sin tables hold 65536 positions of the first-half angles (base 10000 over the rotary
width, rounded to bfloat16); positions are random in the table to exercise the gather.

Workloads: decode batches and short prefills below about 1024 tokens (about 34 KB of traffic per
token against the 10 us timing floor) are correctness-only, as are 1000 tokens (unaligned) and
65537 tokens (crosses the 65535-token launch chunk). Timed cases run from 1024 to 131072 tokens.
The largest case needs about 2 GB per bfloat16 tensor; the reference processes tokens in chunks to
bound its float64 temporaries.

Compare: both outputs use the default `baseline_relative` bound.
