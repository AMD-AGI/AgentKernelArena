# Fused QK RMSNorm + GPT-J RoPE, bfloat16, 128 query heads, head size 512, rotary width 64

For each token the kernel normalizes every 512-wide query head and the single shared key/value row
by their root mean square (epsilon 1e-6), scales the kv row by a per-channel `kv_weight` (queries
have no weight), and applies GPT-J (pair-interleaved) rotary embedding to the last 64 channels with
cos/sin rows gathered by `positions[t]`. The first 448 channels pass through unrotated. This is the
q/kv preparation of DeepSeek-V4-style attention (head size 512, rotary tail 64) with all 128 query
heads on one device. The reference works in float64 and is independent of the baseline.

Baseline: the FlyDSL kernel of AITER (`aiter/ops/flydsl/kernels/qk_norm_rope_quant.py`, MIT),
one wave64 per (token, head) row, restricted to its bfloat16 path. The FP8 quantization, the
optional q_weight and the scale outputs are removed (the operator has no quantized output in this
scenario); the 16-head variant is the sibling scenario `bf16_h16_d512_rd64`. The runtime image's FlyDSL no longer
ships `flydsl.expr.buffer_ops` or `flydsl.expr.vector`, so the used subset of those modules (and
their `meta` helper) is vendored next to the kernel as `buffer_ops.py`, `vector.py` and `meta.py`
from ROCm/FlyDSL commit `28a18d328b4882c999864b2df2f8f9fe3fcc8b47` (Apache-2.0; both licenses are in `baseline/LICENSE`). The kernel body follows the upstream kernel but is written against the FlyDSL helper API of those vendored files rather than aiter's own `buffer_ops`, `GTensor` and `Vector` helpers, which the vendored copies replace.
To satisfy the candidate rules the vendored helpers drop their `os` environment lookup
(`FLYDSL_GPU_ARCH`), the `traceback`-based debug locations, a numpy stride computation and star
imports; kernel arithmetic is unchanged. The kernel launches at most 65535 tokens per grid (hardware
limit on grid Y), so larger batches use several launches. In the MI355X QA audits the baseline compiled,
passed its correctness sanity check on every case and was timed from 256 to 32768 tokens.

Inputs: `q` and `kv` are contiguous `[T, 128, 512]` and `[T, 512]`. Production passes kv as a strided
slice of a wider projection; the kernel supports a row stride and the task declares the dense
layout. cos/sin tables hold 65536 positions of the first-half angles (base 10000 over the rotary
width, rounded to bfloat16); positions are random in the table to exercise the gather.

Workloads: token counts only. Decode batches up to 128 tokens and the unaligned 1000-token chunk are
correctness-only; the kernel's 65535-token launch chunking is exercised by the 16-head sibling. Timed
cases run from 160 to 16384 tokens (102x). The largest case holds 2 GiB per bfloat16 `q`-sized tensor;
the reference processes tokens in chunks to bound its float64 temporaries.

Timing floor: the MI355X baseline medians (`cuda_graph_cold_cache`) are 22.3 and 22.8 us at 256 tokens
(two audits), 36.5 us at 512, 114.9 us at 2048, 448.3 us at 8192 and 1770.4 us at 32768. A least-squares
fit of the relative error gives 8.95 us + 0.0532 us per token (residuals within 2.6%), so the smallest
timed case, 160 tokens, is predicted at 17.5 us, 1.75x the 10 us floor (13.9 us even if the time were
proportional to the token count from the 256-token measurement), and 16384 tokens at about 880 us.
Below 160 tokens the margin over the floor shrinks, so decode batches stay correctness-only.

Largest case size: the evaluator's candidate correctness holds every case's inputs, references and baseline
outputs (two seeds) and, in the `baseline_relative` comparison of `q_out`, thirteen float64 copies of the
output (104 bytes per element); selecting elements with a boolean mask (`errors[mask]`) also builds an int64 index list of 24 bytes
per element. With N = n_tokens * 65536 `q_out` elements, an allocation trace of the evaluator's own code on
the CPU gives 132.2 bytes per element (plus 32 MiB of cos/sin tables) live before the bias test's
selection, and a peak of 164.2 bytes per element (172.2 when the bias test's fallback also runs). The
audits ran out of device memory at 32768 tokens (a 16 GiB request with 274.4 GiB allocated) and at 25600
tokens: a 37.5 GiB request (the index list) with 206.6 GiB allocated, 44.4 GiB cached but unallocated
(fragmentation) and 36.2 GiB free on the 288 GiB MI355X; the trace predicts exactly 206.6 GiB and 37.5 GiB
there. Replaying the traced allocation sequence of all cases through the rules of PyTorch's caching
allocator reproduces that failure (37.5 GiB request, 206.6 GiB allocated, 44.4 GiB cached), and the
32768-token one under the earlier audit's comparator (16 GiB request, 274.4 GiB allocated); it gives the
smallest device budget that completes: 291.5 GiB for the 25600-token workloads (above the 287.2 GiB
available to PyTorch), against 179.3 GiB for these workloads (187.3 GiB with the bias fallback in the
largest case) and 182.3 GiB for the QA controls run, whose float32 calibration probe holds the same
tensors. That is at most 65.3% of the device, including 0.8 GiB used outside PyTorch, so at least 34.7%
headroom. 19200 tokens (with a 192-token smallest case) would need up to 216.1 GiB (75.3%), so 16384
tokens, a common chunked-prefill budget, is the largest timed case and the axis maximum. The bounds,
tolerances and the correctness-only cases are unchanged.

Compare: both outputs use the default `baseline_relative` bound.
