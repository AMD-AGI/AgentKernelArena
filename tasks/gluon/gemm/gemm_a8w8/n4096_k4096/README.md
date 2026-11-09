# FP8 GEMM with per-token and per-channel scales (Gluon baseline), N=4096, K=4096, bfloat16 output

`out = (x @ w^T) * (x_scale * w_scale)` with float8_e4m3fn `x` `[m, 4096]` (one float32 scale per row, `x_scale`
`[m, 1]`) and a float8_e4m3fn `[4096, 4096]` weight `w` in the `nn.Linear` layout (one float32 scale per output
channel, `w_scale` `[1, 4096]`). The products are accumulated in float32, multiplied by the outer product of the
scales and rounded to bfloat16. There is no bias. The reference is the same expression in float64 (float8 values and
their products are exact in float64). The operator, interface, inputs and reference are the same as the Triton
scenario `tasks/triton/gemm/gemm_a8w8/n4096_k4096`; only the baseline differs (a Gluon kernel).

## Scenario and workload origin

The layer is the square 4096-wide linear projection of Llama-3.1-8B (the attention `q_proj` and `o_proj`; hidden
size 4096, 32 heads of 128 dimensions). Config citation, Llama-3.1-8B `config.json` (Hugging Face): `hidden_size`
4096, `num_attention_heads` 32 (so `q_proj` is 4096 -> 32 x 128 = 4096 and `o_proj` is 4096 -> 4096), served with
row-wise FP8 quantization. K = 4096 and N = 4096 are the constant dims.

`m` is the number of tokens in the call. Cases span `m = 1 .. 65536`: decode batches and chunks below 512 tokens
are correctness-only; `m = 512 .. 65536` (prefill chunks up to large batched prefills) are performance cases
(128x span, 8 cases). Boundary correctness cases: `m = 1` (axis minimum), `m = 3` (not a power of two), and
`m = 17, 100, 257, 511, 1000, 1025` (not multiples of 16 and above 16; 257 and 511 straddle the 256-row tile of the
baseline).

**Inputs.** `inputs.py` (shared with the Triton scenario) quantizes normally distributed activations and weights
row-wise (scale = row amax / 448, clamped, cast to float8_e4m3fn), as a quantizing producer would, so codes and
scales are consistent. The weight scale is generated as a `[1, n]` row like the AITER interface.

## Baseline

AITER's Gluon `_gemm_a8w8_kernel` (`ROCm/aiter` `aiter/ops/triton/gluon/gemm_a8w8.py`; gfx950 only: MFMA scaled
`mfma_scaled` with the fp8 e4m3 format, direct `buffer_load`, LDS double buffering) with its launcher `gemm_a8w8`.
The Gluon table `configs/gemm/gluon/gfx950-GEMM-A8W8.json` has one entry (256x256x256 tiles, 8 warps, group size 4)
used for every shape, so there is no M dispatch. The kernel is unchanged; the launcher is reduced to the no-bias
float8_e4m3fn path writing into the preallocated output (`NUM_XCDS` is the upstream hardcoded 8); the preshuffled
variant, the int8 and bias launcher branches, the logger and the architecture check (the task already requires
gfx950) are removed. `remap_xcd` and `pid_grid` are `@triton.jit` helpers that the Gluon kernel calls inline (as
upstream does); they are copied unchanged into `baseline/pid_preprocessing.py` and are not launched, so the
baseline launches only Gluon kernels and no split-K reduction kernel is involved (a single kernel per call).

Provenance: `ROCm/aiter` at `652f2d4b768239aefd515c929e84817681ff598d`, MIT. The upstream `gemm_a8w8.py` has no
SPDX or copyright header, so the repository LICENSE (MIT, Copyright (c) Advanced Micro Devices, Inc.) applies, and is
included as `baseline/LICENSE`; `pid_preprocessing.py` keeps its upstream header
(`SPDX-License-Identifier: MIT`, `Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.`). The
history of the file starts with an AITER commit ("gluon gemm_a8w8 and gemm_a8w8_preshuffled"); neither the file nor its
upstream directory records a third-party origin.

## Estimated timing

Roofline estimate (fp8 2e15 flop/s, 4.5e12 B/s, 4 us fixed): `m = 512` is 1.7e10 flop = 8.6 us + 4 us, about 12.6 us;
`m = 65536` is 2.2e12 flop, about 1.1 ms. The real kernel computes whole 256x256 output tiles, so decode-sized `m`
(below 512) would take well over the roofline estimate (tens of microseconds for 16 tiles of K = 4096), but those
counts are not scored.

## Notes

Gluon API: the kernel needs the Gluon `gl.amd.cdna4` namespace of the image's Triton (3.8) and gfx950.
The kernel is deterministic (one K loop per output tile, no split-K, no atomics). The float32 accumulation order
differs between candidates; the `baseline_relative` bound absorbs it. The `triton.jit` pid helpers called from a
Gluon kernel follow the upstream usage.
