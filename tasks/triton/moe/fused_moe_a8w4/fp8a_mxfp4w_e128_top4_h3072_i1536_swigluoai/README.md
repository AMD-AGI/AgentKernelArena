# GPT-OSS MoE, static-FP8 tokens x MXFP4 experts (W4A8), 128 experts, top-4, hidden 3072, intermediate 1536

The routed-expert block of GPT-OSS-120B on one tensor-parallel-2 rank, in the W4A8 form vLLM runs on ROCm: the
bfloat16 hidden states are quantized to FP8 with a static (calibrated) per-tensor scale, the expert weights are
MXFP4, and the SwiGLU output is requantized to FP8 with a second static scale before the down projection. Each token
is sent to 4 of 128 experts (`topk_ids`, with router weights `topk_weights`). For token `t` and slot `j` (expert `e`):

```
x_q    = float8_e4m3fn(x[t] / w1_input_scale)
g      = (x_q @ W1[e]^T) * w1_input_scale + w1_bias[e]       # 3072 outputs: gate = g[:1536], up = g[1536:]
act    = min(gate, 7) * sigmoid(1.702 * min(gate, 7)) * (clamp(up, -7, 7) + 1)
a_q    = float8_e4m3fn(act / w2_input_scale)
y      = (a_q @ W2[e]^T) * w2_input_scale + w2_bias[e]
out[t] = sum_j topk_weights[t, j] * y                         # bfloat16
```

`W1` [128, 3072, 3072] and `W2` [128, 3072, 1536] are MXFP4: e2m1 codes packed two per byte along the contracted
axis (low nibble first) and one e8m0 scale byte, `2^(byte - 127)`, per 32 contracted elements. The router weights
multiply the down-projection output (after its bias), as vLLM does when the router weight is not applied on the input.

## Scenario and config citations

GPT-OSS-120B `config.json` (Hugging Face): `hidden_size` 2880, `intermediate_size` 2880, `num_local_experts` 128,
`num_experts_per_tok` 4, `swiglu_limit` 7.0. The SwiGLU `alpha` 1.702 is the constant of the model's activation
(`triton_kernel_fused_experts` and `triton_kernel_fused_mxfp4_w4a8_experts` in vLLM default to alpha 1.702, limit 7.0).

- `inter_size` 1536: the intermediate width 2880 split over tensor-parallel size 2 is 1440, which vLLM's ROCm MXFP4
  path rounds up to a multiple of 256 (`mxfp4_round_up_hidden_size_and_intermediate_size` in
  `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, `current_platform.is_rocm()` branch).
- `hidden_size` 3072: 2880 rounded up the same way.
- `num_experts` 128, `topk` 4: `num_local_experts`, `num_experts_per_tok`.

Tensor-parallel size 2 is the largest size at which vLLM keeps the CDNA4 swizzled MX-scale layout on gfx950
(`should_use_cdna4_mx_scale_swizzle`: `on_gfx950() and tp_size <= 2`; its comment explains that larger sizes make the
A8W4 dispatch pick K tiles below 256, which the swizzle cannot express). Both GEMMs of this scenario (N=3072/K=3072 and
N=3072/K=1536) have 256-wide K tiles in AITER's gfx950 A8W4 table, so the swizzled layout is valid.

Padding: in production the padded rows and columns are zero and the block-m-16 (decode) launches skip them through
`unpadded_N`/`unpadded_K`. This task uses the padded sizes as the problem sizes with full random weights, so every
padded element is real work and the unpadded arguments are not needed.

## Inputs and layout

`inputs.py` draws uniform random e2m1 code bytes and e8m0 scale bytes in a few binades (2^-8..2^-5 for `W1`,
2^-10..2^-7 for `W2`), so that unit-normal tokens give gate/up values with a standard deviation of about 3: the
SwiGLU clamp at 7 is active for about 1% of the gate values and 2% of the up values, and the output has a standard
deviation of about 1.7 (measured on the reference at 256 tokens). The expert biases are synthetic (normal, standard
deviation 0.25 for `W1` and 0.1 for `W2`; GPT-OSS has expert biases, their values are not part of any config).

The two static scales are fixed calibration constants (vLLM takes them from the checkpoint's `w13_input_scale` and
`w2_input_scale`, reduced to their maximum): 0.015 for the hidden states and 0.15 for the SwiGLU output. They keep the
largest hidden-state value of the largest case (about 5.8 sigma, 5.8 / 0.015 = 387) and the largest possible SwiGLU
output (7 * 8 = 56, 56 / 0.15 = 373) inside the float8_e4m3fn range (448), so no element saturates. About 2% of the
SwiGLU outputs fall below the e4m3 normal range after scaling, which exercises its subnormals.

Routing takes the top-4 of unit-normal logits (distinct experts) with softmax weights, plus a fixed per-expert offset
(standard deviation 0.5) so that expert loads are uneven (at 256 tokens an expert receives between 0 and 72 of the
1024 slots; mean 8; 116 of 128 experts are used).

The logical weights (what `reference.py` reads) keep the AITER layout: `W1` rows 0..1535 are the gate projection and
rows 1536..3071 the up projection; scales are `[E, N, K/32]`. The kernel-facing layout (`to_kernel_layout`) is the weight
preparation this Triton kernel reads on gfx950, done once at load time in production:

- `w1`, its scales and `w1_bias` interleave gate and up rows (row `2i` = gate `i`, row `2i+1` = up `i`), which the fused
  SwiGLU epilogue splits as even/odd columns (GPT-OSS checkpoints store these rows interleaved);
- `w1_scale` [128, 3072, 96] and `w2_scale` [128, 1536, 96] are the scales permuted into the gfx950 (CDNA4) MX-scale tile
  order of AITER's `shuffle_scale_moe` (32-row stripes of N, 8-wide groups of K/32), the layout the kernel reads with
  `SWIZZLE_MX_SCALE="CDNA4_SCALE"`; `unswizzle_mx_scale_cdna4` in `baseline/moe_gemm_a8w4.py` is its inverse inside each
  tile.

`w1` and `w2` keep their `[E, N, K/2]` shape; the baseline views them as column-major `[E, K/2, N]`.

## Reference

`reference.py` models both FP8 quantizations exactly (a float32 division by the static scale, then round to nearest
even into float8_e4m3fn), dequantizes each routed expert's MXFP4 weights in float64 and evaluates the GEMMs (the
products of FP8 codes and e2m1 values are exact in float64), the biases, the clamped SwiGLU and the weighted top-k sum
in float64 with no other intermediate rounding. The baseline stores the per-slot weighted GEMM2 rows in bfloat16 before
the final sum; that rounding is part of the baseline's error. Because the reference reproduces the operator's FP8
activations, the baseline's error against it is dominated by bfloat16 rounding and float32 accumulation order, and the
`baseline_relative` bound (k = 2) is tight: an implementation must reproduce the same FP8 activation values
(division by the scale, not multiplication by its reciprocal; SwiGLU evaluated from the float32 accumulator, not from a
bfloat16-rounded value), up to the rare element that lies within float32 noise of an FP8 rounding boundary.

### Bound override: `max_violation_fraction: 0.0005` on `out`

The GPU calibration probe evaluates `reference.py` with float32 accumulation and the default bound (no violations
allowed) failed on 5 cases (for example t1024: violation fraction 7.6e-5, peak 2.99 bounds). Mechanism: the
SwiGLU output is requantized to FP8 between the GEMMs. A float32 GEMM1 differs from the float64 one by accumulation-order
noise of a few 1e-7 relative, so an activation that lies that close to an e4m3 rounding boundary rounds to the
neighbouring code, a step of 6-12% of that element. One such flip shifts every output element of its token slot by about
2^-3 * |a| * |w| / sqrt(1536) of the output scale, a few times the baseline's own (bfloat16-rounding-sized) error, so a
sparse set of elements exceeds `2 * max(|baseline error|, row RMS) + u * |y|` by a small factor. The bound
otherwise stays at k = 2 and the 16x peak cap still applies.

CPU estimate (this task's inputs, 1024 tokens, an emulated baseline with blocked float32 GEMM1 and bfloat16 per-slot
rows against a candidate whose GEMM1 carries independent relative noise): noise 1e-6 gives a violation fraction of
4.7e-4 (peak 4.4 bounds), 1e-5 gives 5.1e-3 (peak 9.4), i.e. the fraction scales with the noise level; the observed
7.6e-5 corresponds to about 2e-7, the float32 level. The fraction is a per-element rate and does not grow with the token
count. 0.0005 leaves about 6x margin over the observed 7.6e-5 and equals
the emulated rate at 1e-6 noise, five times the float32 level; it is far below the 1e-5-noise rate, so an implementation
whose GEMM1 is materially less accurate than float32 (for example TF32 or bfloat16 inputs, noise 5e-4 or more, fractions
above 10%) is still rejected.

The override does not admit wrong outputs: elements over the bound must stay within 16 bounds, and the control
perturbations land far from it. Emulated on this task's outputs (256 tokens): `shifted_16_ulp` (bfloat16 output scaled by
1 + 16 u) violates 93% of the elements, peak 13; `fp8_rounded` (output rounded to e4m3) violates 62%; `half_written`
violates 50% with peak 200; zeros violate 99.6%, peak 205; `noop` (poisoned NaN output) is non-finite. A skipped
or zeroed token row is under 5e-4 of the elements only in batches above 2000 tokens, and its error is hundreds of
bounds, so it fails the peak cap.

## Baseline

AITER's Triton MXFP4-weight, FP8-activation MoE GEMM, `aiter/ops/triton/moe/moe_op_gemm_a8w4.py` and
`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py` (`ROCm/aiter` at commit
652f2d4b768239aefd515c929e84817681ff598d, MIT), with AITER's routing kernels and vLLM's `pack_bitmatrix`. All of it is
Triton:

1. `_downcast_to_static_fp8` quantizes the hidden states (`downcast_to_static_fp8`).
2. `pack_bitmatrix` (vLLM `make_routing_data`, `vllm-project/vllm` at commit
   edbc4969a76b25c531f7b9bb16b79984e65ef023, Apache-2.0) packs `topk_ids` into a token x expert bitmatrix.
3. AITER's `sort_tokens` / `sort_tokens_fused` (the latter for at most 16 tokens, as in AITER `routing()`) compute the
   expert histogram, token and tile offsets, the GEMM tile map and the gather/scatter permutations with the router
   weights in expert order. AITER's `routing()` takes router logits and runs its own top-k kernel first; here the top-k is
   given, so that kernel is replaced by `pack_bitmatrix` and the scratchpads it zeroes are allocated with zeros.
4. `_moe_gemm_a8w4` GEMM1 gathers token rows per expert, adds the bias, applies the SwiGLU in its epilogue and writes FP8
   (division by `w2_input_scale`); GEMM2 adds its bias and multiplies by the router weight; `_reduce_grouped` sums each
   token's 4 rows into `out`. Tile sizes come from AITER's gfx950 A8W4 table (`configs.py`, exactly
   `gfx950-A8W4.json`) keyed by (routing `block_m`, N, K); `block_m` follows AITER's routing heuristic (16..128 from tokens
   per expert). Neither GEMM has a `block_m` 16 entry at N=3072 (K=3072 or 1536), so, as AITER's fallback does, the tile of
   the entry with the same N and K at another `block_m` is used (the `bm32` entry: BLOCK_N 128, BLOCK_K 256, 4 warps, 2 stages).

vLLM's AITER_MXFP4_FP8 MoE backend (`triton_kernel_fused_mxfp4_w4a8_experts` in
`vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py`) makes the same calls after routing from router
logits (`aiter_triton_kernel_w4a8_moe_forward`).

The kernels are unchanged apart from local imports, the removed profiling-only `launch_metadata` hook, one parameter
renamed from the builtin name `input` (`_swiglu`), and the removal of `_moe_gemm_a8w4`'s unused MXFP8 write-back
branch (it depends on AITER's MXFP8 quantization kernels, which the vLLM path does not use). The gfx1250 Gluon,
tensor-descriptor and unpadded-size paths are not vendored, and the reduction writes into the preallocated `out`.

## Provenance notes

- The AITER Python and JSON files used carry no copyright header (only `pid_preprocessing.py` has one, kept verbatim);
  the repository LICENSE applies, and `baseline/LICENSE` is that MIT license. Each vendored file states its upstream path.
- Third-party origin: the header of `_triton_kernels/moe/moe_op_gemm_a8w4.py` reads "adapted from triton_kernels package;
  original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py",
  and the wrapper names `matmul_ogs.py` of the same package. AITER's commit 19bca6d38 ("Simplify and optimize triton_kernels moe
  code and move it into aiter") brought the routing, reduction and quantization kernels in from that package. The
  triton-lang/triton `triton_kernels` package is MIT licensed; its own copyright notice is not part of the pinned AITER
  checkout, so it is not reproduced here. `vpopc` credits "@apgoucher" in its docstring, and vLLM's `pack_bitmatrix`
  docstring cites `python/triton_kernels/bench/distributed.py` of triton-lang/triton as its code reference. `pack_bitmatrix.py`
  keeps its vLLM Apache-2.0 header.

## Workloads

Token counts from decode batches to prefill chunks: 4 to 16384 tokens are timed (4096x). The weights dominate small
batches: one expert is 4.7 MB of `W1` + 2.4 MB of `W2` + 0.4 MB of scales (about 7.5 MB), so 4 tokens (16 routed slots,
about 13 distinct experts) read about 100 MB (an estimated 26 us) and 64 tokens touch nearly every expert (about 0.9 GB,
an estimated 200 us). At 16384 tokens the two GEMMs are 1.9 TFLOP of FP8 (an estimated 1 ms at peak). Batches of 1-3
tokens touch at most 12 experts, within about 10 us of the floor, so they are correctness-only, together with unaligned
batches (17, 33, 100, 1000, 3000). 17 tokens is the first batch past the fused-routing bound (at most 16 tokens);
16384 tokens is the first with more than 65535 routed rows (32-bit permutation indices). The 8192/16384 chunks are the
usual vLLM `max_num_batched_tokens` settings.

The routing kernels take their shapes from `topk_ids` only and nothing on the host reads GPU data, so the launches
(plus the zero fills of the routing scratchpads) are capturable in a CUDA graph. The kernels use `tl.dot_scaled` (e4m3 x
e2m1 with `fast_math`), `tl.reduce_or`, `tl.sort` and `tl.associative_scan` as AITER does at the recorded commit. The
GEMMs have no split-K and no atomics, so results are deterministic.
