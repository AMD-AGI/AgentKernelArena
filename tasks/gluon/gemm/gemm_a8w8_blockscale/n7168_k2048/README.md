# a8w8 block-scaled GEMM (Gluon baseline), N = 7168, K = 2048

`out = dequant(a) @ dequant(b)^T`: float8_e4m3fn activations `a` [M, 2048] with one float32 scale per 1x128 group
(`a_scale` [M, 16], row-major) times float8_e4m3fn weights `b` [7168, 2048] in the natural `nn.Linear` order with one
float32 scale per 128x128 block (`b_scale` [56, 16]). Each K block is accumulated in float32, multiplied by the
activation and weight scale of its block and summed; the result is rounded to bfloat16. The reference
(`reference.py`, shared in structure with the Triton scenario `tasks/triton/gemm/gemm_a8w8_blockscale/n7168_k2048`)
dequantizes both operands and multiplies in float64, chunked over rows, independent of the baseline.

## Scenario and workload origin

N = 7168 and K = 2048 are the shared expert's `down_proj` of DeepSeek-V3 (and R1), a dense GEMM with M = tokens. (The routed experts' `down_proj` runs through grouped MoE kernels, not this GEMM.) Config citation, DeepSeek-V3 `config.json`
(Hugging Face): `hidden_size` 7168 (output channels N), `n_shared_experts` 1 x `moe_intermediate_size` 2048 = 2048 (contracted dimension K),
`quantization_config`: `quant_method` fp8, `fmt` e4m3, `activation_scheme` dynamic, `weight_block_size` [128, 128]
(the 1x128 activation groups and 128x128 weight blocks). This shape has its own tuned file in AITER's Gluon config
directory (`gfx950-GEMM-A8W8_BLOCKSCALE-N=7168-K=2048.json`).

`m` is the token count of the call; the timed cases are weighted by a recorded serving workload (see Workloads below).

**Inputs.** `inputs.py` plays the role of the quantizing producers. Activations are normal values with a per-(row,
128-column group) magnitude drawn over a 16x range, quantized to e4m3fn with scale `amax / 448`; weights are the same
per 128x128 block. The Gluon baseline consumes the plain layouts (row-major `a_scale`, natural-order `b`), so the task
has no layout transform; the Triton siblings of this operator use the AITER preshuffled weight and a column-major
`a_scale` payload because their baselines read those layouts.

## Workloads

The performance cases come from a recorded agentic serving workload: the AgentX corpus
(`semianalysisai/cc-traces-weka-062126`, agentic coding sessions) replayed with 32 concurrent sessions against
`deepseek-ai/DeepSeek-V3` (FP8 checkpoint) on SGLang 0.5.23 and MI355X (capture `deepseek-v3-tp8-mi355x-v3`):
TP 8, no DP attention, MTP speculative decoding with the checkpoint's own MTP layer (EAGLE, 3 steps, 4 tokens
verified per sequence), chunked prefill 16384, FP8 KV cache, aiter attention, and the model's 163840-token
context (requests that do not fit are skipped: 45% of the corpus requests, 81% of its input tokens). Each
case's `weight` is how often its shapes ran (calls per second of the steady state, TP rank 0), and the task
scores the serving-weighted speedup `sum(w*t_baseline)/sum(w*t_candidate)`, the time the workload spends in
the operator before and after (see Case weights in `docs/how-to/add-task.md`). `weight_provenance` in
`workloads.json` records the capture, the effective serving settings and the share of the corpus the context
window dropped. The 10 weighted cases cover 98.3% of the operator's recorded GPU time:

| `m` | Phases | Share of the operator's time | Weight (calls/s) |
| ---: | --- | ---: | ---: |
| 80 | draft, draft_prefill, prefill, verify | 2.7% | 47.11 |
| 120 | draft, draft_prefill, prefill, verify | 17.5% | 302.3 |
| 160 | draft, draft_prefill, prefill, verify | 5.1% | 80.6 |
| 1601 | draft_prefill, prefill | 1.2% | 5.531 |
| 2241 | draft_prefill, prefill | 1.3% | 4.62 |
| 2944 | draft_prefill, prefill | 3.5% | 8.51 |
| 5057 | draft_prefill, prefill | 2.1% | 3.1 |
| 6081 | draft_prefill, prefill | 1.6% | 2.127 |
| 10753 | draft_prefill, prefill | 3.6% | 3.161 |
| 16384 | draft_prefill, prefill | 59.7% | 34.1 |

With MTP the target model's decode is a verify step of 4 tokens per sequence, so the decode GEMM runs at `m` =
4 x the padded batch (80 to 160 at this load); the MTP draft layer runs the same GEMM (phases `draft` and
`draft_prefill`). At TP 8 this per-rank shape is the attention output projection (`o_proj`: 128 heads x 128
value dimensions / 8 = 2048 inputs, 7168 outputs); the shared expert's `down_proj` named above is 7168 x 256
per rank at TP 8, so the weights describe `o_proj` calls of the same GEMM shape. Prefill takes about 79% of
the time, 60% in full 16384-token chunks. In the MI355X QA audit the baseline took 24-25 us for the
verify-sized cases (the 15 MB weight read alone would take about 3.3 us) and 50 us to 0.42 ms for the prefill
cases.

The earlier performance cases (`m` = 512 to 65536, except 16384, which is now weighted) are correctness-only,
as are the earlier decode-sized and boundary cases (`m` = 1 (axis minimum), 3 and 5 (not powers of two), 17,
100, 129, 1000 and 2049 (not multiples of 16 and above 16)), the other recorded shapes and `m` = 65537 (row
indices past 16 bits) and 131072 (axis maximum) (63 cases in total).

## Baseline

AITER's Gluon kernels in `aiter/ops/triton/gluon/gemm_a8w8_blockscale.py` (gfx950 only): `_gemm_a8w8_blockscale_kernel`
with `_compute_MN_tile`, a pipeline of direct global-to-LDS async copies for operands and scales feeding
`mfma_scaled`, plus the Gluon split-K reduce kernel `_gemm_a8w8_blockscale_reduce_kernel` and the launcher with its
config lookup (see `provenance` in `task.yaml`). Kernel and launcher code are identical to upstream except for the
changes listed in `provenance.modifications` (including one added bound, described under the risks): the config lookup reads the inlined table of
`gfx950-GEMM-A8W8_BLOCKSCALE-N=7168-K=2048.json` instead of files, and the wrapper writes into the preallocated output.

*Config selection.* The Gluon kernel supports only the tiles (64, 128), (128, 128) and (128, 256) with BLOCK_K = 128
and 4 warps. The N=7168, K=2048 table lists 16x32 and 32x32 tiles for M <= 128, which the lookup skips as unsupported,
so M <= 2048 runs 128x128 tiles and M > 2048 the `any` entry (128x256, group size 4). Every selected entry has
`NUM_KSPLIT = 1`, so the split-K reduce kernel is never launched in this scenario; it is a Gluon kernel and stays in the
baseline unchanged, so a candidate copy of the baseline keeps the whole production wrapper. A single Gluon kernel
is launched per call. `remap_xcd` and `pid_grid` are `@triton.jit` helpers that the Gluon kernel calls inline, as upstream does;
they are copied unchanged into `baseline/pid_preprocessing.py` and are never launched. Upstream calls
`remap_xcd(pid, GRID_MN)` without using the result (a no-op); it is kept unchanged.

Provenance: `ROCm/aiter` at `652f2d4b768239aefd515c929e84817681ff598d`, MIT; `baseline/LICENSE` is the AITER license.
`kernel.py` and `pid_preprocessing.py` keep the upstream header verbatim
(`SPDX-License-Identifier: MIT`, `Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.`). The
file's history starts with AITER commits ("Add Gluon Blockscale A8W8 GEMM"); neither the file nor its directory
records a third-party origin.

## Estimated timing

Roofline estimate (fp8 2e15 flop/s, 4.5e12 B/s, 4 us fixed): `m = 512` is 1.5e10 flop = 7.5 us + 4 us, about 11.5 us;
`m = 65536` is 1.9e12 flop, about 1 ms (it writes 0.94 GB of bfloat16). The real kernel is
slower than the roofline.

## Notes

- Gluon API: the kernel needs `gl.amd.cdna4.async_copy` (`buffer_load_to_shared`, `load_shared_relaxed`),
  `mfma_scaled` and `PaddedSharedLayout` in the image's Triton 3.8 on gfx950.
- Upstream reads out of range in M: the scale async copies are unmasked in M, so for ragged M (all small decode counts, M=1
  included, which dispatch 56 workgroups of 128x128 tiles) the last tile reads `a_scale` rows past the end of the tensor
  (up to 127 rows x 64 bytes = 8 KB), which gave a memory fault on MI355X. `_compute_MN_tile` now wraps the row
  offsets of the `a_scale` copy with `% M` when `NEED_M_MASK` is set (the same wrap the Gluon `gemm_a8w8` kernel
  applies to its operand rows). Wrapped rows are valid scales of other rows; their outputs are discarded by the C
  store mask. Upstream has no mask parameter, pads nothing in its wrapper and its tests pass unpadded scales, so this
  small change in the kernel is the minimal fix; it is recorded in `provenance.modifications`. `a`, `b` and `b_scale`
  need no change: `a` is masked by `m_mask`, and N = 7168 and K = 2048 are exact multiples of the tiles and scale
  blocks, so no `b_scale` row or K group is read out of range.
- Deterministic: one K loop per output tile and no atomics (the unused split-K path would reduce partials in a fixed
  order). The float32 accumulation order differs between candidates; the `baseline_relative` bound absorbs it.
- The 65536-token case compares against a 3.8 GB float64 reference output.
