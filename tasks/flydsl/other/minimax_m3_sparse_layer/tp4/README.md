# MiniMax-M3 sparse MoE layer monokernel, tensor parallel 4 (multi-GPU)

One sparse MoE decoder layer of MiniMax-M3 (hidden 6144, 64 q / 4 KV heads of 128, 128 routed experts top-4 plus a
fused shared expert, MXFP4 experts) for a decode or EAGLE3-verify step, run as **one persistent launch per rank on
four MI355X GPUs**. The ranks exchange their partial rows through peer memory inside the launch (both all-reduces),
so the task uses the framework's `distributed` contract (`world_size: 4`, see
[add-task.md](../../../../../docs/how-to/add-task.md#multi-gpu-tasks)). TP4 is built into the kernel (16 q heads, one
KV head, one index q head and a 768-wide expert slice a rank); there is no single-GPU mode.

## What one call computes (per rank)

See `task.yaml`'s description and `m3_numerics.py` for the exact numerics. In order: input add and Gemma RMSNorm,
per-token FP8, the rank's fused FP8 q | k | v | index q | index k GEMV, per-head Gemma norm and partial NeoX RoPE,
the page-16 SHUFFLE K / V cache writes (fixed per-layer scales) and the index-K write; indexer block scores (max over
a 128-key block of bf16(index k) . index q); the 15 best blocks plus the row's last block; sparse FP8 attention in
256-key splits with FP8 probabilities, merged; per-token FP8 and the rank's FP8 o_proj columns; the rank-ordered
all-reduce (`h_mid`); router (Gemma norm, bf16 gate, sigmoid + bias top 4, renormalized x 2); MXFP8 x MXFP4 experts
with swiglu (alpha 1.702, limit 7) and MXFP8 mids; the second rank-ordered all-reduce (`ar_out`).

Outputs `h_out`, `h_mid`, `ar_out` are replicated (`ranks_agree`: bitwise equal on every rank). The K / V caches
(the rank's KV head) and the index cache are mutable inputs checked after the call.

## Multi-rank contract

- `inputs.py::make_inputs` draws the **logical** inputs once per case and seed: replicated tensors at the rank's
  shape, rank-specific weights and caches stacked on a leading TP axis (`w_qkv[r]`, `k_cache[r]`, ...), caches in
  token-major layout and one block-table row per request. Every rank draws them identically.
- `shard_inputs` picks rank r's slices, writes its caches in the kernel's page-16 SHUFFLE layout (separate K and V
  planes) and gives it its own index-cache copy; `to_kernel_layout` then preshuffles the weights as SGLang / AITER
  load them (`shuffle_weight (16, 16)` on the FP8 and MXFP4 codes, `e8m0_shuffle` on the MXFP4 scales, from
  `minimax_m3/weights.py` and AITER).
- `rank_expected` maps the reference's logical outputs to each rank (replicated rows; the rank's caches shuffled).
- Reserved call inputs `rank`, `world_size`, `peers`, `symmetric` come from the framework. `symmetric_bytes`
  (3,965,184) is the kernel's `sym_layout(4)` (attention / FFN partial rows, the reduce-scatter regions, the
  selection candidates) plus the guard's 256-byte peer error flags. The framework zeroes it before every call.
- The kernel's device-local mailbox scratch (K1 0.58 MB, K4 3.76 MB) and its epoch / guard words are allocated and
  zeroed inside `run()` (ordinary device memory, as production allocates it), so every call starts clean with a
  fixed mailbox tag; production instead advances an epoch word between steps. The memsets are inside the timed
  graph (a few microseconds), the same for every implementation; a candidate may organize its own state.
- `q_len` (rows a request) is `rows / requests`; the candidate derives it from `seq_lens` and `block_table` shapes.

## Reference and its assumptions

`reference.py` (float64, via the task-root helper `m3_numerics.py`, shared with the generator) was written from the
kernel source; **the repository has no numeric test of this kernel** (only compile tests, and an SGLang check mode
with loose bounds), so the reference is our reading of it. It applies the dataflow roundings the kernel defines
exactly (bf16 stage outputs, per-token FP8 of the qkv input and of the attention output, FP8 q with unit scale, FP8
probabilities per 256-key split relative to the split max, MXFP8 with aiter's round-up exponent rule for the router
output and the expert mids, bf16 partial rows before each rank-ordered sum, bf16 router logits) and returns final
values before their last rounding. Assumptions:

- Accumulation order (float32 MFMA and wave sums) and the hardware `exp2` / `rcp` / `rsq` approximations are not
  modelled; `h_out`, `h_mid`, `ar_out` and the new cache entries carry those differences only.
- Block selection follows the kernel's placement (selected full blocks by descending (score, block id), the tail
  block last), because the 256-key split boundaries follow that order. A long request (> 512 blocks) is selected in
  eight shares in the kernel; the result equals the global top 15, which the reference computes directly.
- The shared expert weight is 1 and the routed weights are renormalized and scaled by 2 (spec.py); the routing key's
  low 7 mantissa bits (dropped by the kernel's `route_key`) and its tie rule never decide because of the margins.
- Every row is live (no graph pad rows: every slot > 0, every `seq_len` > 0).
- The index k head is replicated (the same projection rows on every rank), so every rank writes the same index cache.
- MiniMax-M3 constants: eps 1e-6, sm_scale 128^-0.5, routed scaling 2, swiglu limit 7 / alpha 1.702, selection
  (16, init 0, local 1) from `spec.py` / the model config as the repository's tests state them. The RoPE table uses
  theta 5e6 (from AITER's experimental M3 bench); it is an input, so its exact base does not change the operator.

## Planted margins (inputs.py)

Two decisions are discrete. **Block selection**: for each rank and scored request, 15 random hot blocks below the
request's first new key hold index keys from the dual basis of the rows' index q (computed with the reference's
numerics), with descending strengths 1 .. 0.44 (steps of 0.04, wide enough that the order survives the FP8 rounding of
the planted keys) so their order is also clear; all other index keys are 0.25 N(0, 1).
The same slots of the rank's K cache hold keys the row's q heads attend to (mean logit 6), so the selected blocks
carry most of the attention mass: on CPU, swapping one selected block for an unselected one moved 89 % of `h_mid`
elements by more than two bf16 roundings. **Routing**: the router weight is N(0, 0.02) plus the least-squares
correction that gives each row 4 random winners with logits in [1.5, 3] and the rest in [-4, -0.25] for the row's
exact router input; sigmoid + bias then separates the top 4 by >= 0.27. The generator checks both margins and raises
if either is too small.

## Baseline

Vendored from sgl-project/sgl-amd-mono at 52486703 (SGLang ROCm images pin e52803f9, 11 commits older): the
`minimax_m3/gfx950` kernel files (MIT, AMD, from ROCm/aiter PR 6047; `select.py` from ROCm/ATOM 454d0e3023), the
`common/` device runtime it imports (Apache-2.0, adapted from ROCm/ATOM 97359d6), AITER's `swiglu_mul_batch`
(act.py at 6482901d), FlyDSL v0.3.2's `buffer_ops.py` / `dpp_utils.py`, and a FlyDSL 0.3.2 bridge (`compat.py`,
from sgl-amd-mono's muse_spark bridge, as plain functions because candidates may not patch modules). See
`task.yaml` `provenance` for every modification and `baseline/LICENSE` for the MIT and Apache-2.0 texts. K1's
per-layer pointers are kernel arguments instead of the runner's device pointer table.

End-to-end serving of MiniMax-M3 on this monokernel path additionally needs the unmerged SGLang PRs #41998 (page-16
SHUFFLE K / V pool) and #41708; this kernel task does not depend on them.

## Workloads

Derived from corpus statistics; not yet confirmed by a live capture. AgentX
(semianalysisai/cc-traces-weka-062126, 98,827 requests): decode-step-weighted context p10 52K, p25 108K, p50 220K.
MiniMax-M3 serves 131,072 tokens, so contexts are clipped at the cap: **53 % of AgentX requests (about 70 % of
decode steps) exceed it and cannot use this model**; the cap is therefore the most frequent perf context. Rows are
the production widths 1 (decode), 4 (EAGLE3 verify of one request, 3 draft tokens) and 16 (four such requests,
ragged contexts in [ctx / 2, ctx]): three compiled row counts. Monokernels serve decode / verify only; prefill does
not apply.

Estimated baseline time (bytes a rank reads at ~5 TB/s cold: 30 MB of attention weights, 7.5 MB an expert slice,
128 B of index K a context token; the in-kernel synchronization adds more): rows 1, 16K / 64K / 128K: 14 / 15 /
17 us; rows 4 (17 experts), 52K / 108K / 128K: 33 / 34 / 35 us; rows 16 (~51 experts), 64K / 128K: 88 / 93 us.

## Not verified

No GPU was available when this task was written: the FlyDSL compile of the vendored kernel on FlyDSL 0.3.2, its
numerics against the reference, the bounds (default `baseline_relative`), multi-rank timing and the 10-minute budget
are unverified and need `validate-task` and a QA audit on a 4-GPU MI355X group. The CPU checks covered `check_task`,
the generator, reference, shard, layout and expected on CPU for the small and medium cases.
