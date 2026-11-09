# MLA decode, FP8 queries and latent KV cache, 128 heads, kv_lora 512 + rope 64, one-token pages

Absorbed multi-head latent attention for one decode step: each sequence has one query token whose 128 heads are 576 wide
(`[512 latent | 64 RoPE]`, float8_e4m3fn); the single latent KV head lives in a cache of one-token pages
(`kv_cache[pages, 1, 1, 576]`, float8_e4m3fn) addressed through `kv_indptr` / `kv_page_indices`; the value is the
512-wide latent part of the same cache row. Queries and cache are used as stored (the FlyDSL kernel takes no scales).
The output is bfloat16 `[batch, 128, 512]`. This is the FP8, 128-head, DeepSeek-V3 tensor-parallel-1 sibling of the bf16, 16-head, 64-token-page Triton task
`tasks/triton/attention/mla_decode/bf16_h16_lora512_rope64_p64` (same operator, same `q` and `softmax_scale` names); the
interface differs by the worklist and the page addressing, so it is a separate scenario.

## Scenario and config citations

- DeepSeek-V3 / R1 attention with `num_attention_heads` 128, `kv_lora_rank` 512, `qk_rope_head_dim` 64, `qk_nope_head_dim` 128
  and the absorbed value width 512: SGLang `test/manual/attention/test_trtllm_mla_backend.py` (commit
  `887c401e151e2d1b082d74ef8769a23eef5c465c`, its "MLA model config" dict) and FlyDSL `tests/kernels/test_mla_decode.py`
  (tag v0.3.2, "Model constants (DeepSeek-V3 / R1)": `KV_LORA_RANK` 512, `QK_ROPE_HEAD_DIM` 64, `NHEAD` 128, `NHEAD_KV` 1,
  `PAGE_SIZE` 1).
- `softmax_scale` 0.1352 = `192^-0.5 * mscale^2` with the YaRN `mscale = 0.1 * mscale_all_dim * ln(factor) + 1`, `factor` 40,
  `mscale_all_dim` 1.0 (the `ROPE_SCALING_CONFIG` of the same SGLang test, `compute_mla_mscale_scaling` in
  `srt/configs/model_config.py` and `yarn_get_mscale` in `srt/layers/rotary_embedding/yarn.py`); the arithmetic gives 0.13523.
- Production use: SGLang's aiter backend (`python/sglang/srt/layers/attention/aiter_backend.py`) runs 128-head MLA with an
  FP8 KV cache on AITER's persistent kernel with `fast_mode=True`, `kv_granularity=max(page_size, 16)` and
  `max_split_per_batch` capped at 64 without draft tokens (`min(ceil(cu_num / batch), cap)` under DP attention), which is how
  the worklist below is built.

## Cache and worklist

- `kv_page_indices[kv_indptr[b] + t]` is the cache row of token `t` of sequence `b`. Every sequence owns `ctx` scattered rows
  and lists the first `seq_len` of them; the longest sequence has `ctx` tokens, the others between half of it and all of it.
  Cache rows that are not listed hold stale but finite codes that must not reach the result. Cache codes are `float8_e4m3fn` of
  N(0, 1) values and queries of N(0, 0.5) values (logit standard deviation about 1.6), cast directly.
- The worklist (`work_indptr`, `work_info`, `reduce_indptr`, `reduce_final_map`, `reduce_partial_map`) is what AITER's
  `get_mla_metadata_v1` builds in fast mode for 256 compute units: the parallel planner
  (`kn_get_mla_metadata_v1_2_parallel` in `csrc/kernels/mla/metadata/v1_2_device.cuh`, the default for one query token, page size 1)
  deals the blocks of all sequences (plus a fixed cost per work item) evenly over
  `min(256, 64 * batch)` splits; a sequence cut at a compute-unit boundary becomes several work items, each with a partial
  slot (`partial_qo_loc`) that the reduce merges. `inputs.py` ports the planner step by step (both phases), with one
  deliberate difference: blocks are 64 tokens, not AITER's 16, and the last work item of a sequence takes the partial tail
  block (see "Work items shorter than the kernel tile" below). Production builds
  it once per decode step and shares it across all layers, so it is an input rather than timed work; a candidate may use it
  or schedule itself.

## Baseline

`baseline/kernel.py` is ROCm/FlyDSL `kernels/attention/mla_fwd_decode_m16x8_fp8_fp8.py` at tag v0.3.2
(`dd8373349e33638c8d158a9ea009b7b28598984a`): a persistent 8-wave kernel (512 threads, one workgroup per compute unit) that walks
its work rows, streams 64-token KV tiles through a double-buffered LDS layout with transposed-during-load V reads, and computes
both products with FP8 `mfma_f32_16x16x32` and an online softmax. Probabilities `exp(s - running max)` are converted to FP8 without
rescaling, the row sum stays float32, and every work item writes a normalized float32 partial output plus its log-sum-exp.
`reduce.py` merges the partials per sequence and head with a running log-sum-exp and writes the bfloat16 output (FlyDSL's
deterministic replacement of AITER's `mla_reduce_v1`). `run()` allocates the two partial buffers per call from
`reduce_partial_map`'s length, launches the kernel and the reduce on the current stream, and fixes the LDS budget to the 160 KB per
CU of CDNA4 (upstream asks `rocminfo` through `subprocess` and uses the value only for an assertion). Per-file changes are in
`task.yaml` `baseline.provenance.modifications`; no kernel code was altered.

### A limit of the baseline kernel: every sequence must be split

The kernel stores every work item to the partial buffers, at row `partial_qo_loc * 128 + ...`. AITER's planner gives a work
item `partial_qo_loc = -1` when a whole sequence fits in the rest of one compute unit, and the C++ kernel then writes the final
output directly; the FlyDSL port always takes the split path (`_do_last_gemm2_and_store` passes `is_split=True`), so for such
a sequence it would store to a negative row and the output would stay unwritten. Upstream's tests never meet this case (their
sequences are always cut). This task therefore keeps the batch and the context in a region where every sequence is cut:
`batch` is at most 64 and `ctx` at least 512. With ragged lengths between `ctx / 2` and `ctx`, the shortest sequence has more
blocks than the payload of one split in the worst case (for example 4 blocks of 64 tokens against at most 3 for 64 sequences
of 512 tokens); a worklist with an unsplit work item raises an error in `inputs.py`. Smaller contexts at high batch (for example 64
sequences of 256 tokens) and batches of 128 and more are outside the task. This is a limitation of the upstream kernel; AITER
uses its own HIP kernel for those shapes.

### Work items shorter than the kernel tile and how the worklist avoids them

With AITER's own parameters (`kv_granularity` 16, fixed cost 16 blocks) the baseline fails the SQNR sanity check on
short contexts on MI355X: 7.07 / 7.40 dB for `(1, 512)`, 7.08 / 7.01 dB for `(2, 513)`, 13.4 dB for `(3, 1000)` and
13.5 dB for `(5, 1500)`, while every case with `ctx >= 16384` passed at 37-44 dB. The ported planner's work lengths match the
pattern: in `(1, 512)` and `(2, 513)` every work item has 16 tokens (median 16), in `(3, 1000)` and `(5, 1500)` they have 32
(median 32), so 100% of the tokens sit in work items shorter than the kernel's 64-token tile `BLOCK_N`. In the cases that passed,
at most 4.5% of the tokens do (`(17, 777)`: 152 of 169 items have exactly 64 tokens, 17 have 7-63; `(33, 2000)`: 0.3%; `ctx >= 16384`:
under 0.1%, items of 800-3000 tokens whose tails are the only short ones). The error falls as the item grows (16 tokens: 7 dB, 32: 13 dB,
64 or more: pass), and a partial last tile of a multi-tile item (a tail of 1-63 tokens after full tiles) is fine, so the failing path is the
single-tile item with `kv_len < 64`: the `first_tile_needs_boundary` branch of `kn_mla_fwd_decode_m16x8_fp8_fp8`
(`kv_len < BLOCK_N`, then `_process_tile_gemm1(check_boundary=<runtime>)` and the single-tile `_do_last_gemm2_and_store`).
**The exact instruction-level cause inside that branch was not identified** (it needs a GPU trace); the association is with the
work-length statistics. Upstream's own test only reaches such items through `(1, 63)` and
`(1, 64)` and checks a cosine threshold on N(0, 1) data; it never checks them at the SQNR level used here.
Mitigation in `inputs.py` (no kernel change): the planner schedules in blocks of 64 tokens (`kv_granularity` 64, the kernel's `BLOCK_N`),
keeps AITER's fixed cost in tokens (4 blocks of 64 = 16 blocks of 16 = 256 tokens) and lets the last work item of a sequence take the
partial tail block, so every item has at least 64 tokens (the generator asserts it); non-final items are whole multiples of 64
(full tiles). This departs from AITER's `get_mla_metadata_v1`, which would be called with `kv_granularity=max(page_size, 16)` in
production; the departure only changes how work is cut (fewer, tile-sized items, for example 8 items for `(1, 512)` instead of 32), not
the kernel or the data.
### Accuracy and the bound

The default `baseline_relative` bound is kept. A CPU emulation of the arithmetic (FP8 queries and cache exactly, float32 logits and
sums, tile-wise online maximum over 64-token tiles, probabilities rounded to FP8 e4m3, bfloat16 output) against the float64
reference gives 34.8-37.0 dB SQNR at 512, 8192 and 49152 tokens with the input scales above, above the 25 dB sanity floor. An
implementation that reproduces the FP8 probabilities has the baseline's error pattern; one that keeps them in bfloat16 or better
has a smaller error. A variant that quantizes at another granularity may need a larger `k`, which this task does not grant.

## Workloads

Concurrency 1-64 and contexts of 16k-96k tokens (AITER's `op_tests/test_mla.py` sweeps `-c 49152` and `-c 98304`, batches
1-256, and the 128-head, 1-query-token case). Estimated baseline time from the cache traffic (576 bytes per token, 4.5 TB/s) and the
FP8 flops (`2 * 128 * (576 + 512)` per token, 2.0 PFLOP/s) plus 4 us: the smallest performance case, one 48k request, is about 11 us (a
single request is split into at most 64 work items, so its real time is likely higher); the largest, 64 requests of up to 96k
tokens (about 4.7M tokens, 2.7 GB), is about 0.65 ms (flop-bound). The span of `batch * ctx` is 6.3M / 49152 = 128x. Correctness-only cases: the
minimum `(1, 512)` (8 work items of 64 tokens), odd batch and context `(3, 1000)`, batch 17 with contexts that are
not multiples of 16 `(17, 777)`, `(5, 1500)`, `(2, 513)` and `(33, 2000)`.

## Provenance

- `kernel.py` carries `SPDX-License-Identifier: MIT` and `Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.`
  verbatim. Its docstring says it is "Transplanted from csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh" of ROCm/aiter,
  which is AMD code under the same MIT license (`baseline/LICENSE` is the MIT text of the ROCm/aiter repository at
  `652f2d4b768239aefd515c929e84817681ff598d`). That AITER C++ kernel is built on HipKittens (`kittens.cuh`;
  HazyResearch/HipKittens, MIT-licensed, which AITER fetches as a third-party dependency in `aiter/jit/core.py`). No
  HipKittens code is vendored in this baseline, so no HipKittens license file is carried. The file also says some helpers
  were "ported from FlyDSL/kernels/mla_decode_fp8.py" (an earlier FlyDSL file of the same authors).
- `reduce.py`, `buffer_ops.py`, `utils.py` and `kernels_common.py` carry `SPDX-License-Identifier: Apache-2.0` and
  `Copyright (c) 2025 FlyDSL Project Contributors`; the Apache-2.0 text is `baseline/LICENSE-FlyDSL`. `buffer_ops.py` cites the LLVM
  AMDGPUToROCDL source of the descriptor flag layout in a comment.
- `inputs.py` begins with a note that its planner ports AITER's MIT-licensed `v1_2_device.cuh`.
- The MIT notice in the kernel header comes from the file, and the license text from the repository the kernel was transplanted from;
  the FlyDSL repository itself is Apache-2.0.

## Not covered

DeepSeek's FP8 variants with a bf16 RoPE part (the `v40` kernels), multi-token queries, the 16- and 64-head variants and the bf16 cache are
other kernels or scenarios. The bf16 Triton sibling is `tasks/triton/attention/mla_decode/bf16_h16_lora512_rope64_p64`.
