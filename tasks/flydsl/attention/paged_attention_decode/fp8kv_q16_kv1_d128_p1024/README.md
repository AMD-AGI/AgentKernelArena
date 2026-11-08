# Paged attention decode, FP8 KV cache (per-token scales), 16 query heads / 1 KV head, head size 128, 1024-token pages

Single-token (decode) grouped-query attention over a paged FP8 KV cache. For each sequence `s` and query head `h`,

```text
out[s, h] = softmax(softmax_scale * q[s, h] . K_s^T) . V_s
K_s[t] = key_code[t] * key_scale[t],  V_s[t] = value_code[t] * value_scale[t]
```

over the first `context_lengths[s]` cached tokens; the sixteen query heads share the single KV head,
`softmax_scale = 128 ** -0.5`, and the result is rounded to bfloat16. The float64 reference dequantizes the
codes exactly and is otherwise exact attention. This is the 16-head sibling of
`fp8kv_q8_kv1_d128_p1024` (same operator, interface names, cache layouts and reference): the query-head count,
the baseline (FlyDSL v0.3.2 instead of an older FlyDSL commit) and the worklist differ, so it is a separate scenario.

## Scenario and config citations

16 query heads per KV head with head size 128 is the per-rank attention of a 128-head / 8-KV-head model at tensor
parallel 8, for example Llama-3.1-405B.

- vLLM `benchmarks/kernels/weight_shapes.py` (commit `edbc4969a76b25c531f7b9bb16b79984e65ef023`) lists
  `meta-llama/Llama-3.1-405b-hf` with a fused QKV weight of `[16384, 18432]`, whose output width is
  `16384 + 2 * 1024`: with 128-element heads that is 128 query heads and 8 KV heads, split along the head
  dimension by tensor parallelism. The file does not state `head_dim`; 128 is taken from the kernel's own tests.
- FlyDSL `tests/kernels/test_pa.py::test_metadata_accuracy` (tag v0.3.2) runs this exact configuration of the
  baseline kernel: `num_heads=(16, 1)`, `head_size=128`, `block_size=1024`, per-token or per-tensor scales,
  `context_length=1027`, `batch_size=3`.
- 1024-token pages and the per-token FP8 cache come from AITER's persistent paged-attention path
  (`op_tests/test_pa_ps.py`, `block_size` default 1024, `batch` default `sorted([64, 158, cu_num // 2])`,
  contexts up to 10240).

## Cache and worklist

- `query[num_seqs, 16, 128]` bfloat16: one decode token per sequence.
- `key_cache[pages, 1, 8, 1024, 16]` float8_e4m3fn: element `d` of the token in `(page, slot)` is at
  `[page, kv_head, d // 16, slot, d % 16]`.
- `value_cache[pages, 1, 64, 128, 16]` float8_e4m3fn: element `d` of the token in `(page, slot)` is at
  `[page, kv_head, slot // 16, d, slot % 16]` (the persistent kernel's transposed value layout).
- `key_scale`, `value_scale[pages, 1, 1024, 1]` float32: per token and head, `scale = max|x| / 448` over the head
  dimension as AITER's `pertoken_quant` writes them, `code = float8_e4m3fn(x / scale)`.
- `context_lengths`, `kv_indptr`, `kv_page_indices`: token `t` of sequence `s` is in page
  `kv_page_indices[kv_indptr[s] + t // 1024]`, slot `t % 1024`. Every sequence owns `ceil(max_seq_len / 1024)`
  scattered pages and lists the first `ceil(context_lengths[s] / 1024)`; the longest sequence has `max_seq_len`
  tokens, the others between half of it and all of it. Slots past a sequence's length and unlisted pages hold
  stale but finite codes and scales that must not reach the result.
- The worklist (`partition_indptr`, `work_indptr`, `work_info`, `reduce_indptr`, `reduce_final_map`,
  `reduce_partial_map`) is what FlyDSL's `get_pa_metadata_v1` scheduler (`pa_metadata.py`, uniform query length 1)
  builds for 256 compute units. The 256-token partitions of all sequences are dealt out evenly, so the
  work is balanced to one partition (`work_info` kv_start/kv_end are cumulative partition indices, and
  `partition_indptr` converts them to sequence-local ones); a sequence cut at a compute-unit boundary becomes
  several work items whose partial outputs and log-sum-exps go to partial slots that the reduce merges
  (`reduce_*`). `inputs.py` ports the scheduler kernel step by step. Production builds it once per decode step and shares it
  across all layers, so it is an input rather than timed work; a candidate may use it or schedule itself.

## Baseline

`baseline/kernel.py::run` is the block-1024 persistent branch of `pa_decode_ps_launch`
(`kernels/attention/pa_decode_fp8.py`, ROCm/FlyDSL tag v0.3.2, `dd8373349e33638c8d158a9ea009b7b28598984a`): the
persistent decode kernel `compile_pa_decode_metadata` from `pa_metadata.py` (grid of 256 workgroups, one per compute
unit, each walking its work rows with an online softmax over 256-token partitions; FP8 `mfma` for both products), followed by
the FlyDSL `compile_pa_metadata_reduce` kernel that merges split partials per (sequence, query head) with a
running log-sum-exp (upstream replaced AITER's racy `pa_reduce_v1` by it). `pa_metadata.py`, `pa_common.py`, `dpp_utils.py`
are the upstream files; `utils.py`, `kernels_common.py` and `buffer_ops.py` are the parts of FlyDSL's
`kernels/common/` that they import. What was changed, file by file, is recorded in `task.yaml`
`baseline.provenance.modifications`: only imports and environment/helper stand-ins that the candidate rules require, no
kernel code. `pa_metadata.py` also contains the scheduler kernel and the `get_pa_metadata_v1` host wrapper; `run()` calls
neither, because the scheduler needs host synchronization (`reduce_indptr[-1].item()`) and, in production, is not
part of each layer's attention call.

`run()` allocates the two fp32 partial buffers per call from the worklist shapes
(`(max_split_tiles + 1) * query_length` rows; upstream sizes them from `reduce_indptr[-1]`, a host read). The first
`query_length` rows are the kernel's reserved rows, which the reduce skips by slicing.

### Accuracy and the bound

The default `baseline_relative` bound is kept. The persistent kernel quantizes each query head to FP8 (`scale = max|q| / 240`)
and the probabilities to FP8 after folding each token's value scale in and normalizing by the largest value scale of the
256-token partition, then rescales the product. A CPU emulation of that arithmetic (exact running maximum, unquantized
softmax sum, bfloat16 output) against the float64 reference gives 28.2-29.0 dB SQNR for contexts of 1000, 8192 and 32768 tokens,
above the 25 dB sanity floor but with a margin of only about 3 dB. An implementation that reproduces the baseline's FP8
quantization (same per-head query scale, same partition-wise probability normalization) has the baseline's error pattern;
one that keeps the query and the probabilities in bfloat16 or better has a much smaller error. A variant that quantizes at
another granularity has errors of the baseline's size at different elements and may fail at `k` = 2; the bound is not widened.

## Workloads

The case list of the 8-head sibling: batch sizes 1-512 and contexts up to 128k. The 16-head kernel reads the same bytes per
token (one KV head), so the estimated times follow the cache traffic of 264 bytes per token (256 FP8 bytes plus two
float32 scales), 4.5 TB/s and 4 us: the smallest timed case, one 128k request, moves 35 MB (about 12 us by that rule); the largest, 256
sequences of up to 64k tokens (about 12.6M cached tokens, 3.3 GB), about 0.74 ms. The single 128k request splits across
256 compute units, so its 256 partials make the serial reduce a large part of its real time. Correctness-only cases: the
minimum `(1, 1)`, `(3, 1027)` (the upstream test's batch and context), contexts one token past a page, one page per
sequence, more sequences than compute units and batches whose cache is a few hundred pages.

## Provenance

Third-party origin. No file of the baseline says "adapted from" or "ported from". Two upstream notes point at AITER:
`pa_decode_fp8.py` (only its launch logic became `kernel.py::run`) says the FlyDSL scheduler is "faithful to the C++
kernel", i.e. the scheduler kernel in `pa_metadata.py` re-implements AITER's
`csrc/kernels/mla/metadata/v1_2_pa_device.cuh` (MIT, Copyright (C) 2025-2026, Advanced Micro Devices, Inc.), and the
vendored file carries FlyDSL's Apache-2.0 header without that MIT notice; and kernel-geometry comments refer to AITER's SP3
assembly kernel (`kBlockSize`, `kTileKV`). The scheduler kernel is not called by `run()`, but it is part of the vendored
`pa_metadata.py`, and `inputs.py` ports it, which its first lines say. `buffer_ops.py` cites the LLVM AMDGPUToROCDL source
of the descriptor flag layout in a comment. Every vendored file carries `SPDX-License-Identifier: Apache-2.0` and
`Copyright (c) 2025 FlyDSL Project Contributors` verbatim, so no `baseline/LICENSE` is added.

## Not covered

The upstream test also runs multi-token (MTP) decode with 2-4 query tokens per sequence, per-tensor KV scales, 8 query
heads and sliding windows. Those change the interface (query rows and causal mask, a scale tensor of shape `[1]`) or are
different scenarios; the per-tensor and 8-head variants of the older baseline are the sibling tasks.

## Launcher calls

The direct calls of the jitted launchers replace upstream's `_run_compiled`, and the pointer-argument signature of the
decode launcher is used exactly as upstream passes it.
