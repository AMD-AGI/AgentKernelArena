# DeepSeek-V4 paged top-512 transform, float32 scores, width 262208, page size 64

DeepSeek-V4's sparse attention picks, for every decode query row, the 512 compressed (c4) KV
positions its indexer scores highest, and hands the attention kernel their KV-cache slots. For row
`b` with valid length `L = seq_lens[b]`:

- if `L <= 512`, the selected positions are `0, 1, ..., L - 1`, in this order;
- otherwise they are the 512 positions of `scores[b, :L]` with the largest scores, in any order;
- each selected position `p` is written as the slot `page_tables[b, p // 64] * 64 + p % 64`, and
  the remaining output slots of the row are `-1`.

Only the valid prefix may influence the result: scores past `L`, page-table entries past the
pages it covers and the previous contents of `out_page_indices` must not. `metadata` is the
top-k v2 routing plan (`[batch + 1, 2]`: row 0 holds `(cluster_threshold, N)`, rows `1..N` the
`(batch_id, seq_len)` of rows routed to the CUDA cluster path). Every input carries the plan that
routes no row (`cluster_threshold = 2^31 - 1`, `N = 0`); an implementation may read or ignore it.

## Production origin and baseline

SGLang serves DeepSeek-V4 decode on ROCm with `topk_transform_paged_v2`
(`python/sglang/kernels/ops/attention/dsv4/topk.py`; the DeepSeek-V4 model hook forces
`SGLANG_OPT_USE_TOPK_V2` on for HIP), whose kernel is the JIT source
`python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh` with the selection classes of
`python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh`. The baseline is that
kernel, vendored from sgl-project/sglang at `887c401e` (Apache-2.0) as a HIP extension:

- `kernel.hip`: the device code of `topk_v2.cuh` with its `USE_ROCM` branches selected, and its
  host dispatcher `TopKKernel::transform_paged`, rewritten from the tvm-ffi interface to ATen
  tensors (same checks, same ROCm launch) and registered as the arena `run` op.
- `topk_impl.cuh`: `topk_impl.cuh` with the CUDA-only thread-block-cluster path removed.
- `sgl_kernel_utils.cuh`: the few `sgl_kernel` helpers the kernels use (fp16 conversion,
  `AlignedVector`, the 32-lane warp sum, PDL no-ops), flattened from `utils.cuh`, `type.cuh`,
  `vec.cuh` and `warp.cuh`.

On ROCm the kernel runs one 1024-thread block per row. Rows with `L <= 512` take the trivial path;
longer rows build a 4096-bin fp16 coarse histogram, find the threshold bin, collect the positions
above it and the threshold bin's candidates by two fp32 boundaries, and resolve the threshold bin
exactly by radix selection. Rows up to 16384 keep their scores in registers; longer rows are
streamed from global memory twice. The routing plan is a no-op on ROCm (the cluster path is
compiled out), so the baseline never reads `metadata`.

The earlier arena version of this task (a FlyDSL port target) timed the same v2 entry point
through the installed SGLang package. SGLang's AOT kernel `deepseek_v4_topk_transform_512`
(`python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu`) is the older v1 path
(`topk_transform_paged`), which DeepSeek-V4 on ROCm uses only when it also needs the raw
indices, so it is not the baseline.

## Scenario and inputs

Float32 scores of width 262208 (4097 pages of 64 compressed tokens: the c4 indexer capacity of a
1M-token context plus one page), k = 512, page size 64; the number of rows `batch` varies.
`inputs.py` draws:

- `seq_lens`: row `r` takes length range `r % 16` of a fixed decode mix: near-full, half,
  quarter and three-quarter context (each over a 4096-token range), exactly the width and
  width - 1, 16385 (the shortest streamed row), 16384 (the longest in-register row), a random
  in-register length, 513, 512, 511, a random short padded row, 63 or 64, 1 and 0. Every batch of
  16 or more rows exercises every path; the exact lengths change between the two input draws
  while the work per batch stays the same.
- `scores`: each row is a random permutation of 262208 distinct float32 values, so the valid
  prefix is tie-free, in one of two families fixed by the row index. Rows 0-15, 32-47, ... use
  the standard-normal quantiles `Phi^-1((i + 0.5) / 262208)`, shaped like indexer logits with a
  sparse top tail: at most about 55 scores share the coarse (fp16) value bin of the 512th
  largest. Rows 16-31, 48-63, ... use evenly spaced values over `[-0.5, 1)`, like the earlier
  task's uniform scores: their top 512 crowd into a few coarse bins, and 1365 scores share the
  bin of the 512th largest at full width, so an implementation must resolve a crowded threshold
  exactly (the baseline's radix tie selection over more than 1024 candidates). The earlier
  task's uniform `[0, 1)` scores put 2048 scores into that bin at full width, the baseline's
  whole candidate buffer; the `[-0.5, 1)` range keeps the crowded rows well inside it. Every
  batch of 17 rows or more contains both families; batches 1 to 16 hold only normal rows. Scores past `L` are raised above every valid score, so
  reading past `seq_lens` selects wrong positions.
- `page_tables`: distinct random page ids (below 2^24) per row, so slots stay in int32.

## Workloads

Batches of 1 to 2048 decode rows (requests times MTP draft rows). Batch 1, 4, 16, 64, 256, 512,
1024 and 2048 are timed. Up to about 512 rows the GPU holds every row's block at once and the
time is that of the longest rows (one block streaming a 1 MB score row twice); beyond, it grows
with the scores read. Batch 2, 3, 17, 100 and 1000 are correctness-only boundary shapes.

## Comparison

`out_page_indices` compares exactly (`bound: exact`) with `element_order: any_in_full_rows`: a row
with no `-1` (`L >= 512`) is the selected set, compared after sorting, because the operator
defines no order there (the kernel emits in atomic order and the sparse-attention consumer reads
the slots as a set). A row with padding (`L < 512`) has a defined order and is compared exactly,
including the positions of its `-1` padding. For the boundary row `L = 512` this checks the set,
not the order `0..511`. Because the baseline fills full rows in atomic order, two runs on the
same inputs can order a full row differently; the selected set is deterministic, so the QA
auditor's bitwise-determinism note for the baseline is expected and harmless here.
