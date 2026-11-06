# Sparse MLA prefill, DeepSeek-V4 (64 heads, head size 512, top-k 512)

DeepSeek Sparse Attention prefill: every query token attends only to a selected subset of the
MLA-latent KV pool, given as ragged CSR lists (`indices`, `indptr`). The 512-wide latent row
(448 NoPE + 64 RoPE) is both key and value; all 64 query heads share the single latent head. The
softmax is online over the selected rows and a row with no indices yields zeros. Attention sinks are
not part of this task.

**Production origin.** vLLM's ROCm DeepSeek-V4 backend, `_sparse_attn_prefill_ragged_kernel` in
`vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`. The kernel was about 10 % of the timeline wall of a
profiled DeepSeek-V4-Flash serving session on MI355X.

**Scenario.** bfloat16, 64 heads, head size 512, `index_topk` 512, 8192-row KV pool, `n_queries`
varies. The input generator gives most rows the full 512 indices (random, sorted, drawn with
replacement as the previous harness did) and every 16th row one of the lengths
`0, 1, 15, 16, 17, 31, 32, 33, 127, 255, 511, 512`, so empty rows and partial key blocks are always
exercised. `indices` has capacity `n_queries * topk`; rows are packed from offset 0 and the tail beyond
`indptr[n_queries]` is unused (`-1`).

**Workloads.** The traced prefill buckets (64, 1073, 7211 queries) are correctness cases together with
the boundary cases; the performance cases are power-of-two query buckets from 128 to 16384. The
trace records this leaf with empty input dims, so the KV pool size is a representative
reconstruction (any size at least `topk` is valid for a pure gather-attention kernel).

**Baseline.** vLLM's Triton kernel, vendored unchanged; `run` follows the launch code of
`_rocm_sparse_attn_prefill_ragged_triton` and compiles the attention-sink branch out. The dense-to-ragged
packing kernel (`build_ragged_indices_from_dense`) is not used: the task's contract starts at the CSR
lists.

**Comparison.** Default `baseline_relative` (bfloat16 probabilities in the second matmul are the
dominant error). The reference returns float32 (not float64) only to bound its memory at the largest
case; it is computed in float64 internally. An empty row's softmax denominator is replaced by 1 (a
non-empty row's sum is at least 1, since it includes `exp(0)` for its maximum score), so the reference
yields the empty row's zeros in any floating dtype. A float32 evaluation of the reference passes the
default bound: against an emulation of the baseline's numerics (float32 online softmax over 16-key
blocks, bfloat16 probabilities; about 56 dB SQNR), its largest error is 0.33x the allowed bound and its
bias z-score at most 0.003 on cases `q3` to `q2048`, while the 16-ulp shift and fp8-rounding controls
are rejected. The default bound therefore accepts a correct float32-accumulating kernel.
