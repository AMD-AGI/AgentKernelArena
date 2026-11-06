# Request-to-global index conversion, int32, block size 64, top-k 2048

Sparse attention (DSA-style) selects `topk` request-local token positions per query token.
Before the attention kernel reads the paged KV cache, each position is translated to a global
cache slot: `block_table[req_id[t], i // 64] * 64 + i % 64`. Entries equal to -1 (unused) and
positions whose block index lies beyond the block table become -1.

Production origin: vLLM v1 `vllm/v1/attention/backends/mla/sparse_utils.py`
(`_convert_req_index_to_global_index_kernel`), used by the sparse MLA backends. Block size 64 and
top-k 2048 are the DeepSeek sparse-attention values.

## Scenario and workloads

64 requests, a block table of 2048 blocks per request (131072 tokens of context), all tensors
int32. The var axis is the number of query tokens. Performance cases are prefill-sized batches
(2048 to 262144 tokens, a 128x span); 262144 tokens is a batched-prefill upper bound. A row of
2048 int32 indices read and written is 16 KiB, so batches below about 2048 tokens run under the
10 us floor and are correctness-only. Index generation (inputs.py): about 10% of entries are -1
and about 0.5% lie beyond the block table, so both guard paths are checked; the rest are
uniform over the table.

## Correctness

The reference is an exact integer computation (chunked by rows to bound memory). The output is
integer slot ids, so the default float bound does not apply: it is compared bit for bit
(`exact`). The optional valid-count output of the upstream kernel is not
part of the task.

## Baseline

The plain translation kernel of vLLM's `sparse_utils.py` (vendored from the simplified task
version, which omits the upstream prefill-workspace, compaction and decode-context-parallel
options not used by this scenario), wrapped in `run()` with the tile width (128) and block size
(64) as module constants and the valid-count path disabled.
