# Expert token counts, 256 experts, top-8

Histogram of a mixture-of-experts routing table: for each of 256 experts, how many entries of
`topk_ids [n_tokens, 8]` name it. Entries of `-1` (assignments not served) are ignored. vLLM runs this
before building the expert-ordered layout of its DeepGEMM / DeepEP MoE path, to size each expert's slice.

The baseline is vLLM's `_count_expert_num_tokens` Triton kernel (one program per expert scanning the whole
routing table), without the optional `expert_map` remapping, which the scenario does not use.

## Comparison bound

The output is an integer histogram, so it is compared with the `exact` bound against `torch.bincount`. Any
tolerance would accept a wrong count: an expert's count is either right or the routing that follows it
(expert slice sizes, scatter offsets) is wrong. The baseline is deterministic (integer sums), so the exact
bound does not penalize a correct kernel.

## Workloads

Token counts of decode batches, prefill chunks and batched prefills. The kernel's cost grows linearly with the
table (every expert's program scans all of it). The MI355X QA audit measured the baseline (CUDA-graph,
cold-cache median) at 9.9 us for 2048 tokens, 12.2 us for 4096, 27.8 us for 16384, 88.5 us for 65536 and
329 us for 262144: about 8.3 us of fixed cost plus 1.22 ns per token. The 2048-token chunk is below the 10 us
floor and is correctness-only, with the decode batches and the 1000-token prefill. The smallest performance
case is 4096 tokens; the largest is 524288 tokens (about 650 us by that fit), which keeps the span at 128x.
524288 tokens is the MoE input under data-parallel attention with vLLM's default all-gather MoE dispatch:
8 data-parallel ranks, each with a 65536-token batch, all-gathered before routing to the 256 experts. Counts
above 65536 exceed a single rank's chunked-prefill batch and stand for such gathered or batched prefills; they
are the only way to keep a 100x span while every timed case stays above the timing floor.

Inputs are generated with skewed expert popularity and about 6% `-1` entries. The largest case's table is
16 MB of int32; its generator draws a 524288 x 256 float32 score matrix (512 MB) once.
