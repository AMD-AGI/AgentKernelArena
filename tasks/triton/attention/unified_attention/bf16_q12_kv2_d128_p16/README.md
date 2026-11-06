# Unified paged attention, bfloat16, 12 query / 2 KV heads, head size 128, block size 16

Causal attention of mixed decode and chunked-prefill batches over a paged KV cache addressed by a
block table. Each sequence has `q_len` new query tokens (1 for decode) and `seqused_k[s]` KV tokens
including them; the query tokens attend causally to everything at or before their absolute position.
Scale is `head_dim ** -0.5`, scores and accumulation are float32, the output is bfloat16.

**Production origin.** AITER's Triton `unified_attention`, which serves the `unified_attention` kernel of
MiniMax-M2.7 (12 query heads over 2 KV heads, head size 128, block size 16; shape taken from the
production serving traces on MI355X, where it was 16% of GPU time at 64 decode sequences with 1024 tokens
of context). Both old tasks for this operator (`image_kernel/mi355x_vllm_triton_unified_attention` and
`triton2flydsl/aiter/unified_attention`) call the same AITER function; the other head configurations of the
image task (head size 64, 256, 512, fp8 KV) are separate scenarios.

**Scenario and workloads.** One scenario: bfloat16 query and cache, 12/2 heads, head size 128, 16-token
pages, no sliding window, softcap, sinks or fp8. The varying axes are `num_seqs`, `q_len` and `ctx_len`
(the longest KV length). The generator (`inputs.py`) draws ragged KV lengths (the longest sequence has
`ctx_len`, the others between `ctx_len / 2` and `ctx_len`, never fewer than `q_len`) and scatters the pages
of all sequences over the pool with a random permutation, so cache reads are not contiguous. Performance
cases: decode batches (16 to 256 sequences at 1K to 16K tokens, a single sequence at 64K) and prefill chunks
(2048 to 8192 query tokens); they span the 2D kernel (prefill, short context) and the segmented 3D kernel with
its reduction. The small decode batches, the 512/513-token context pair around the 2D/3D switch, and the
small ragged prefill shapes are correctness-only. Correctness-only cases also reach the other dispatch branches: a 2D prefill with fewer than 256 query rows (`s64_q16_c1024`), the 2D pure-decode configuration at a large batch (`s640_q1_c600`) and a 3D launch on a prefill batch (`s2_q300_c700`).

**Baseline.** AITER at `652f2d4b`, vendored: the two attention kernels and `reduce_segments` verbatim, plus the
host dispatch that selects the 2D kernel (prefill, context up to 512, or more programs than the GPU can hold) or
the 3D kernel with `reduce_segments`, with its tuned warp, stage and segment-count tables. The dispatch is
reduced to the gfx950 bfloat16 causal path of this scenario (the gfx1250 Gluon kernels, RDNA and fp8 branches
are dropped). `max_seqlen_q` and `max_seqlen_k`, which production receives from the scheduler, are taken from
the tensor shapes (`q.shape[0] // num_seqs` and the block-table capacity in tokens); both lie on the same side
of every dispatch threshold as the true values because the generator's pool capacity differs from `ctx_len`
by less than one 16-token page and 512 is a multiple of 16. The 3D path's segment buffers, which vLLM and AITER
callers keep as persistent workspace, are allocated inside `run()`; inside a CUDA graph this costs nothing per
replay. The kernels are adapted from vLLM's Triton unified attention; the license of the vendored files is AITER's
MIT license (`baseline/LICENSE`).

**Comparison.** Default `baseline_relative` on the bfloat16 output.
