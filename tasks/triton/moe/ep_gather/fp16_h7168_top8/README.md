# Expert-parallel gather, float16, hidden 7168, top-8

Last step of vLLM's DeepEP / DeepGEMM MoE path: each token sums the weighted outputs of its (up to) 8 experts,
which sit in an expert-ordered buffer, into one float16 row. Accumulation is float32; the reference uses float64.
Assignments with a negative expert id are skipped and their indices are never read, so the generator only
guarantees in-range indices for them (real kernels leave them unwritten).

The baseline is vLLM's `_fwd_kernel_ep_gather` Triton kernel and its launcher (BLOCK_D = 1024, 2 warps, grid of
hidden blocks x up to 1024 token programs), without the optional `expert_map`. The hidden size 7168 and 256
experts / top-8 follow DeepSeek-V3. The buffer holds one row per assignment (`n_tokens * 8`), the unpadded size of
the expert-ordered layout. The default `baseline_relative` bound is used.

Workloads: moved bytes are about 130 KB per token, so batches under roughly 256 tokens fall below the 10 us
timing floor and are correctness-only. At 32768 tokens the input buffer is 3.8 GB.

This is the float16 sibling of `bf16_h7168_top8` (the original arena task gathered into float16 outputs); only the
expert-output and result dtype differ. Everything else, including the unchanged kernel, is identical.
