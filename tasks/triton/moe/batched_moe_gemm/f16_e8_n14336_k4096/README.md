# Batched MoE GEMM, float16, 8 experts, K=4096, N=14336

Expert-major batched MoE GEMM, as used by vLLM's batched (expert-parallel / all-to-all) MoE path:
tokens are already grouped per expert into padded buffers `A[E, max_tokens, K]`, `expert_num_tokens[e]`
says how many rows of each are valid, and one launch computes `C[e, :n_e] = A[e, :n_e] @ B[e]^T` for all
experts, skipping the work of empty experts. Shapes are Mixtral-8x7B's expert up-projection (8 experts,
K=4096, N=14336); `max_tokens` is the variable axis. Float32 accumulation, float16 output.

Output semantics: rows `>= expert_num_tokens[e]` of `C` are defined to be zero (the previous task package
allocated `C` with `torch.zeros`). The baseline kernel writes those zeros itself, so no separate memset is timed.
Padded rows of `A` hold random data, so a kernel that fails to mask them is caught.

`inputs.py` draws `expert_num_tokens` uniformly in `[0, max_tokens]` per expert, then forces expert 0 full,
expert 1 empty and expert 2 to a single token, so every case covers a full tile column, the empty-expert skip
and a ragged one-row tail. Case `m33_empty` (id ending in `_empty`) sets every count to zero, so the whole output must be zero.
Case sizes: decode-sized buffers (16-64 rows) are weight-bandwidth bound
(940 MB of weights, well above the 10 us floor); 1-3 and ragged sizes are correctness-only.

Baseline: vLLM's `batched_triton_kernel` (with its `expert_triton_kernel` and `moe_mmk` helpers folded into
one kernel, as in the earlier task package this replaces), simplified to the unquantized case, fixed
64x64x32 tiles, and extended to write zeros for padded tiles. It is a simplified adaptation rather than the
verbatim upstream kernel (upstream carries quantization scales, activation fusion and config plumbing).
Comparison is the default baseline-relative bound against the float64 reference.
