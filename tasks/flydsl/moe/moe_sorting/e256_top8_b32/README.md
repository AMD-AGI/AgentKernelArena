# MoE sorting, 256 experts, top-8, block size 32

The sorting step in front of the grouped GEMMs of a MoE layer (DeepSeek-style router: 256 routed experts, 8 per
token): a counting sort of the `(token, slot)` assignments by expert with each expert padded to blocks of 32 entries,
packed ids `(slot << 24) | token`, the matching routing weights, the expert of every block and the valid length. As in
AITER's `moe_sorting`, the operator also zeroes the layer's output buffer `moe_buf` (`[n_tokens, 7168]` bfloat16),
which the second GEMM later accumulates into; at prefill sizes this memset is a large part of the time.

- **Baseline**: the FlyDSL kernels `moe_sorting_kernel` with their AITER-style host dispatch: one kernel when
  `n_tokens <= 16` (everything in LDS), two kernels up to 2048 tokens, four kernels (workspace clear, scatter, count,
  prefix/gather) above, with the workspace allocated per call. Kernel code unchanged (Apache-2.0). `buffer_ops.py`
  and `meta.py` are the FlyDSL runtime helpers that FlyDSL 0.3.2 does not ship, with the `os`/`traceback` uses removed
  (see `task.yaml`).
- **Output order**: within an expert the entries are in increasing (token, slot) order, which the kernels produce
  deterministically (the scatter writes a per-expert token mesh that is then read in token order), so all outputs
  compare `exact`: they are integers, copied float32 weights and zeros.
- **Routing table**: `inputs.py` draws, for every chunk of 32 tokens, a random permutation of the 256 experts and
  deals it to the chunk's 32 x 8 positions, so every token's experts are distinct (the kernels store one slot per
  (token, expert) pair) and expert counts are balanced to within one. Balanced counts make the padded length a closed-form
  function of `n_tokens`, so every output entry is written (production sizes the buffers for the worst case and leaves
  the tail undefined, which cannot be compared). Real routers are skewed; the kernels' cost does not depend on which
  expert an assignment goes to except through the per-expert counts.
- **Workloads**: 2560 to 262144 tokens are performance cases. The estimate of 10 us or more per case rests on the
  four-kernel path, the `moe_buf` memset (37 MB at 2560 tokens) and the workspace clear (`256 * n_tokens` bytes);
  decode-sized batches, 1000 and 2048 tokens are correctness-only and cover the single-kernel and two-kernel paths.
  `num_experts` and `block_size` are scalar inputs with their scenario values as defaults.
- **Reference**: PyTorch stable sort by expert, block padding and tiling written independently of the kernels
  (verified on CPU against a per-expert `torch.where` loop for 1 to 5000 tokens).
- **GPU audit risks**: the smallest performance case is the timing risk (estimated 12 to 15 us); the workspace
  allocation inside `run()` and the compiled-function dispatch cache must be capturable in a CUDA graph; the helper
  modules were adapted for the candidate import rules without GPU validation here.
