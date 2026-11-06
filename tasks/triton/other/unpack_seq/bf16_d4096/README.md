# unpack_seq, bfloat16, hidden size 4096

Reverses `pack_seq`: the padded tensor `packed [batch, max_len, hidden]` is compacted into
`out [n_tokens, hidden]`, the first `lengths[b]` rows of every sequence stored back to back. vLLM uses it
(`unpack_seq_triton`) to restore the flat token layout after batched decode and speculative-decode
attention in the sparse-attention indexer and the ROCm AITER sparse-MLA backend.

- **Scenario.** bfloat16, `hidden = 4096`, a batch of 8 sequences. The var axis is the total token count;
  `max_len` is derived as `min(n_tokens, ceil(1.5 * n_tokens / 8))`, i.e. the longest sequence is up to
  1.5x the mean. Production sizes the padded tensor to the longest sequence, so the padded input is
  larger than the output.
- **Inputs.** `inputs.py` draws random lengths that sum to `n_tokens`, none above `max_len`, exactly one
  equal to `max_len`, in random order (same generator as the `pack_seq` task). Small `n_tokens` yields
  empty sequences. Padding rows of `packed` hold arbitrary random values and must not be copied. The
  kernel's offsets are 32-bit; the largest padded input has 1.6e9 elements, below 2^31.
- **Comparison: `exact`.** Pure data movement: every output element must match bit for bit. The output is
  poisoned with NaN before each call, so any unwritten row fails.
- **Reference.** float64 concatenation of the valid slices; independent of the kernel.
- **Baseline.** vLLM's Triton `_unpack_seq_triton_kernel` and the launch parameters of `unpack_seq_triton`
  (64 x 64 blocks, 4 warps, 2 stages) at the recorded revision. The wrapper's `lengths.sum().item()` host
  sync and allocation are replaced by the output's shape, as `run()` receives a preallocated output.
- **Workloads.** Total tokens from 2048 to 262144 (128x) are timed; token counts below 2048 are expected
  to run below the 10 us floor and are correctness-only, together with the boundary cases (1, 3, 37,
  1000). Estimated baseline time at 2048 tokens: roughly 10 us (about 17 MB read and 17 MB written); this
  is the case closest to the floor on MI355X.
