# pack_seq, bfloat16, hidden size 4096

Packs variable-length sequences that are stored back to back as rows of `x [n_tokens, hidden]` into a
padded tensor `out [batch, max_len, hidden]`; rows past a sequence's length hold `pad_value`. vLLM uses
it (`pack_seq_triton`) to batch decode and speculative-decode query tokens for the sparse-attention
indexer and the ROCm AITER sparse-MLA backend, and `unpack_seq_triton` reverses it.

- **Scenario.** bfloat16, `hidden = 4096`, a batch of 8 sequences, `pad_value = -1` (a finite value;
  production's default `-inf` would be rejected by the baseline sanity check of the framework, and the
  copy semantics do not depend on the value; the workload format cannot express infinity). Two correctness cases
  use `pad_value = 0`, as the original task's tests did). The var axis is the total token count; `max_len` is derived
  as `min(n_tokens, ceil(1.5 * n_tokens / 8))`, i.e. the longest sequence is up to 1.5x the mean.
- **Inputs.** `inputs.py` draws random lengths that sum to `n_tokens`, none above `max_len`, exactly one
  equal to `max_len`, in random order. Small `n_tokens` yields empty sequences. The kernel's offsets are
  32-bit; the largest case produces 1.6e9 output elements, below 2^31.
- **Comparison: `exact`.** Pure data movement: every output element, including padding, must match
  bit for bit (the default bound would only add rounding slack). The output is poisoned with NaN before
  each call, so any unwritten padding or data row fails.
- **Reference.** float64 allocation filled with `pad_value`, sequence slices copied in; independent of
  the kernel.
- **Baseline.** vLLM's Triton `_pack_seq_kernel` and the launch parameters of `pack_seq_triton`
  (64 x 64 blocks, 4 warps, 2 stages) at the recorded revision. The wrapper's `lengths.max().item()` host
  sync and allocation are replaced by the output's shape, as `run()` receives a preallocated output. The
  kernel pads every output tile and then overwrites valid rows (two stores), which is what is shipped.
- **Workloads.** Total tokens from 2048 to 262144 (128x) are timed; token counts below 2048 are expected to
  run below the 10 us floor and are correctness-only, together with the boundary cases (1, 3, 37, 1000).
  Estimated baseline time at 2048 tokens: roughly 10 to 20 us (about 17 MB read and 25 MB written
  twice); this is the case closest to the floor on MI355X.
