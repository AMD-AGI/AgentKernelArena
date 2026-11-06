# SiLU-and-mul with MXFP4 quantization and sorted scales, bfloat16, I=1024, top-2

The epilogue of the split-K first grouped GEMM of a MoE layer: for each (token, slot) row of the bfloat16 GEMM
output `[gate | up]` it computes `silu(gate) * up`, quantizes the result to MXFP4 (packed E2M1 codes, one E8M0 scale
per 32 columns) and writes the scales in the sorted, tiled layout read by the second grouped GEMM. The exact
rounding rules and the layout are specified in `task.yaml`. Note the two coordinate systems: the packed codes are
indexed by the row's own position `token * 2 + slot`, the scales by the row's position in `sorted_ids`.

- **Baseline**: the FlyDSL kernel `silu_and_mul_fq` (one block of 256 threads per sorted row, buffer loads and
  stores, hardware `exp2`/`rcp` for the sigmoid), kernel algorithm unchanged apart from the addressing adaptation
  below. Its helper modules (`buffer_ops.py`, `vector.py`, `meta.py`, Apache-2.0, derived from the FlyDSL runtime
  library) are included because FlyDSL 0.3.2 does not ship `flydsl.expr.buffer_ops`; they differ from upstream only
  by the removed `os`/`traceback`/star imports listed in `task.yaml`. See `baseline/LICENSE`.
- **Baseline adaptation (4 GiB input)**: in the 524288-token case x is exactly 4 GiB (2^20 rows of 4096 bytes).
  Upstream addresses x as a 32-bit byte offset from one buffer descriptor at the tensor base, and a descriptor
  covers at most 2^32 - 1 bytes. The last dword of the last row was therefore out of range and loaded as 0. The MI355X
  audit saw exactly one mismatched code byte in `t524288` (both seeds) and none in any other case. The baseline now
  builds the x descriptor per row from the row's 64-bit address. The row index is block-uniform (every thread reads
  `sorted_ids[bid]`), so `readfirstlane` keeps the descriptor in scalar registers. Offsets inside the row, the loads,
  the arithmetic and the stores are unchanged. Chunked launches (as used for 4 GiB softmax inputs) cannot replace this
  here, because `sorted_ids` can point at any row of x. Compiled for gfx950 with FlyDSL 0.3.2 in compile-only mode
  (no GPU), the ISA differs from upstream only by one `v_readfirstlane_b32` and a 64-bit scalar add for the
  descriptor base. Both `buffer_load_dwordx2` per thread remain, now at in-row offsets.
- **Scenario**: 64 experts, top-2, intermediate size 1024, FP4 output, SiLU, no bias, gate/up stored separated
  (`gui_layout=False`), 32-row blocks.
- **Routing table**: `inputs.py` assigns the `n_tokens * 2` rows to the experts uniformly at random with exactly
  balanced counts and sorts them by expert in token order, padding each expert to a multiple of 32 rows with the
  sentinel `(2 << 24) | n_tokens`, as AITER's `moe_sorting` does. Balanced counts make the padded length a closed-form
  function of `n_tokens`, so every sorted row is written and every output byte can be compared. Every row appears
  exactly once, so every code byte is written. The kernel writes scale 0 for sentinel rows.
- **Workloads**: 4096 to 524288 tokens (x is 34 MB to 4.3 GB) are performance cases; decode-sized batches and 2000
  tokens (estimated below the 10 us floor) are correctness-only. `token_num` equals `n_tokens` in every case.
- **Bounds**: both outputs use `exact`; they are packed codes and E8M0 exponents of a fully specified rounding. The
  kernel's `exp2`/`rcp` carry about 5e-7 relative error, which would flip a few of the millions of rounding decisions
  of a large case. To keep the exact comparison decided by the specification only, the generator moves each value
  whose magnitude lies within 5e-6 (relative) of a decision (scale-exponent threshold, FP4 grid midpoint) to the
  next bfloat16 neighbour of `up`; about 1e-4 of the elements are moved. Checked on CPU against a float32 pipeline
  with random one-ulp errors in `exp2` and reciprocal: no scale or code differed on cases up to 4096 tokens.
- **Reference**: float64 `silu(gate) * up`, then the quantization decided in float64: scale exponent from the group
  maximum (mantissa 1.5 rounds up, minus 2), codes round to nearest with ties to the even code and keep the sign of
  zero, saturating at 6. The tiling is performed independently of the kernel.
- **GPU audit risks**: the smallest performance case (about 12 us estimated) is tight. The 64-bit row-base
  adaptation was checked only by compiling it to gfx950 ISA on a machine without a GPU.
