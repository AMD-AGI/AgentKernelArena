# Fused MXFP4 quantization and MoE scale sorting, bfloat16, K=1024, 256 experts, top-8

Quantizes the stage-1 output of a MoE layer (one bfloat16 row per token and expert slot, 1024 columns) to MXFP4
(packed E2M1 codes with one E8M0 scale per 32 columns) and, in the same kernel, writes the scales in the sorted,
tiled layout the second grouped GEMM reads (`sorted_ids` order, tiles of 32 rows by 8 scale groups). The layout is
specified in `task.yaml`.

- **Baseline**: AITER's Triton `fused_dynamic_mxfp4_quant_moe_sort` (kernel unchanged, MIT, see `baseline/LICENSE`),
  one program per 128 rows x 32 columns for the quantization plus one per 32 sorted rows x 8 groups for the scales.
- **Scenario**: 256 experts, top-8, width 1024 (the intermediate size of the quantized second-stage input),
  `block_size = 32`.
- **Routing table**: the generator (`inputs.py`) assigns the `n_tokens * 8` (token, slot) pairs to the 256 experts
  uniformly at random with exactly balanced counts, sorts them by expert in token order and pads each expert to a
  multiple of 32 rows with the sentinel `(8 << 24) | n_tokens`, as AITER's `moe_sorting` does. Balanced counts make the
  padded length a closed-form function of `n_tokens` (`sorted_len`), so the sorted output has no rows the operator
  leaves unwritten and every byte can be compared. Real routers are skewed; the kernel's cost does not depend on
  which expert a row goes to.
- **Workloads**: prefill-sized token counts (2048 to 262144; x is 33 MB to 4.3 GB). Decode-sized batches are
  correctness-only. Scalar `token_num` equals `n_tokens` in every case (the framework cannot derive it).
- **Bounds**: both outputs use `exact`. The FP4 codes and E8M0 scale codes are integer encodings of a fully
  specified rounding, so there is no tolerance to choose.
- **Reference**: `reference.py` reproduces the quantization with integer arithmetic on the float32 bit patterns
  (bfloat16 inputs are exact in float32; no higher precision exists for a code) and gathers/tiles the scales
  independently of the kernel. Scale codes of sentinel rows are those of an all-zero group (0), which is what the
  kernel produces for them.
- **GPU audit risks**: the sentinel-row value (0) relies on the kernel's masked load returning zeros and the
  float-to-uint8 conversion wrapping; timing of the smallest performance case (about 13 us estimated) is tight.
