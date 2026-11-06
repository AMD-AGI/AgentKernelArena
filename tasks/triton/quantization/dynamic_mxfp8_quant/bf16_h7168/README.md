# Dynamic MXFP8 quantization, bfloat16, hidden size 7168

Quantizes each row of a bfloat16 matrix to MXFP8: blocks of 32 elements share an E8M0 scale and elements
become `float8_e4m3fn`. Outputs are `y[T, 7168]` and `scale[T, 224]` (uint8 E8M0 bytes).

- **Scale rule.** amax is rounded to a power of two by adding a quarter of the mantissa field to its fp32
  bits (mantissa >= 1.75 rounds up); the E8M0 exponent is `floor(log2(.)) - 8`, clamped to [-127, 127],
  biased by 127. Elements are multiplied by `2^-exponent` (exact) and rounded to nearest-even to E4M3; the
  largest scaled magnitude stays below 448, so nothing saturates.
- **Comparison.** `exact` for both outputs. The earlier harness allowed codes within one step; the scaled
  value is exactly representable in float32, so a single round-to-nearest-even conversion is deterministic
  and the reference (float32 value cast by the evaluator to `float8_e4m3fn`) is compared bit for bit.
  Verified against a PyTorch emulation of AITER's kernel arithmetic. If the GPU audit shows the hardware
  conversion differing from round-to-nearest-even, that is a finding to resolve, not a reason to loosen it.
- **Reference scope.** All-zero blocks and blocks whose amax is below the smallest normal float32 are not
  covered (the production op yields an undefined scale there); the normal-distributed inputs never produce
  them.
- **Baseline.** AITER's Triton `dynamic_mxfp8_quant` kernels (MIT; `baseline/LICENSE`), unchanged, one
  program per row holding the full (power-of-two padded) row. The old triton2flydsl and torch2flydsl tasks
  are merged here: both used the same operator, only at tiny shapes (at most 128 x 1024).
- **Workloads.** Token counts at hidden size 7168 (a DeepSeek-V3-class model): decode batches (1, 3, 48), an
  unaligned chunk (1000) and prefill chunks from 2048 to 204800 tokens (100x span). Decode batches are below
  the 10 us timing floor and are correctness-only.
