# Dynamic MXFP4 quantization, bfloat16, hidden size 7168

Quantizes each row of a bfloat16 matrix to MXFP4: blocks of 32 elements share an E8M0 scale and elements
become E2M1 codes (two per byte, the even element in the low nibble). Outputs are `x_fp4[T, 3584]` and
`blockscale_e8m0[T, 224]`, both uint8.

- **Scale rule ("even" rounding, AITER's default).** amax is rounded to a power of two by adding a quarter
  of the mantissa field to its fp32 bits (mantissa >= 1.75 rounds up); the E8M0 exponent is
  `floor(log2(.)) - 2`, clamped to [-127, 127], biased by 127. This differs from the round-up
  (`ceil_pow2(amax / 6)`) rule of AITER's HIP `quant_mxfp4`, which the earlier torch2flydsl task used.
- **Codes.** `x / scale` is rounded to nearest-even onto {0, 0.5, 1, 1.5, 2, 3, 4, 6}, saturating at 6, with
  the sign bit preserved (so small negatives become -0, nibble `0x8`).
- **Comparison.** `exact` for both outputs: codes and scales are integers produced by deterministic
  rounding, and the reference reproduces the rule bit for bit (verified against a PyTorch emulation of
  AITER's kernel arithmetic on bfloat16 data).
- **Reference scope.** All-zero blocks and blocks whose amax is below the smallest normal float32 are not
  covered (the production op yields an undefined scale there); the normal-distributed inputs never produce
  them.
- **Baseline.** AITER's Triton `dynamic_mxfp4_quant` kernels (MIT; `baseline/LICENSE`) and the launch
  heuristics of the production wrapper, unchanged. The production wrapper returns the scale as a transposed
  view of an `[N/32, M]` buffer; here the scale output is row-major, which the kernel handles through its
  strides.
- **Workloads.** Token counts at hidden size 7168 (a DeepSeek-V3-class model): decode batches (1, 3, 48),
  an unaligned chunk (1000) and prefill chunks from 2048 to 204800 tokens (100x span). Decode batches are
  below the 10 us timing floor and are correctness-only.
