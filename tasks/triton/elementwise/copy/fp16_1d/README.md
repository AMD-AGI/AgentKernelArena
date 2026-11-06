# Copy, float16, 1D

`out = x` for a contiguous one-dimensional float16 tensor. This is a memory-bandwidth calibration
task (one read and one write per element); the headroom over the baseline is small by nature.

- **Origin.** Merges two old tasks: the Inductor-extracted identity kernel (`refk_identity`, the
  baseline) and the ROCm Triton block-pointer copy test (`test_block_copy`), which is the same operation
  (its padding options and its half-length copy are test-harness details, not operator semantics).
  The upstream revision of the extracted kernel was not recorded.
- **Baseline.** The old autotuned Triton kernel (XBLOCK 128 to 1024) in `run()`. The kernel upcasts to
  float32 and stores back to float16, which is lossless, so the comparison is `exact`.
- **Workloads.** Performance cases span 10.5M to 1.07G elements so each moves at least ~40 MB.
- **GPU-audit risk.** The baseline uses `triton.autotune`; the first call tunes outside graph capture
  only if the evaluator runs an eager call first (correctness does).
- **Bound.** `exact` instead of the default: a copy is value-preserving (the float32 round trip of the
  kernel is lossless), so any difference is a defect and no approximation allowance is justified.
- **Inputs.** `inputs.py` plants signed zero, +/-inf, the largest finite float16 and a few round values at the
  start of each tensor (the old suite did not test them for copy, and the old flip suite did); NaN is omitted
  because outputs are poisoned with NaN.
- **Correctness-only sizes.** Besides the minimum and an odd size, 127, 129, 4096, 65536 and 1048576
  (the sizes of the old identity and block-copy suites). The old block-copy dtype variants (bool and others)
  and its half-length padding test are harness details of a different kernel, not covered by this scenario.
