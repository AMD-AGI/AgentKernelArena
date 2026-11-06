# Linear layer + log_softmax, float32, hidden 512, vocabulary 4096

`out = log_softmax(x @ weight^T + bias, dim=-1)`: the masked-language-model head of BERT (`nn.Linear(512, 4096)`
followed by `nn.LogSoftmax(dim=-1)`) applied to `n_tokens` token rows. Everything is float32 with float32
accumulation; the log_softmax runs over the 4096 vocabulary entries of each row.

- **Merged sources.** `hip2hip/MaskedLanguageModel` and `torch2hip/gpumode/8325_MaskedLanguageModel` are the same
  operator (gpumode 8325). The old inputs were 4D `[B, 4, 4, 512]`; the leading axes are flattened into one
  `n_tokens` axis since the operator acts on rows independently.
- **Inputs.** `x` is standard normal. `weight` and `bias` follow PyTorch's default `nn.Linear` initialization
  (uniform within `+-1/sqrt(512)`), so the logits have a standard deviation of about 0.6 and the outputs are
  about `-8.3` with a spread of a few tenths.
- **Baseline.** The provided HIP kernel of `hip2hip/MaskedLanguageModel` (`hip_8325_MaskedLanguageModel_ref.hip`,
  task-original, not an upstream production kernel), float32 only: a transpose kernel produces `weight^T` on
  every call, then one block per row recomputes the logits three times (row maximum, sum of exponentials,
  output) so nothing is stored. The old initial candidate (`hip_8325_MaskedLanguageModel.hip`) is the same
  kernel with double-precision accumulation, added only to make the old fixed-tolerance check pass; it is not
  the baseline here. The host-side `contiguous()` copy, the output allocation and the pybind module are gone: the
  kernel writes the preallocated output and the transpose scratch is a per-call allocation. One change from the
  source: each block reduction now ends with a `__syncthreads()` after reading its result, because the original
  reused the same shared-memory slot for the maximum and the sum reductions without a barrier in between (a race
  between a slow reader of the maximum and the first writer of the sum).
- **Comparison override.** `baseline_relative` with `k = 6` instead of 2. The baseline's per-row error is
  dominated by its float32 sum of 4096 exponentials, which shifts every element of a row together; an equally
  accurate candidate that reduces in another order (torch's fp32 GEMM, a tree reduction, double accumulation
  with a float32 log-sum-exp) differs from the baseline by about that much, and with `k = 2` these fail on
  a few elements (peak 1.05x of the bound), while `k = 6` leaves a margin of about 40% on CPU emulations
  of those candidates. A candidate that is wrong (a missing bias, a wrong log-sum-exp) is off by O(1).
  The QA calibration probe (a plain float32 PyTorch evaluation of the reference, the kind of candidate a correct
  float32-accumulating kernel resembles) still exceeds the `k = 6` bound on a handful of elements in every
  large case: a violation fraction of 2.4e-7 to 3.3e-6 and a peak of 1.03x to 1.38x of the bound, with a bias
  z-score of about 0.6 (unbiased). These are tail elements of a long-tailed per-element error, not a systematic
  shift, and raising `k` to its maximum of 8 would still leave a peak of 1.04x. The comparison therefore also
  sets `max_violation_fraction = 1e-4`, about 30x above the largest measured fraction of a correct float32
  evaluation (at most 0.4 elements may fail in a 4096-element row, so the single-row cases stay strict, and the
  `n1` case still fails any element). A wrong candidate (a missing bias, a wrong log-sum-exp, a 16-ulp shift,
  zeros, stale or half-written outputs) violates the bound on a large fraction of elements, so the audit's
  known-wrong controls are still rejected.
- **Workloads.** 512 to 65536 token rows (128x span; the largest output is 1 GB). The baseline needs
  `3 * N * 512 * 4096` float FMAs executed by a one-block-per-row kernel whose per-thread loop is
  latency-bound, so even 512 rows take far longer than the 10 us floor. Token counts 1, 3, 48 and 1000 are
  correctness-only boundary cases.
- **Risk.** HIP build and timing were not verified on a GPU. The baseline's per-call weight transpose
  (8 MB moved) is part of the measured time.
