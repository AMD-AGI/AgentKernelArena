# gelu_tanh, bfloat16, flat

Operator: elementwise `0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))` over a flat bfloat16 tensor,
the activation of ungated GELU-tanh MLPs and AITER's `gelu_fast` op (the former
`torch2flydsl/gelu_fast_kernel` task, whose shapes were `[m, n]` with m = 1..4096, n = 4096..8192).

Scenario: bfloat16, one flat axis `numel` (1 to 838,860,800). The operator is shape-agnostic, so
the sizes are token count times intermediate width of real MLPs (2048 to 65536 tokens at widths
4096, 12800, 16384).

Workloads: five performance cases from 8,388,608 elements (about 11 us estimated) to 838,860,800 (100x
span). Sizes 1, 3, 1000 and 1,048,583 (not a multiple of any block) are correctness-only; the last
is also below the 10 us floor.

Reference: float64 tanh GELU computed in chunks, stored in float64. The default `baseline_relative`
bound is used; inputs are normal with std 2.

Baseline: there is no production Triton kernel to vendor. The baseline is a standalone Triton
rewrite of the pointwise kernel PyTorch Inductor generates for `gelu(x, approximate="tanh")`
(flat grid of 2048-element blocks, float32 math, one rounding on store), with `tanh` expressed as
`2 sigmoid(2x) - 1`. `baseline/LICENSE` is the PyTorch BSD-3-Clause license; provenance names
`pytorch/pytorch` v2.9.0 and says it is a rewrite.

GPU risks: not yet run. The largest case holds a 6.7 GB float64 reference output plus comparison temporaries (estimated under 40 GB); the 8.4M-element case sits near the 10 us floor.
