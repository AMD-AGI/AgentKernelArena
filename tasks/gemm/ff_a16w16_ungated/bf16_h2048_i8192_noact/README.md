# Ungated bf16 feed-forward block, hidden 2048, intermediate 8192, none

`out = (x @ w_up^T) @ w_down^T` in bfloat16 with float32 accumulation: AITER's `ff_a16w16_nogate`, two
`gemm_a16w16` launches with no activation in the first. The intermediate `[n_tokens, 8192]` activations
are rounded to bfloat16 between the GEMMs (the float64 reference does not round them; the baseline's own error,
which includes that rounding, sets the bound). `w_down` is the `[hidden, intermediate]` weight of the down
projection; production passes its transpose, which `run()` does as a view.

- **Source.** `triton2flydsl/aiter/ff_a16w16` is the only source. Its suite mixed five `(batch, hidden,
  intermediate)` shapes and four activations; this task fixes one production pair (hidden 2048, intermediate
  8192) and one of the four activations of the old suite (`silu_exp2`, `gelu_tanh`, `relu`, none); each is its own sibling scenario under `tasks/gemm/ff_a16w16_ungated/`. This one uses no activation. Other widths are separate scenarios.
- **Baseline.** The standalone Triton port from the old task, from the AITER sources listed in
  `task.yaml`. The tuned-config lookup of production is replaced by one static tile (128x128x64, 4 warps, no
  split-K) that the old task already used; the gluon/gfx1250 backend and split-K reduce kernel were never part of it.
  The intermediate buffer is allocated inside `run()`; the second GEMM writes straight into `out`.
- **Inputs.** Normal weights with the standard deviations `1/sqrt(intermediate)` (`w_up`) and `1/sqrt(hidden)`
  (`w_down`) of the AITER test generator.
- **Workloads.** Token counts 1 to 8192. The weights alone are 67 MB, about 15 us to stream, so even a batch-1
  decode exceeds the 10 us floor; the 8192-token case is about 0.55 TFLOP. Counts 3, 48 and 1000 are correctness-only
  boundary cases (together with the batch-1 case, which is also timed). The largest case holds a 537 MB float64
  intermediate in the reference.
- **Comparison.** Default `baseline_relative`; no override.
- **Risk.** Triton build and timing are not verified on a GPU. At batch 1 the static 128x128 tile is wasteful,
  so decode-sized baselines are slower than a tuned production config would be.
