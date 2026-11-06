# SiLU-and-mul + FP8 quantization, masked MoE layout, bfloat16, 32 experts, intermediate size 2048

The activation step between the two grouped GEMMs of an expert-parallel MoE layer that uses the
DeepGEMM masked layout (DeepEP low-latency dispatch). Each of the 32 local experts owns `max_tokens`
rows of `y = [gate | up]` (width 4096); only the first `tokens_per_expert[e]` rows are valid. The
operator computes `silu(gate) * up` in float32 and quantizes it to `float8_e4m3fn` with one
float32 scale per 128 columns, `scale = max(max|v|, 1e-10) / 448` (no power-of-two rounding).

- Scenario: 32 local experts (256 routed experts at expert parallelism 8), intermediate size 2048 (the
  DeepSeek-V3 MoE FFN width), group size 128. The directory name of the original plan said `h4096`;
  4096 is the width of `[gate | up]`, the output width is 2048.
- Scale layout: `y_s` is declared `(experts, groups, tokens)`, i.e. the column-major memory layout DeepGEMM
  expects for a logical `(experts, tokens, groups)` scale tensor (vLLM allocates it with strides
  `(T*G, 1, T)`). Candidates write it contiguously in this shape.
- Rows at or beyond an expert's count: vLLM leaves them unspecified. This task makes them part of the
  contract (zero codes, zero scales) because the evaluator checks whole outputs and poisons them first. The
  baseline therefore has an extra zero-fill loop that the vLLM kernel does not have.
- Counts: expert 0 is full, expert 1 empty, the others uniform in `[T/2, T]`.
- Workloads: the capacity `max_tokens` follows DeepEP low-latency sizing (128 tokens per rank times
  the EP world size, 1..128 ranks) plus small decode capacities. Capacities below 160 take the baseline
  under the 10 us floor and are correctness-only.
- Comparison: the codes `y_q` use the default `baseline_relative` bound; they are compared as fp8
  values against the unrounded float64 `v / scale`, within the fp8 rounding the baseline itself has.
  The scales `y_s` are compared against the float64 scale with `tolerance` (`rtol` 7.5e-7, `atol` 0),
  not with `baseline_relative`. The reasons follow.
- Scale bound: the MI355X QA calibration probe ran the reference in float32 PyTorch, which is a correct
  implementation. Against the default bound, its scales failed in all 12 cases. Between 0.03% and 0.6% of
  the scales were over the bound, at most 2.0 times it (t8192). Both the baseline and a float32
  implementation get the scale wrong by only a few float32 roundings, but their errors fall in different
  places, so twice the baseline's per-row error cannot cover the other. The scale error does not grow with
  the shape: a max-reduction has no accumulation. On the path to the scale there are about seven float32
  roundings or hardware approximations: exp (computed as exp2(x * log2 e) on AMD), 1 + e, the reciprocal
  or division, two multiplies, and the multiply by `fl(1/448)` (or a division by 448). Each is within
  1 ulp. In units of u = 2^-24 (half an ulp), these errors add up to at most about 8 u when the group
  maximum has a positive gate. They add up to about 10 u in the rare group whose maximum has a negative
  gate. A CPU emulation over 2^21 groups of bf16 N(0, 1) inputs tried seven correct float32 variants.
  They used PyTorch's sigmoid with division or with multiply-by-reciprocal, exp2 and a reciprocal at
  1 ulp (the baseline's arithmetic) or at 2 ulp, `g / (1 + exp(-g))`, a tanh sigmoid, and a reordered
  product. The largest relative scale error was 5.9 u, and 5.2 u for the baseline-like path.
  The comparator admits `(rtol + u) |y_s|` = 13.6 u. That leaves 2.3x margin over the emulated maximum
  and covers the analytic worst cases. In the same emulation on task-shaped cases (t1 to t2048), the
  float32 reference used at most 25% of this bound and the baseline-like path at most 36%.
- Wrong scales are still rejected. A uniform 16 u shift of the scale (the shifted_16_ulp control) exceeds
  the bound on 59-75% of the scales. The median errors of wrong implementations are 200 times the bound or more:
  dividing by 447 (2.2e-3), rounding `v` to float16 (median 1.7e-4) or to bfloat16 (1.4e-3), and a
  bfloat16 sigmoid or SiLU (1.2e-3). At most 0.41% of their per-group scales fall inside the bound, and one
  scale outside it fails the case. With `atol` 0, the rows past an expert's count must be exactly zero.
- Baseline: vLLM's Triton `_silu_mul_fp8_quant_deep_gemm` kernel (one program per expert and group,
  one warp, software-pipelined token loop), specialized to float8_e4m3fn (the OCP format used on gfx950; the
  old task chose the fnuz format) and group size 128.
- Risk for the GPU audit: the smallest performance cases (T=160, 256) are estimated at 10-13 us.
- Scope versus the previous task: that task also ran group sizes 64 and 256, float16 inputs and tiny hidden
  sizes (128-1024). They are not production configurations of this operator (the DeepGEMM masked path uses
  bfloat16 activations and group size 128), so they are not scenarios here.
- Baseline revision: the kernel follows the upstream Triton fallback at the recorded revision (including the
  multiply-by-reciprocal scale), plus the added zero-fill loop.
