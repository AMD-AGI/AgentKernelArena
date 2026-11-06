# Fused MoE router with logit soft-capping, Grok-1

For every token, `logits = x @ router_weight^T` (bfloat16 inputs, float32 accumulation), then
`moe_softcapping * tanh(logits / moe_softcapping)` with `moe_softcapping = 30`, then the 2 largest capped
logits are selected. The output weights are the softmax probabilities over all 8 experts of the selected
experts (not renormalized over the top 2), highest first, as float32; ids are int32.

Production origin: the Grok-1 MoE router (8 experts, hidden size 6144, top-2, soft-cap 30) as served by SGLang,
`python/sglang/srt/layers/moe/router.py`.

## Baseline

Vendored from sgl-project/sglang at `887c401e151e2d1b082d74ef8769a23eef5c465c`: the CUDA-core kernel
(`tl.sum` reduction, one program per token) and the tensor-core kernel (`tl.dot`, 32 tokens per program).
`run()` inlines the upstream dispatcher: batches of 512 tokens or more use the tensor-core kernel, smaller
batches the CUDA-core kernel. Changes: outputs are caller-provided, the unused correction-bias and
deterministic-inference arguments are dropped (the bias pointer is `None`), the HIP warp cap is inlined.
Kernel code is unchanged. The kernels compute row offsets in 32-bit integers, so the largest token count is
262144 (262144 x 6144 < 2^31).

## Workloads

Token counts from decode batches to batched prefill; hidden size, experts and top-k are fixed by the
scenario. Cases below 2560 tokens run under the 10 us timing floor (x is 12 KiB per token) and are
correctness-only. The timed cases span both upstream kernels only above 512 tokens; the CUDA-core kernel is
covered by correctness cases (t1 to t256).

One extra correctness case (`t77_nocap`) sets `moe_softcapping = 0`, which the upstream kernels and the old task
treat as "soft-capping disabled". The old task also covered other routers (16 to 64 experts, top-1 and top-4,
a correction bias); those are not Grok-1 scenarios and are not part of this task.

## Inputs and comparison bounds

`inputs.py` draws `x ~ N(0, 0.1^2)` in bfloat16 (router weights are `N(0, 0.1^2)`, giving logits of
standard deviation about 0.8) and redraws any row whose capped-logit gap between the 1st and 2nd or the 2nd
and 3rd expert is below 1e-3. Without it, float32 accumulation order would flip the id of a near-tied row
against the float64 reference at large token counts, which the exact id comparison cannot tolerate; the
production router is indifferent to such ties.

`topk_ids` is compared exactly, as the contract requires for an integer output; with the margin above, any
id difference is a routing error.

`topk_weights` is compared by signal-to-noise ratio against the float64 reference, `min_db: 90`, instead of
the default baseline-relative bound. The default bound gives each element `k` times the baseline's own error
on that element or the RMS of its row, and a row here holds only two weights: wherever the baseline happens
to be very accurate (its CUDA-core kernel sums with a tree reduction), the allowance collapses and any
other correct float32 implementation exceeds it. The MI355X QA calibration probe (the reference evaluated in
float32 PyTorch) failed the default bound in 9 of 12 cases, on up to 8.4% of the weights (`t77_nocap`) and
1.8% at 262144 tokens, at up to 17.5 times the allowance. A CPU emulation of float32 summation orders
(the baseline's tree-reduced and MFMA-tiled kernels; sequential, split-K and 16-product MFMA-style
accumulation; exp-based and library tanh) shows that no admissible `k` (at most 8) with
`max_violation_fraction` at most 0.01 accepts them all: at `k = 8` a kernel accumulating sequentially in
float32 still exceeds the allowance on 27% of the `t77_nocap` weights and 1.5% of the 32768-token weights,
and the baseline's own tensor-core kernel run on `t77_nocap` exceeds it on 16%.

SQNR of `topk_weights` in that emulation, every case from 1 to 32768 tokens:

| Implementation | SQNR |
| --- | --- |
| Baseline (CUDA-core and tensor-core kernels) | 124 to 142 dB |
| Correct float32 implementations, round-to-nearest accumulation | 121 to 147 dB; worst single token of 32768: 107 dB |
| float32 accumulation rounded toward zero (a pessimistic MFMA model) | 104 to 109 dB; worst single token of 32768: 96 dB |
| Soft-cap constant 32 instead of 30 | 78 to 82 dB (99 dB for the single token of `t1`) |
| Logits or weights rounded to float16 | 71 to 77 dB |
| Soft-cap omitted | 60 to 64 dB (80 dB for `t1`; `t77_nocap` has no cap) |
| Logits or weights rounded to bfloat16 | 55 to 62 dB |
| Weights rounded to fp8, or renormalized over the top 2 | 30 to 37 dB; about 0 dB |

The 90 dB bound lies 14 dB below the most pessimistic correct float32 implementation (6 dB below its worst
single token, which matters for the one-token case) and 7 dB above the closest wrong variant: any float32
evaluation passes, while a float16 or lower-precision stage, an omitted soft-cap or a wrong cap constant fails
in every soft-capped case with more than one token. Unwritten (NaN-poisoned) elements fail it as non-finite
mismatches.

## Risks

The smallest timed case (2560 tokens, about 31 MB of activations) passed the 10 us floor in the MI355X QA
audit but is the closest to it.
