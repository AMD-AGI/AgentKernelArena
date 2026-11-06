# Biased grouped top-k routing, DeepSeek-V3

Routes each token to 8 of 256 experts. Selection scores are `sigmoid(logit) + correction_bias`;
each of the 8 groups of 32 experts is scored by the sum of its two best selection scores, the 4 best
groups are kept (the rest are masked to 0, as in `aiter.ops.topk.biased_grouped_topk_torch`), and the 8
best experts are chosen. Weights are the unbiased sigmoid scores of the chosen experts, renormalized and
multiplied by `route_scale` (2.5). The order inside a row is unspecified, so rows are compared sorted
(`element_order: any_in_full_rows`); ids compare exactly and weights by signal-to-noise ratio (see below).

Shapes follow DeepSeek-V3 routing (256 routed experts, 8 groups, top-4 groups, top-8 experts, route scale
2.5, float32 logits and bias). Token counts range from decode batches to batched prefill; decode-sized
batches and small prefill chunks run below the 10 us timing floor and are correctness-only.

## Inputs

`inputs.py` draws standard-normal logits and redraws the rows whose margin at either decision boundary
(4th versus 5th group score, 8th versus 9th expert score) is below 1e-4. The production fused op
tolerates such boundary ties; an exact id comparison cannot, because float32 rounding in a kernel
legitimately differs from the float64 reference there. Everything else about the distribution is plain
`randn`; the correction bias is `N(0, 0.1^2)`.

## Baseline

Triton rewrite of the routing semantics of ROCm/aiter's `biased_grouped_topk_hip`
(`csrc/kernels/topk_softmax_kernels_group.cu`, a HIP kernel with no Triton form upstream): one program
per token, iterative arg-max selection over a `[8, 32]` register tile. No upstream code is copied; the
semantics (masked-to-zero groups, unbiased weights, renormalization, scale) follow the aiter source and
its torch reference. The repository's MIT license is in `baseline/LICENSE`.

The routing kernel is cheap per token, so only the large prefill token counts are timed. The MI355X QA
audit found every performance case at or above the 10 us floor.

## Comparison bounds

`topk_ids` is compared exactly (an integer output; rows are sorted first). Because the inputs have no
decision-boundary ties, any id difference is a genuine routing error.

`topk_weights` uses `sqnr` with `min_db: 125` instead of the baseline-relative default. Each weight is a
sigmoid score divided by the sum of its row's 8 scores, so the float32 rounding of the division in the
sigmoid and of the row sum, division and scaling sets every weight's error, and a row of 8 weights gives the
default bound a noisy estimate of the baseline's error. In the MI355X QA audit a float32 PyTorch evaluation of
the reference, which is a correct float32 implementation, failed the default bound in 9 of 11 cases. It
exceeded the per-element allowance on 0.19% to 0.34% of the weights from 256 tokens on, at a peak of 1.44x to
2.19x, and on 1 of the 24 weights of `t3` (1.14x). The bias test passed. `max_violation_fraction` is limited to
0.01 and cannot admit one weight in 24 (`t3`) or in 8 (`t1`). A larger `k` does not help either. A CPU
emulation at `k = 8`, the largest allowed, still rejects correct float32 implementations. Against an emulated
baseline with CPU exp and IEEE division, the GPU-style implementation of the table below fails all 8 draws
of 2560 tokens (peak 2.1x). Against the GPU-style baseline emulation, one with exp up to 2 ulp and divisions
up to 2.5 ulp fails 12 of 400 draws of `t3`.

`sqnr` compares the norm of the whole output's error (against the reference rounded to float32) with the norm of
the reference and has no per-element or bias test. CPU emulation with the task's input generator and the
framework comparator: the cases' seeds (drawn on the CPU) from 64 to 131072 tokens for the first column, 32
draws of 32768 tokens (1048576 rows) for the single-row column, and 2000 draws of `t1`:

| float32 implementation | SQNR, 64 tokens and more | worst single row | worst `t1` draw |
| --- | --- | --- | --- |
| reference in float32 PyTorch (`torch.sigmoid`, sum, divide, scale) | 142.4 to 142.8 dB | 133.3 dB | 135.0 dB |
| baseline formula, CPU exp, IEEE division, tree sum | 143.2 to 143.8 dB | 136.4 dB | 138.3 dB |
| baseline formula, GPU-style exp2(x log2 e) and divisions as a 1-ulp reciprocal and a multiply | 138.3 to 139.2 dB | 130.8 dB | 131.5 dB |
| `torch.sigmoid`, sequential sum, multiply by 2.5 / sum | 142.5 to 143.0 dB | 132.7 dB | 133.8 dB |
| exp(x) / (1 + exp(x)) for negative x, GPU-style, pairwise sum | 140.5 to 140.8 dB | 134.0 dB | 135.3 dB |
| 0.5 tanh(x / 2) + 0.5, sequential sum | 142.5 to 143.1 dB | 133.3 dB | 135.0 dB |
| exp up to 2 ulp, divisions up to 2.5 ulp (the OpenCL single-precision limit), IEEE adds | 132.0 to 132.3 dB | 126.1 dB | 127.6 dB |
| every operation, adds included, off by up to 1 ulp | 134.3 to 135.0 dB | 124.5 dB | 125.6 dB |
| every operation off by up to 2 ulp | 128.8 to 129.5 dB | 118.8 dB | 121.7 dB |

ROCm Triton lowers `/` to an approximate division, and the baseline uses it twice (in the sigmoid and in the
renormalization). The GPU-style row models that, and a candidate written the same way has about the same
error. Its worst row of 10^6 is 130.8 dB, so a 130 dB threshold would leave the baseline's own check (it faces
the same bound) a margin of 0.8 dB in `t1`. It would also reject a correct candidate with exp up to 2 ulp and
divisions up to the 2.5 ulp limit on 8% of `t1` draws. The emulated baselines bracket the audit's float32-probe
statistics under the default bound (0.06% to 0.11% and 0.8% to 0.9% violations, against about 0.2% measured),
so the GPU-style row is the pessimistic end for the real baseline.

The 125 dB threshold is at least 7 dB below every case of 64 tokens and more for all implementations except the
"every operation off by up to 2 ulp" model, and 5.8 dB below the GPU-style baseline's worst row. Only the two
models that also perturb every float32 add have rows under it: 4 of 10^6 at 1 ulp, and 4.8% at 2 ulp (that model
still passes every case of 64 tokens and more). IEEE adds never carry such errors.

The threshold rejects the `shifted_16_ulp` control (baseline output scaled by 1 + 16 u) in every case and draw.
It measures 120.3 to 120.6 dB from 64 tokens on and at most 123.3 dB in any single row of 10^6, so `t1` rejects it
too. Weights computed from a bfloat16 or float16 sigmoid reach at most 77.6 dB and 96.9 dB in any single
row, and fp8-rounded weights at most 49.9 dB. Weights taken from the biased selection scores instead of the
unbiased sigmoid reach at most 51.4 dB. Zeroed (0 dB), half-written (3 to 4 dB), stale or unwritten
(NaN-poisoned, non-finite) outputs fail outright.

A uniform scaling by up to 9 ulp is not detected (6 ulp gives 128.2 to 128.9 dB from 64 tokens on, 9 ulp
125.1 to 125.2 dB, 10 ulp 124.2 to 124.3 dB). A localized error is caught when it is large against the whole
output. In the largest case (262144 tokens), one row scaled by 1.01 gives about 94 dB, one zeroed row about
54 dB and one weight off by 1e-3 relative about 122 dB. A single weight off by 3e-4 relative (about 133 dB) is
not detected. At 1024 tokens a single weight off by 1e-4 gives 119 dB.
