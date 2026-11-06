# KDA decay gate, bf16 input, 32 heads, head size 128

The per-channel decay gate of Kimi Delta Attention (Kimi-Linear), a fused elementwise operator:

```text
out[t, h, d] = -exp(A_log[h]) * softplus(g[t, h, d] + dt_bias[h, d])      (beta = 1, threshold = 20)
```

`g` is the `[T, 32, 128]` bfloat16 output of the gate projection, `A_log` `[32]` and `dt_bias`
`[32, 128]` are float32 parameters, and `out` is `[T, 32, 128]` float32 (the log-decay that the
chunked kernels later cumulate). The softplus is the identity where its argument exceeds 20.

- **Origin.** `fused_kda_gate` / `kda_gate_fwd_kernel` of flash-linear-attention as carried in vLLM
  (`vllm/third_party/flash_linear_attention/ops/kda.py`). It replaces the earlier task that ran it on
  4 x 64 tokens with 4 heads of 32 channels; the dimensions here follow Kimi-Linear.
- **Inputs.** `g ~ N(0, 1)`, `A_log ~ U(0, ln 16)` (production initializes `log(U(1, 16))`) and
  `dt_bias ~ N(0, 0.5^2)`.
- **Baseline.** The vLLM kernel, autotune kept, with the unused lower-bound sigmoid branch removed
  and `beta`, `threshold` and the bias fixed to the production values. It evaluates
  `log(1 + exp(x))` in float32, which loses relative precision for very negative `x` (the
  absolute error stays at float32 rounding level), so it is not more accurate than other float32
  formulations; the float64 reference uses `log1p`.
- **Workloads.** 2048 to 262144 tokens (128x span). Traffic is about 24 KB per token (8 KB of
  input, 16 KB of output), so the smallest case is about 50 MB and should take the baseline over
  10 us. Decode-sized batches are below the timing floor and are correctness-only, together with an
  unaligned 1000-token prefill. The largest case has about 2 GB of input and 4 GB of output; its
  float64 reference is about 8.6 GB.
- **Comparison.** `sqnr` with `min_db: 130` against the float64 reference.
  - *Why not the default.* `out` is an elementwise float32 result of a few roundings (the gate
    sum, `exp`, `log`, the per-head rate `exp(A_log)` and the product). The default bound allows
    each element twice the baseline's error, or its per-row RMS (one head, 128 channels), plus one
    output rounding. Where the softplus is large, the baseline's error is often below one
    rounding, so an equally accurate implementation that rounds in another order exceeds it. In
    the MI355X QA audit a float32 PyTorch evaluation of the reference (with `log1p`, more accurate
    than the baseline) failed the default bound in 8 of 9 cases, on 2.1e-4 to 3.4e-4 of the
    elements (9.8e-4 at 1 token, 4 of 4096), with peaks of 1.07x to 2.49x the bound.
  - *Emulation.* A CPU float32 emulation of the baseline reproduces these numbers (on average
    3.1e-4 to 3.6e-4 of the elements from 48 tokens up, peaks up to 2.0x). It models
    Triton's AMD lowering of `exp` as `exp2(x * log2e)` with a float32 product and `log` either
    correctly rounded or as `log2 * ln2`. Against it, these correct float32 designs violate the
    default bound on up to 3.4e-3 of the elements (peaks up to 2.3x): `log1p(exp(x))`, the stable
    `max(x, 0) + log1p(exp(-|x|))`, `log(1 + exp(x))` with an accurate `exp`, `exp2` with `log1p`,
    and the other `log` lowering. With `k: 6` and `max_violation_fraction: 1.0e-3` the violations
    drop to at most 5.1e-6 of the elements (1 to 4096 tokens), but the bias test still rejects
    some of them. A design that rounds the per-head rate `exp(A_log)` differently from the
    baseline repeats that rounding error over every channel and token of the head. The bias test
    then rejects it whenever the baseline's own per-head errors happen to cancel: in 0.2 % to
    0.3 % of draws from 3 to 48 tokens (6 of 2000 at 3 tokens), and still in 5 of 2000 at
    `k: 8`. No `k` or violation fraction fixes this.
  - *Choice.* Against the float32-rounded reference, the correct designs above reach 138.3 to
    146.3 dB: the two baseline models 138.3 to 144.0 dB and the `log1p` evaluation 144.2 to
    146.3 dB, measured on 1 to 4096 tokens with 200 draws at 1 token. Variants with an extra 1 or
    2 ulp of error in every `exp` and `log`, more pessimistic than the hardware (a baseline like
    that would have given the audit's float32 evaluation a violation fraction of 2e-5 to 4e-5, not
    the measured 2e-4 to 3e-4), reach 132.6 to 141.7 dB. The evaluator's 16-ulp shift control sits at 120.4
    to 121.2 dB, fp16 rounding of the output at 73 dB, bfloat16 rounding at 55 dB and fp8 at
    31 dB. 130 dB lies midway between the shift control and the least accurate correct design
    (margins of 8.8 dB and 8.3 dB). It also rejects a systematic 8-ulp scale error (126.3 to
    128.0 dB). Because correct outputs sit near 140 dB, a single typical element zeroed or wrong
    by its own magnitude drops the SQNR to about 90 dB, even in the largest case (about 1.1e9
    elements).
