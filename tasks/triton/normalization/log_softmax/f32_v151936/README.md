# Log-softmax, float32, vocabulary 151936

Row-wise `out = x - max(x) - log(sum(exp(x - max(x))))` over the vocabulary axis, the log-probabilities a
sampler or prompt-logprob path computes from the logits. The reference is `torch.log_softmax` in float64.
The logits are drawn from a normal distribution with standard deviation 4, a plausible logit spread.

Baseline: vLLM's batch-invariant Triton log-softmax kernel (`vllm/model_executor/layers/batch_invariant.py`),
one program per row with a three-pass loop (max, sum of exponentials, output) in blocks of 1024. It is wrapped
in `run()` and otherwise unchanged. The old task used float16 and toy shapes (up to 512 columns); this
scenario uses the float32 logits of a 151936-entry vocabulary (Qwen-style), where one row is 608 KB.

Workloads: a decode or sampling batch is 1 to a few dozen rows. Fewer than 32 rows are correctness-only: their
traffic is small and the time is dominated by launch overhead. From 32 rows up, one program per row runs three serial
passes over 608 KB and takes well above the 10 us floor (estimated). The performance cases (32 to 3200 rows, 100x) model
large sampling batches and prompt-logprob requests over a chunk of prompt positions. The largest case holds
about 1.9 GB of logits plus the float64 reference.

## Error bound

`out` uses `sqnr` with `min_db: 135`. The output is `x - lse` and almost all of its error is one per-row shift from the
float32 rounding of `lse`, whose size depends on the summation order over 151936 terms. All 151936 elements of a row
share that shift, so the elementwise error is not independent noise: the `baseline_relative` bound (k=3,
max_violation_fraction=1e-4, round 1) accepted the shift per element (peak 1.07x, violation fraction 3.5e-6 in the
round-2 MI355X audit, well inside the 16x peak limit) but its bias test counts every element as an independent sample.
A float32 evaluation of the reference, with a different summation order, was therefore rejected in six cases (bias z
0.33 against an allowance of about 5/sqrt(elements)), and so could a correct float32 candidate. The effective number of
independent samples is the number of rows, so no bias allowance of the elementwise bounds fits.

`sqnr` compares the norm of the error of the whole output with the norm of the reference, with no bias test. CPU
emulation (64 rows of the task's distribution; the reference rounded to float32 is the signal): a single fp32 sum of the
exponentials, the baseline's 1024-wide blocked fp32 sum and a sequential fp32 sum all give 147.3 dB (an exact-lse
candidate is higher). The control `shifted_16_ulp` (output scaled by 1 + 16 u) is 120.4 dB, below the bound by 14.6
dB; a 2-ulp shift is 138.1 dB. The threshold of 135 dB leaves 12 dB (4x in amplitude) over any float32 summation order
and still rejects a row-wise shift of about 3 output ulps or more. Gross errors fall far below it: a skipped
or zeroed 1% of the output is about 20 dB, a single element off by 0.01 about 128 dB (a lone element off by 1e-3 or less is not detected); the zeros, stale and half-written
controls are orders of magnitude outside it.
