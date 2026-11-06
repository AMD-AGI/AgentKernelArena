# FP8 MQA logits, DeepSeek-V3.2 indexer (64 heads, head size 128)

For each query token the lightning indexer scores every key in its window:
`logits[i, j] = sum_h weights[i, h] * relu(q[i, h, :] . kv[j, :] * kv_scales[j])`, with `-inf` outside
`[cu_seqlen_ks[i], cu_seqlen_ke[i])`. Q and KV are float8_e4m3fn (MI355X native format), the dot is
accumulated in float32, and the float32 logits feed a later top-k.

**Production origin.** AITER's `fp8_mqa_logits`, used by vLLM and SGLang for the DeepSeek-V3.2 sparse
attention indexer (64 indexer heads, head size 128).

**Scenario and workloads.** Prefill chunks: `seq_len` query tokens against `seq_len_kv` packed keys.
The window generator (`inputs.py`) packs two requests (one when a side is too small to split) and gives
each row its chunked-prefill causal window, so the ragged `cu_seqlen_ks/ke` offsets and the start > 0 path
are exercised. Decode and tiny shapes finish below the 10 us floor and are correctness-only. The largest
case materialises a 1 GiB float32 logits tensor.

**Baseline.** AITER's Triton kernel, vendored. One deliberate modification: production pre-fills the
logits with `torch.full(-inf)` before the launch; the evaluator hands over an uninitialised (NaN-poisoned)
buffer and candidates may not run PyTorch compute, so the baseline kernel writes `-inf` into the
out-of-window columns itself. This adds one pass of stores over the masked region that production performs
as a separate fill, so the work is equivalent. The Gluon path and arch detection were removed.

**Comparison.** `baseline_relative` with the default `k = 2` and `max_violation_fraction: 1e-4`; the
out-of-window `-inf` positions must match exactly. The baseline's fp8 MFMA accumulation is the dominant
error. On MI355X a float32 PyTorch evaluation of the reference (fp8 products are exact in float32, so only
the scale, weight and head-sum roundings remain) stays within 0.04-0.10x of the bound in every case whose
rows have many in-window columns, which puts the baseline's error near 50 u|y| rms (u = 2^-24), far
above an exactly accumulating candidate's (under 1 u|y| rms). The bound degenerates only where a row has
a single in-window column: the per-row RMS of the baseline error is then that one element's error, and
when the baseline happens to land close to the exact value the allowance `2 |baseline error| + u|y|`
shrinks to about u|y|. `s2048_kv1024` (more queries than keys) has 1,026 such rows among its 263,680
in-window elements, and the MI355X audit's float32 evaluation exceeded the bound in 2 elements of that
case (violation fraction 7.6e-6, peak 1.56x, bias z 0.001) while passing every other case. A CPU error
model of the baseline (products truncated about 15 bits below the largest product of each MFMA step)
reproduces the audited peaks on `s1_kv4096`, `s7_kv2000` and `s100_kv777` (simulated 0.04-0.05x, audited
0.04-0.06x). Over 11 draws of
`s2048_kv1024` it gives 0-2 violations for exactly accumulating float32 candidates and 4-13 for a copy of
the baseline that sums the heads in another order, all in single-column rows; no row with two or more
columns violated. `1e-4` allows 26 violations in `s2048_kv1024` (twice the worst draw) and none in
`s1_kv1`, `s1_kv4096` and `s7_kv2000`, where the audit first rejected every control, so the controls fail
exactly as before; `k` and the bias test are unchanged. `s1_kv1` has a single element and cannot use the
allowance: by the same mechanism a correct candidate that rounds differently from the baseline can exceed
the bound there, at a simulated rate of about 0.1% per draw for an exactly accumulating candidate and 0.7%
for a reordered copy of the baseline.
