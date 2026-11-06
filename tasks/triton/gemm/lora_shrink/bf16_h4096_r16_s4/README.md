# LoRA shrink (punica SGMV), bfloat16, hidden size 4096, rank 16, four adapters

Projects each token from the hidden size to the LoRA rank with the A matrix of the adapter it is routed to:
`out[t] = scaling * x[t] @ lora_a[a].T`, accumulated in float32 and returned in float32 (the dtype vLLM uses for the
intermediate that feeds the expand kernel). The kernel reads the activations once (8 KB per token), so it is bound by
activation bandwidth.

**Production origin.** vLLM's `lora_shrink` (`vllm/lora/ops/triton_ops/lora_shrink_op.py`), called for every LoRA-enabled
linear layer of a served model. Hidden size 4096 and rank 16 are Llama-3-8B-class values; four adapters are active.

**Routing metadata.** `inputs.py` produces vLLM's `LoRAKernelMeta` tensors: token indices sorted by adapter group,
tokens per group, group start offsets and the adapter slot of each group (a random permutation of the four slots).
Group sizes are skewed; every token belongs to an active adapter (no `-1` ids, so every output row is written).

**Baseline.** The vLLM Triton kernels, launched with vLLM's default shrink configuration (split-K 64 and block-K 256
below 128 tokens, split-K 8 and block-K 32 above). vLLM zeroes the output with `output_tensor.zero_()` before the
split-K atomic accumulation; the baseline does it with a small Triton kernel, as a candidate must. Only the
single-slice path is kept. The kernel code is the version of the previous task package, which matches upstream apart
from comments.

**Workloads.** Prefill token counts. Batches under about 4096 tokens (including decode) move less than 34 MB and run
below the 10 us timing floor, so they are correctness-only; performance cases span 4096 to 409600 tokens (100x).
Large counts are well beyond one scheduler step; they are needed to keep the 100x span at a realistic per-token cost.

**Notes.** Split-K accumulation uses float32 atomics, so the baseline is not bitwise deterministic (two runs differ by a
few float32 roundings). The error bound compares with the float64 reference only, so it does not depend on the baseline
run a candidate meets. The timing floor is tight at the smallest performance case.

**Error bound: `sqnr` with `min_db: 122` instead of the default `baseline_relative`.** Every output is a float32 sum of
4096 bfloat16 products, which are exact in float32, so a correct kernel's error is its accumulation rounding, and that
depends on the accumulation order. The baseline splits K eight ways (512 products per split, combined by atomics) and is
unusually accurate: measured on MI355X, 137.0 to 137.2 dB SQNR against the float64 reference in every case of 777 tokens
or more and 135.7 to 138.9 dB in the 1-, 3- and 48-token cases (split-K 64). The element-wise bound allows at most 8
times (`k` is capped at 8) the baseline's error (or its 16-element row RMS) and at most 1% of the elements beyond that,
and the 1- and 3-token cases (16 and 48 elements) cannot admit a single element beyond it. The previous bound (`k: 8`,
`max_violation_fraction: 4e-6`, set for the baseline's own run-to-run differences) rejected the QA audit's float32
PyTorch evaluation of the reference on MI355X in eight cases: up to 0.15% of the elements beyond the bound with a peak
of 2.73 bounds from 8192 tokens up, and one or two elements (peak 1.08) at 777, 2048 and 4096 tokens.

A CPU emulation of float32 accumulation orders on this task's input distributions, scored as the evaluator scores them,
shows that no element-wise setting accepts the common kernel designs. Products are exact; each bfloat16 MFMA is modelled
as float32 accumulation of exact 8-product sums, which reproduces the baseline's measured 137.1 dB and its measured
run-to-run agreement (a second run peaks at 0.89 to 1.03 of the old bound emulated, at up to 1.03 measured). Over 32768
emulated rows (SQNR varies by less than 0.3 dB between cases of 777 tokens or more):

| Implementation | SQNR | Elements over the old `k: 8` bound |
| --- | --- | --- |
| the baseline, and a second run with another atomic order | 137.1 dB | 0 |
| split-K 4 | 134.4 dB | 1.1e-4 |
| split-K 2 | 131.3 dB | 2.5e-3 |
| one accumulator over K, bfloat16 `tl.dot` (the plain Triton GEMM) | 128.0 dB | 1.6e-2 |
| one accumulator over K, float32 MFMA (4-product steps) | 126.1 dB | 3.3e-2 |
| one scalar float32 FMA chain over all 4096 products | 124.3 dB | 5.5e-2 |
| `shifted_16_ulp` control | 120.3 dB | 0.19 |
| `fp8_rounded` control | 31.5 dB | 1 |
| half of the output zeroed, all of it zeroed | 3 dB, 0 dB | 0.5, 1 |

The single-accumulator designs exceed the element-wise bound in 1.6% to 5.5% of the elements, above the 1% ceiling of
`max_violation_fraction`. The audit's float32 evaluation (0.12% to 0.15% from 8192 tokens up, peak 2.73) lies between
split-K 4 and split-K 2, so about 132 dB from 8192 tokens up and 134 dB or more in the smaller cases, about 10 dB above
the floor.

122 dB lies between the least accurate float32 design (124.3 dB) and the 16-ulp control, whose SQNR is fixed by
construction (`-20 log10(16 * 2^-24)` = 120.4 dB, lowered slightly by the baseline's own error): 120.3 to 120.4 dB from
777 tokens up, at most 120.5 dB in 1365 emulated 48-token draws and at most 122.1 dB in 65536 1-token draws (above 122
dB in fewer than 1 in 10,000). The control is therefore rejected in every case from 48 tokens up by about 1.5 dB or
more, and the float8 control by 90 dB. Every float32 design in the table passes every case from 48 tokens up (the scalar
chain's lowest emulated 48-token draw is 122.7 dB); on average the single-accumulator designs clear the floor by 2.3 to
6 dB and the baseline by 15 dB. SQNR over the 16 elements of the 1-token case and the 48 of the 3-token case is noisy:
in 65536 emulated 1-token draws the bfloat16 single-accumulator design fell below 122 dB in 0.4%, the float32 MFMA
design in 2.9% (0.3% of the 3-token draws) and the scalar FMA chain in 13% (6% of the 3-token draws); the baseline's
lowest 1-token draw was 128.3 dB. A scalar chain over 4096 products has about two thirds of the 16-ulp shift's error, so
no whole-output threshold separates the two on 16 elements; the 1-token case stays because it is the decode shape and
the axis minimum.

Unwritten (NaN-poisoned) or non-finite elements always fail, and dropped, stale, zeroed, half-written or float8 outputs
fail by 90 dB or more. The floor allows an error RMS of 7.9e-7 of the output RMS (about 5.7 times the baseline's): a
scale error above about 13 float32 units of roundoff fails in every case, and a single wrong element fails above about
4e-6 in the 1-token case, 3e-5 in the 48-token case and 3e-3 in the largest case (output RMS 1.28). An `sqnr` bound has
no bias test; the 16-ulp shift has no mean error either, so the old bound also rejected it element-wise only. The floor
was calibrated with the CPU emulation and the MI355X baseline measurements above; the MFMA rounding model is an
assumption fitted to the measured baseline SQNR.
