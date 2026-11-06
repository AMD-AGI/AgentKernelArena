# RMSNorm, bfloat16, hidden size 8192

Normalizes each token row by the reciprocal root mean square of its values and scales it by
`weight`. The baseline is vLLM's batch-invariant Triton RMSNorm kernel, one program per row.
Decode-sized batches (fewer than about 2048 tokens) run faster than the 10 µs timing floor and
are checked for correctness only.
