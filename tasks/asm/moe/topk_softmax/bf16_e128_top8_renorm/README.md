# Top-k softmax routing, 128 experts, top-8, renormalized, bfloat16 logits

Routes each token to 8 of 128 experts, the MoE router of Qwen3-235B-A22B (`num_experts: 128`,
`num_experts_per_tok: 8`, `norm_topk_prob: true`). The softmax of the token's bfloat16 router logits is
taken over all 128 experts in float32 or better; the 8 experts with the largest probabilities are chosen,
and their probabilities are renormalized to sum to 1 (the full softmax denominator cancels, so the weights
equal `exp(l_i - l_max) / sum over the chosen j of exp(l_j - l_max)`). Outputs are `topk_weights` (float32)
and `topk_ids` (int32), `[n_tokens, 8]`.

**Ties.** Equal logits give equal probabilities. When experts tie at the selection boundary (the 8th and
9th largest logits are equal), the lower expert id is chosen. This is the production kernel's rule
(verified on MI355X: AITER's kernel returns ids `0..7` for an all-equal row and `5..12` for a row whose 15
experts `5..19` share the maximum), and the reference implements it with a stable descending sort. With the
task's input distribution about 7% of the rows have such a boundary tie, so the rule is exercised in every
case from 32 tokens on. The order of the 8 entries within a row is unspecified (`element_order:
any_in_full_rows`: rows are compared sorted); the production kernel writes them in descending weight order.

## Production source

SGLang (`python/sglang/srt/layers/moe/topk.py`, `fused_topk` with `scoring_func="softmax"`) calls
`aiter.fused_moe.fused_topk` when `SGLANG_USE_AITER=1`. For gfx950, `(experts, topk) = (128, 8)` and a
contiguous bfloat16 or float32 gating tensor, AITER dispatches to `topk_softmax_asm`
(`csrc/py_itfs_cu/asm_topksoftmax.cu`), which launches one of the `hsa/gfx950/topksoftmax/*.co` assembly
kernels. A profile of Qwen3-235B-A22B-Instruct-2507-FP8 serving with SGLang 0.5.20 / AITER 4ad99832 on
MI355X (TP4) attributes 2.1% of decode GPU time to `topksoftmax_4x128x8_bf16` (batches of 4 and 32 tokens)
and shows `topksoftmax_12x128x8_bf16` on the 16384-token (and 15331-token) prefill chunks and
`topksoftmax_4x128x8_bf16` on the 8046 to 8192-token chunks.

## Baseline

`baseline/kernel.py` is a Python translation of the `topk_softmax_asm` launcher over the vendored code
objects, byte for byte from AITER 4ad99832 (MIT, `baseline/LICENSE`):

| n_tokens | kernel (code object) | tokens per workgroup |
| --- | --- | --- |
| < 10000 | `topksoftmax_4x128x8_bf16.co` (`_ZN5aiter24topksoftmax_4x128x8_bf16E`) | 4 |
| >= 10000 | `topksoftmax_12x128x8_bf16.co` (`_ZN5aiter25topksoftmax_12x128x8_bf16E`) | 12 |

This is `get_heuristic_kernel_topksoftmax` over the bf16 / 128 / 8 rows of `topksoftmax.csv` (the largest
`subm` not above `MAX_SUBM = n_tokens < 10000 ? 4 : 12`). The grid is `ceil(n_tokens / subm)` workgroups of
256 threads; the 128-byte packed `KernelArgs` (ids, weights and logits pointers each followed by a `p2`, then
`batch`, `expert`, `topk`, `renormalize = 1` and `out_stride` in bytes, each followed by a `p3`) is
reproduced field by field. The kernel guards the token count: it writes no row at or beyond `n_tokens`, so
the outputs need no padding (SGLang also passes unpadded `[M, topk]` buffers).

Fidelity (MI355X, this image's `import aiter`, `aiter.fused_moe.fused_topk(h, logits, 8, True,
topk_ids=..., topk_weights=...)` as SGLang calls it, identical logits): weights and ids are bitwise equal for
1, 3, 4, 32, 1000, 8046, 8192, 9999, 10000, 15331, 16384, 65536, 262144 and 1048576 tokens; the profiler
shows the same kernel as the table above in each case; warm-launch times agree within 0.05 us.

## Workloads

Routing reads 256 bytes and writes 64 bytes per token, so even a 16384-token prefill chunk runs in about
14 us. Cold-cache CUDA-graph baseline times on MI355X (the framework's performance action):

| case | t8192 | t16384 | t65536 | t262144 | t1048576 |
| --- | --- | --- | --- | --- | --- |
| baseline | 11.1 us | 13.9 us | 27.2 us | 77.4 us | 284.5 us |

The timed cases are the two profiled prefill chunk sizes (8192 on the 4-token kernel, 16384 on the
12-token kernel) and larger batched-prefill token counts. Production serving with `chunked_prefill_size
16384` never routes more than 16384 tokens per call; 65536 to 1048576 tokens are kept because the task
contract needs a 100x span of timed sizes and every timed case must take the baseline at least 10 us
(about 9.3 us is this kernel's floor, reached below 4096 tokens), and they are where the kernel is
bandwidth-bound. Decode batches (1, 3, 4, 32), an unaligned prefill (1000), the profiled 8046 and 15331
(a partial 12-token workgroup) chunks and both sides of the kernel switch (9999, 10000) are
correctness-only.

Logits are drawn from `N(0, 1.5^2)` and rounded to bfloat16.

## Comparison bounds

`topk_ids` is compared exactly (integer output, rows sorted first). Distinct bfloat16 logits differ by at
least one bfloat16 ulp (0.4% to 0.8% relative), far beyond float32 rounding of `exp`, so a correct
float32 implementation selects the same set; equal logits are resolved by the lower-id rule above.

`topk_weights` uses `sqnr` with `min_db: 124` instead of the baseline-relative default. Each weight is a
probability divided by the sum of its row's 8 probabilities, so the rounding of that one sum moves all 8
weights of a row together, and the bfloat16 logits give rows few distinct values: the per-element errors of
two correct float32 designs are neither elementwise comparable nor unbiased relative to each other. Measured
on MI355X against the AITER baseline under the default bound (6 draws of each of 1 to 65536 tokens, 2 of
262144 and 1048576), a float32 PyTorch evaluation of the reference failed 31 of 52 draws, an
`exp(l - l_max) / sum` design 28 and an `exp2`, sequential-sum, reciprocal design 32 (peak 1.1 to 2.1 bounds
on 0.01% to 3% of the weights, and from 8192 tokens on the bias test, which the baseline's own mean error,
about -0.05 of its RMS error, already exceeds from 1000 tokens on).

SQNR of the whole output against the reference rounded to float32, on MI355X (40 draws each of 1, 4, 32
and 1000 tokens, 4 draws each of 8192 to 1048576 tokens; "worst row" is the lowest single-row SQNR in
1048576 tokens):

| implementation | 1 token | 4 to 1000 tokens | 8192 tokens and more | worst row |
| --- | --- | --- | --- | --- |
| AITER baseline | 132.5 dB or more | 136.7 dB or more | 139.2 dB or more | 127.7 dB |
| float32 PyTorch softmax, sort, renormalize | 139.5 dB or more | 141.7 dB or more | 144.4 dB or more | 133.0 dB |
| `exp(l - l_max)` over the top 8, divide by the sum | 139.8 dB or more | 142.1 dB or more | 144.8 dB or more | 133.5 dB |
| `exp2`, sequential sum, multiply by the reciprocal | 137.5 dB or more | 137.0 dB or more | 142.4 dB or more | 130.1 dB |
| the same with `exp2` and the reciprocal off by up to 1 ulp | 133.6 dB or more | 133.3 dB or more | 137.4 dB or more | 126.6 dB |
| the same with up to 2 ulp | 127.8 dB or more | 129.3 dB or more | 133.6 dB or more | 123.9 dB |
| output scaled by 1 + 6 u | 128.0 to 130.0 dB | 128.3 to 129.6 dB | 128.8 to 128.9 dB | 126.7 dB |
| output scaled by 1 + 16 u (the `shifted_16_ulp` control) | 120.1 to 120.9 dB | 120.2 to 120.7 dB | 120.4 dB | 119.5 dB |

The 124 dB threshold is at least 3.8 dB below every observed 1-token draw of every design up to 2-ulp
`exp2` and reciprocal errors, and 8.5 dB below the baseline's worst; it rejects the `shifted_16_ulp` control
by at least 3 dB in every case. A uniform 6-ulp scaling passes. Wrong selections fail on `topk_ids`, and
zeroed, half-written, stale or unwritten (NaN) weights fail outright.
