# L2 normalization, bfloat16, head size 128

`out = x / sqrt(sum(x^2) + epsilon)` along the last axis, with no mean subtraction. Gated delta net
(GDN) linear attention, as served for Qwen3-Next style models, L2-normalizes its query and key heads before
the chunked delta-rule scan, so one row is one (token, head) pair of 128 features; `epsilon` is 1e-6. The
reference computes everything in float64; the output is rounded to bfloat16 and compared with the default
baseline-relative bound.

## Baseline

SGLang's Triton `l2norm_fwd_kernel` (flash-linear-attention's `l2norm`), the path its launcher takes for
widths up to 512: 16 rows per program, one block-pointer load per row block, 8 warps. SGLang commit
`887c401e151e2d1b082d74ef8769a23eef5c465c`; the kernel is unchanged and only the launcher is reduced to
`run()`. The upstream file is adapted from flash-linear-attention (MIT, copyright Songlin Yang and Yu Zhang),
whose license text is in `baseline/LICENSE`. The earlier vLLM copy of the same kernel was folded into this task.

## Workloads

A row moves 512 bytes, so a case needs about 60k rows to exceed the 10 us timing floor: performance cases start
at 65536 rows (a 4096-token chunk with 16 heads) and span 128x up to 8.4M rows (4.3 GB moved). The reference
works in chunks of 1M rows. Smaller row counts, from a single row up, are correctness-only.

## Scope

The earlier tasks also exercised head sizes 64, 256, 512 and 1024 (the last on a separate row-wise kernel) and a
float32 input in a small test. Those are not GDN production shapes (Qwen3-Next style models use head size 128 in
bfloat16), so they are not scenarios of this task; the `T` axis boundary cases (1, 3, 48, 1000, 4100 rows) cover the
row-tail handling of the 16-row blocks.
