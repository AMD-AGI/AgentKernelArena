# bfloat16 GEMM (NT), N=4096, K=4096

`out = a @ b.T` with `a[m, 4096]`, the row-major weight `b[4096, 4096]` and `out[m, 4096]`, all bfloat16, accumulated in
float32 and rounded once. No bias. The var axis is `m`. Both inputs are standard normal. It replaces the earlier
`mi300x_sglang_triton_gemm` task, whose performance cases were `4096x4096x4096` (this task at `m = 4096`) and
`8192x8192x1024` (a different N and K, dropped); that task was an MI300X task, this one targets gfx950.

- **Baseline:** AITER's Triton A16W16 GEMM (`_gemm_a16_w16_kernel` in
  `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py`, ROCm/aiter at the recorded revision, MIT), launched with
  the `GEMM-A16W16` default configuration of AITER for gfx950 (`configs/gemm/gfx950-GEMM-A16W16.json`; there is no
  N=4096, K=4096 specific file for gfx950), selected by `m` with AITER's `M_LEQ_x` rule and kept as a table in
  `baseline/kernel.py`. The kernel body is unchanged apart from dropping the kernel-repr helper; `remap_xcd` and
  `pid_grid` of AITER's `pid_preprocessing` are inlined. All rows of that table use a single K split, so the split-K reduce kernel
  of the wrapper and the unused bias/activation arguments are omitted (the kernel keeps its `ADD_BIAS`, `NUM_KSPLIT` and
  activation parameters, set to their off values). `run()` writes straight into `out` and launches one kernel.
- **Reference:** `reference.py` multiplies in float64 (rows of `a` in chunks) and the result is judged against the bfloat16 output, with the default `baseline_relative` bound. The earlier task gated every element at bf16 1e-2 absolute or relative against an FP32 reference; here each element is judged against the baseline's own error. The baseline is a single-pass kernel (one K split, float32 accumulation in the MFMA tiles, one rounding at the store), so its error is the bfloat16 output rounding plus float32 accumulation error.
- **Workloads:** `m` = 1 to 100 and 200, 1000 (decode, small batch, unaligned sizes) are correctness-only. Performance
  cases are `m = 128 .. 65536` (512x span): at `m = 128` the 32 MiB weight read alone is about 7 us, so the case is expected
  above 10 us, but it is the one nearest the floor. The largest output is 512 MiB. Boundary correctness cases:
  `m = 1`, `m = 3` and `m = 33`.
- **Not verified here:** no GPU was available; only the CPU checks ran. The configuration table is AITER's own and was
  not retuned for gfx950 by this task.
