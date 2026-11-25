# bfloat16 GEMM (NT), N=5120, K=2880

`out = a @ b.T` with `a[m, 2880]`, the row-major weight `b[5120, 2880]` and `out[m, 5120]`, all bfloat16, accumulated in
float32 or better and rounded once. No bias. The var axis is `m`. Both inputs are standard normal. The shape is the
attention QKV projection of a GPT-OSS-style model (hidden size 2880, 64 query plus 8 key and 8 value heads of width 64
gives 5120 outputs). It replaces the `(M, 5120, 2880)` cases of the earlier `hgemm_splitk_kernel` (FlyDSL, f16 and bf16)
and `aiter/gemm_a16w16` (Triton) suites; the float16 cases are dropped.

- **Baseline:** AITER's FlyDSL split-K HGEMM kernel (`aiter/ops/flydsl/kernels/splitk_hgemm.py`, ROCm/aiter at the
  recorded revision, MIT; the same kernel family as the earlier `hgemm_splitk_kernel` task) launched through the per-`m` table
  `CONFIGS` in `baseline/kernel.py`. The table is the set of FlyDSL rows of AITER's GPT-OSS tuning file for N=5120, K=2880
  on gfx950 (`model_configs/gptoss_bf16_tuned_gemm.csv`): split-K 3 for `m <= 48`, single-pass tiles from 16x64 up to
  256x128 above. AITER pads `m` up to the next tuned row, as the bucket bounds do. The tuned `m = 16384` row is a Triton
  kernel and cannot be vendored here, so the `m = 8192` configuration serves every `m` above 2048. The tuned rows are for the
  production QKV projection **with bias**, while this task has no bias; the tile choices are used as they are, with the
  kernel's bias path off. A choice that is tuned for the bias variant may be slightly off for the bias-free one.
- **Baseline changes:** the kernel was a builder returning a launcher; `run()` selects the table row for `m`, compiles each configuration once (`functools.lru_cache`) and launches on the current stream. Absolute `aiter` imports became relative; environment switches, `numpy` and star imports were removed from the helpers so the sources pass the FlyDSL candidate rules. `buffer_ops.py` and `vector.py` are AITER's copies of ROCm/FlyDSL code (Apache-2.0, headers kept); `baseline/LICENSE` is AITER's MIT license. `splitk_hgemm.py` differs from AITER only in its import line.
- **Reference:** `reference.py` multiplies in float64 (rows of `a` in chunks) and the result is judged against the bfloat16 output, with the `baseline_relative` bound at `k` = 8 (see below). The earlier task gated every element at bf16 1e-2 absolute or relative against an FP32 reference, which AITER's own split-K dispatches could not meet; here each element is judged against the baseline's own error. Split-K configurations reduce inside the kernel (zero-fill, then atomic bfloat16 adds guarded by a counter pair the kernel resets), so their rounding error is larger than a single-pass kernel's; that is part of the baseline error budget.
- **Workloads:** weighted by a recorded serving workload of GPT-OSS-120B; see Workloads below.
- **Not verified here:** the kernel needs FlyDSL and gfx950. The table was checked on the CPU against the kernel's tiling,
  LDS and split-K counter constraints only. Risks for the GPU audit: the decode cases (`m` = 24, 32) are near the 10 us floor, bias-tuned
  tiles run without bias, and the spin-waiting split-K counters under CUDA-graph replay.

## Workloads

The performance cases come from a recorded agentic serving workload: the AgentX corpus
(`semianalysisai/cc-traces-weka-062126`, agentic coding sessions) replayed with 32 concurrent sessions against
`openai/gpt-oss-120b` on SGLang 0.5.23 and MI355X (capture `gpt-oss-120b-tp1-mi355x-v1`): TP 1, no DP
attention, no speculative decoding, chunked prefill 16384, bfloat16 KV cache, and the model's 131072-token
context (requests that do not fit are skipped: 53% of the corpus requests, 87% of its input tokens). Each
case's `weight` is how often its shapes ran (calls per second of the steady state, TP rank 0), and the task
scores the serving-weighted speedup `sum(w*t_baseline)/sum(w*t_candidate)`, the time the workload spends in
the operator before and after (see Case weights in `docs/how-to/add-task.md`). `weight_provenance` in
`workloads.json` records the capture, the effective serving settings and the share of the corpus the context
window dropped. The 6 weighted cases cover 98.9% of the operator's recorded GPU time:

| `m` | Phases | Share of the operator's time | Weight (calls/s) |
| ---: | --- | ---: | ---: |
| 24 | decode | 0.9% | 19.02 |
| 32 | decode | 19.5% | 384.7 |
| 6720 | prefill | 0.7% | 1.024 |
| 9024 | prefill | 1.3% | 1.447 |
| 10560 | prefill | 2.1% | 1.8 |
| 16384 | prefill | 74.5% | 51.07 |

Decode runs `m` = the running batch padded to its CUDA-graph batch size (24 and 32 at this load); prefill runs
`m` = the new tokens of a batch, up to the 16384-token chunk, which alone takes three quarters of the time. In
the MI355X QA audit the baseline took 15-17 us for the decode cases and 0.21-0.45 ms for the prefill cases.

The earlier performance cases (`m` = 128 to 65536, except 16384, which is now weighted) are correctness-only,
together with the earlier correctness cases (`m` = 1 to 64 and the unaligned 33, 100, 200, 300 and 1000), `m`
= 1001 and 4099 (not multiples of 4, on single-pass tiles) and the other recorded shapes (77 cases in total).
The largest output (`m` = 65536) is 640 MiB. K = 2880 is not a power of two and not a multiple of 128, so
tile_k 64 with split-K 3 (960 per split) is the largest K tile the table uses.

## FlyDSL 0.3.2 adaptation

FlyDSL 0.3.2 adaptation: in buffer_ops.buffer_load and buffer_store the cache-policy operand is passed as the keyword-only aux integer attribute of rocdl.RawPtrBufferLoadOp / RawPtrBufferStoreOp (the runtime's op constructors take no positional aux value); no change to the kernel algorithm.

## Split-K nondeterminism and the bound (k = 8)

`compare.out` is `baseline_relative` with `k` = 8 and no violation fraction. The baseline dispatches split-K for part of
the M range and combines the partial sums into the bfloat16 output with atomic adds, so its result depends on the
order in which the partial blocks arrive: an exact copy of the baseline differs from the baseline's own run on a few
elements. At the default `k` = 2 that copy fails; the violating elements are dominated by the baseline-error term of
the bound, so their multiple scales as `peak(k=2) * 2 / k`, and `k` = 4 still fails any element measured above 2x.
`k` = 8, the contract maximum, brings the measured peaks of all split-K gemm_a16w16 scenarios (1.65x to 2.36x at
`k` = 2) to about 0.4x to 0.6x. One uniform rule is used for every split-K scenario of this operator. A 16-ulp shift
of a bfloat16 output is about 6% relative error and fp8 rounding is larger, so both controls stay far outside the
`k` = 8 bound of a baseline whose error is one bfloat16 rounding plus the split-K roundings.

Measured for this scenario (MI355X QA audit r2a, initial candidate = exact copy of the baseline): 8 case(s) failed the default bound, up to 0.12% of the elements over it, peak 1.65x at `k` = 2, about 0.41x at `k` = 8.
