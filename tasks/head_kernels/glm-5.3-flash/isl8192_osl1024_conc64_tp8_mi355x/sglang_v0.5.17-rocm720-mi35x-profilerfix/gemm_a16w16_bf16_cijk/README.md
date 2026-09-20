# GLM-5.3-Flash BF16 GEMM

This task optimizes the historically selected `aiter.tuned_gemm:torch_gemm`
launcher behind the GLM BF16 Cijk head. **Correctness covers all 27 declared
operator geometries; performance scores only seven observed M64 cases.** The
other 20 combinations remain unscored robustness/generalization tests, with
deterministic synthetic tensor values. Its references
are FP32 matrix products over those same BF16 operands, plus fresh results from
the version-checked native callable. It has no captured tensor-value oracle.

The fixed contract is `C = A @ B.T`, with contiguous BF16 `A[M,K]`, `B[N,K]`, and
a fresh contiguous BF16 `C[M,N]`. Bias and all scales are `None`, preshuffle is
false, and the selected inner launcher receives `solidx=0`. The full production
signature remains fixed, including optional arguments.

| N | K | Correctness M values | Scored M |
| ---: | ---: | --- | --- |
| 4096 | 1536 | 1, 64, 8192 | 64 |
| 3072 | 4096 | 1, 64, 8192 | 64 |
| 4096 | 1024 | 1, 64, 8192 | 64 |
| 288 | 4096 | 1, 64, 8192 | 64 |
| 128 | 4096 | 1, 64, 8192 | 64 |
| 32 | 4096 | 1, 64, 8192 | None; unprofiled family |
| 8 | 4096 | 1, 64, 8192 | 64 |
| 1024 | 128 | 1, 64, 8192 | 64 |
| 4096 | 512 | 1, 64, 8192 | None; unprofiled family |

The preserved [workload](ut/provenance/workload.json) states that only M64
launches were observed. M1 and M8192 are size/floor priors; the `(32,4096)` and
`(4096,512)` families have no profiled cases at any M. BF16 per-case invocation
counts are not recorded, so they remain null. Original source IDs, row pointers,
and count/weight labels are retained in each case's `scenario_evidence` in
[ut/meta.json](ut/meta.json) and [SHAPES.json](SHAPES.json).

Serving provenance is ISL 8192 / OSL 1024 / CONC 64 / TP 8; the task runs a
single-rank operator. Observed shape support does not establish identical
per-case launch, solution, grid, or current runtime dispatch.

Edit only [source/kernel.py](source/kernel.py). Its function body is a verbatim
stock excerpt from AITER `d9e5ef7ce08ee7045d583aed768cff41aa9210fe`, with explicit
imports. The protected binder updates `torch_gemm` **and** `solMap` entries that
point to the native function. Candidate calls go through `solMap['torch']`, so
editing an unused module attribute cannot pass as an effective optimization.

Correctness checks every case against FP32 and native outputs with the upstream
0.02 mixed tolerance and RMS absolute floor, three additional random draws,
unchanged inputs, and independent output storage. Additional changed-input graph
replay checks qualify the isolated device-time benchmark. The shared Arena runner
measures the seven observed cases with its canonical graph timing, fixed
warmups and sample count, and validates the actual timed graph. The other
20 cases cannot enter timing. Recorded profile weights are provenance only;
the shared Arena aggregation remains unchanged. Historical serving used eager
execution; these device timings do not measure Python launch overhead or claim
serving end-to-end performance.

The runtime requires gfx950, ROCm 7.2, and the v0.5.17 image in `config.yaml`.
The native AITER source file must match the protected SHA-256 before execution.
Use the repository's Docker cohort launcher and task validator. No external
tensor fixture or model checkpoint is needed.

[ut/provenance](ut/provenance) retains the public package's workload and historical
selection evidence. Its selected seam passed the archived eight-rank selection
audit with 128 matched device-kernel calls. That record establishes historical
seam provenance; it is not a fresh GPU result for this task. Fresh native device
identity, negative binding control, correctness, performance, and a framework
finalized `validation_report.yaml` with `overall_status: PASS` remain release
requirements. No GPU validation result is included here.

The shared worker preloads and attests the correctness entrypoint, case helpers,
and benchmark module before candidate binding. Correctness and timing reuse
those exact helper objects through their declared aliases. Native imports remain
deferred until the protected runtime preflight/overlay sequence completes.
