# GLM-5.3-Flash FP8 block-scale GEMM

This task optimizes the historically selected package-level callable
`aiter:gemm_a8w8_blockscale_bpreshuffle`. **Correctness covers all 21 declared
operator geometries; performance scores only seven observed M64 cases.** The
other 14 combinations remain unscored robustness/generalization tests. All use
deterministic synthetic values and an independent FP32
dequantization/matmul reference over the same quantized operands. Fresh native
outputs provide a second comparison. It has no captured tensor-value oracle.

The fixed contract returns a fresh contiguous BF16 `C[M,N]` from:

- FP8 E4M3FN activation `XQ[M,K]`, contiguous and quantized per 1×128 block.
- FP8 E4M3FN weight `WQ[N,K]`, transformed with the exact gfx950
  `shuffle_weight(weight, (16,16))` byte permutation and `is_shuffled=True`.
- FP32 `x_scale[M,K/128]`, materialized by `scale.t().contiguous().t()` as in
  the original package. For M>1 this has stride `(1,M)`; the degenerate M=1
  case retains the package's `(K/128,1)` stride.
- Contiguous FP32 `w_scale[ceil(N/128),K/128]`, with 128×128 weight blocks.

| N | K | Correctness M values | Scored M |
| ---: | ---: | --- | --- |
| 2048 | 1536 | 1, 64, 8192 | 64 |
| 4096 | 1536 | 1, 64, 8192 | 64 |
| 3072 | 4096 | 1, 64, 8192 | 64 |
| 512 | 4096 | 1, 64, 8192 | 64 |
| 4096 | 2048 | 1, 64, 8192 | 64 |
| 4096 | 256 | 1, 64, 8192 | 64 |
| 2048 | 4096 | 1, 64, 8192 | 64 |

The preserved [workload](ut/provenance/workload.json) explicitly states that
only M64 launches were observed. Its M1/M8192 rows repeat each family's count
and `trace` label, but the accompanying notes classify those sizes as priors;
they are not evidence that these M buckets occurred. Only M64 retains an
observed invocation count. Each case's `scenario_evidence` in
[ut/meta.json](ut/meta.json) and [SHAPES.json](SHAPES.json) preserves the original
row ID, pointer, count and label alongside this interpretation.

Serving provenance is ISL 8192 / OSL 1024 / CONC 64 / TP 8; this is a
single-rank operator task.

Edit only [source/kernel.py](source/kernel.py). Its dispatcher body and ABI are
verbatim stock AITER `d9e5ef7ce08ee7045d583aed768cff41aa9210fe`. Explicit imports
retain native backend dependencies. The stock `torch_compile_guard` decorator
is omitted from this eager function excerpt to avoid registering duplicate
custom operators when loading the candidate. CK device bodies remain compiled
vendor dependencies; optimizations can change dispatch/tiling or implement a
replacement behind the same ABI.

The protected binder updates the AITER package-level alias, the implementation
module alias, and an already-imported SGLang `fp8_utils` alias. It never wraps
the native CK `kernel_entry`. Candidate calls use the package alias, proving
that the replacement reaches the selected seam.

Both workers use [ut/dispatch.csv](ut/dispatch.csv), the stock dispatch table
from the same source revision. The table has no GLM family entries, so the
stock dispatcher selects its native CK default for these cases. The historical
package does **not** contain its model-specific tuning table; this stock baseline
is reproducible, but equivalence to the archived physical CK instantiation still
requires a fresh native device trace. It is not a recovered historical tuning
table or a frozen performance denominator.

Correctness covers every case with the upstream 0.05 mixed tolerance and RMS
absolute floor, three additional random draws, native parity, unchanged inputs,
independent output storage, and changed-input graph replay. The shared Arena
runner measures only the seven observed cases with canonical graph timing and
validates the actual timed graph. The other 14 cases cannot enter timing.
Recorded counts/weights are provenance only; the shared Arena aggregation
remains unchanged. The archived serving regime was eager; this operator
benchmark does not measure Python launch overhead or serving end-to-end gains.

Use the gfx950/ROCm 7.2/v0.5.17 Docker runtime in `config.yaml` and the task
validator. Native source and the protected dispatch table are hash-checked
before execution. No external tensor fixture or checkpoint is needed.

[ut/provenance](ut/provenance) retains public workload and historical selection
evidence: the archived eight-rank audit records 200 matched kernel calls and
`deepest_verified=true`. This is historical provenance, not fresh validation.
Fresh native device identity, negative binding control, correctness, performance,
and a framework-finalized `validation_report.yaml` with `overall_status: PASS`
remain release requirements. No GPU validation result is included here.

The shared worker preloads and attests the correctness entrypoint, case helpers,
and benchmark module before candidate binding. Correctness and timing reuse
those exact helper objects through their declared aliases. Native imports remain
deferred until the protected runtime preflight/overlay sequence completes.
