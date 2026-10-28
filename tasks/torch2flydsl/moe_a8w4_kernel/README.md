# A8W4 two-stage MoE on MI355X

Optimize the supplied FlyDSL implementation of routed gate/up GEMM, SiLU(gate)
times up, intermediate requantization, down GEMM and weighted top-k reduction.
Activations use MXFP8 E4M3 and weights MXFP4 E2M1, both with per-32 E8M0 scales.
Weight quantization explicitly preserves the original tie-up scale rounding;
the current AITER default differs at exact scale boundaries.
The two original geometries, seed 20260401, ten warmups, 100 samples and 1%
normalized maximum-error gate are retained in the protected harness.
A separate unscored constant-input case checks a finite BF16 result above the
FP16 range. The oracle rounds its final result directly to BF16; the unfinished
reference's final FP16 conversion could incorrectly produce infinity.

The operator takes BF16 hidden states and raw BF16 expert weights, FP32 routing
weights and INT32 expert IDs. Router calculation stays outside both roles.
Weight/activation quantization, packing, sorting, both FlyDSL GEMMs and top-k
combination stay inside each invocation. The output is finite BF16 with the
hidden tensor's shape and device for the declared workloads. Inputs are
read-only; output must not alias them.

Only kernel.py is editable. The task-local compatibility helpers and all
references, cases and scripts are protected. The GEMMs and activation must use
the bundled FlyDSL device implementation. AITER utilities are allowed for
quantization, packing and routing preparation already used by the initial
implementation; delegating MoE/GEMM/activation to another implementation is not
allowed. No model, reference, harness or baseline imports are allowed from the
candidate. No external downloads or source repositories are needed.

The supplied implemented candidate is frozen by Arena as the independent
performance baseline. Correctness uses the task-local pure-PyTorch mathematical
oracle on CPU, including FP4/FP8 quantization and intermediate rounding. The
oracle processes one selected expert at a time to bound temporary memory; it
does not import or call the FlyDSL implementation. CPU oracle execution is
outside device timing and does not qualify GPU BLAS compatibility.

This replaces the unfinished legacy task's PyTorch speedup comparison. Its
reference included router work that its candidate timing excluded. Historical
timings are not comparable with this baseline or cache policy.

Both correctness and performance execute the same deterministic reduce path.
Timing uses the materialized canonical ReadOnlyCacheFlush and
benchmark_cuda_graph_cold_cache_samples: a read-only 1 GiB eviction precedes
exactly one complete graph invocation, outside same-stream GPU Event timing.
Capture errors fail. Every measured sample's output is checked. The actual graph
is then replayed with poisoned output and independently changed hidden states,
routing and raw weights; each replay must satisfy the same numerical gate. All input checks,
oracle work and restoration stay outside timing. Raw samples and cache metadata
are emitted with the performance result. This is an initial implementation
baseline, not a measured optimization claim.

From an Arena-materialized workspace in the pinned gfx950 runtime:

```bash
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Each action emits arena-eval-v1. Arena owns final validation reports and scores.
Fresh full GPU task-validator qualification applies to the exact source and
runtime; CPU controls or successful imports do not substitute for it.
