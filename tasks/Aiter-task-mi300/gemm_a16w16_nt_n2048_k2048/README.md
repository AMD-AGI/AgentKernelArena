# BF16 GEMM in FlyDSL, MI300X (gfx942) variant

This task is the gfx942 variant of `tasks/Aiter-task/gemm_a16w16_nt_n2048_k2048`. The workload, inputs, reference, comparator
and baseline entry are preserved. The platform, starting candidate, numerical policy and timing differ:

1. **Platform.** `config.yaml` requires `gfx942` and `scripts/evaluate.py` checks for it. The runtime image is
   `docker.io/primussafe/sglang:v0.5.18-rocm720-mi35x-aiter-dbd8bf5-flydsl032-20260921`: AITER `dbd8bf5bd624120197a7a26780a8c72201824f0f`, FlyDSL 0.3.2, ROCm PyTorch built for gfx942 and
   gfx950.
2. **Initial candidate.** The candidate is declared `implemented`. `kernel.py` is a working FlyDSL BF16 GEMM:
   AITER's split-K HGEMM kernel (`aiter/ops/flydsl/kernels/splitk_hgemm.py` at the commit above) with the helper
   modules it imports joined into one file, so that it passes the import guard. `build_gemm_a16w16_nt_n2048_k2048_module` launches it with one
   configuration for M <= 32 and one for larger M. The agent optimizes this kernel in FlyDSL instead of writing one
   from an empty module. `evaluate.py`'s initial-state check follows the state `config.yaml` declares. Under the historical warm-graph protocol on
   MI300X on 2026-10-22, its mean case speedup over the baseline below was 0.55.
3. **Baseline dispatch.** On gfx942, `aiter.tuned_gemm.gemm_a16w16` finds no tuned entry for MI300X (its tuned
   tables are for gfx950). Measured on 2026-10-22, it dispatched to torch (hipBLASLt) for every case. Speedups are against that dispatch,
   not against gfx950's production selection, so they are not comparable with gfx950 results.

4. **Numerical policy.** Baseline correctness and measured-output checks are required on gfx942, just as
   candidate checks are. Historical gfx950 numerical exceptions do not apply to this platform.
5. **Measurement.** Both roles use `cuda_graph_cold_cache`: a read-only 1 GiB sweep precedes each single-operator
   graph replay, outside its GPU Event interval. The declared 20 warmups, 100 samples and all 13 cases are retained.
   The older `target_ms` workload field remains for source compatibility but does not batch cold-cache replays.

The 2026-10-22 initial-validation measurements used warm graph timing. They are retained as historical context,
not cold-cache performance or qualification evidence. This migration changes benchmark identity; remeasure
baseline and candidate and require a fresh framework-finalized `task_validator` PASS for the exact source/image.
Do not compare the old ratios with cold-cache results or relabel them as cold-cache measurements.

Implement `out = a @ b.T`. The fixed `n`, `k`, seed and all variable `m` cases
are in the declared workload JSON. Inputs `a` are `[m, k]` BF16 and `b` are
`[n, k]` BF16; the returned output is `[m, n]` BF16 on the same GPU.

The configured builder signature is `builder(m, n, k) -> launch`, invoked with
keyword arguments. Its launch signature is `launch(a, b) -> out`.

The baseline entry is `aiter.tuned_gemm.gemm_a16w16`, with materialized source at
`aiter_source/aiter/tuned_gemm.py`. This is a production dispatch selecting a
kernel for each M bucket; use `get_GEMM_A16W16_config` in that source to inspect
the choice. It can dispatch to AITER FlyDSL kernels or torch/hipBLASLt. The
runner records the selection per case, not an assumed small-M/large-M split.

`task_initialize.py` draws both inputs from a standard normal distribution,
re-seeding each case independently. `task_reference.py` computes FP32 GEMM and
returns BF16. `task_compare.py` requires **every element** to meet its absolute
or relative tolerance (BF16: 0.01 absolute **or** 0.01 relative). Candidate and
replayed candidate must meet that same rule; it is never calibrated against
baseline errors. FP32 intermediate reduction is generally needed.

The baseline policy in this gfx942 task's `config.yaml` is `required`. Baseline and candidate numerical
correctness and actual measured replay outputs must pass their original comparison; neither has a numerical
exception. Runtime failures, input mutation, wrong shape/dtype/device and nonfinite outputs also fail.

The original gfx950 task records family-level history in commits
`bfe1ef1e0b4f2b391f426de31387c3a703d08eb5` and
`329bc9861f7199c4df4d6fc0fc0eb16353cfe995`, plus any task-specific evidence below.
Those records concern different hardware/runtime identities and do not authorize a gfx942 diagnostic waiver.
Actual runs record every case's comparison, executed baseline source hash, package versions and dispatch.

## Task contract

`config.yaml` is the only task configuration. `evaluation.workloads` locates the
protected case data; `candidate.entrypoints` declares the candidate file and the
exact builder symbol. The symbol is not derived from the operator identity.
The initial candidate is the working FlyDSL GEMM described above; the agent optimizes it
in FlyDSL in the declared editable file. All other task files are protected.

The builder receives shape arguments only, once per case, and returns a callable
launch. Prepare compilation, shape-dependent tile choices and reusable scratch
in the builder. Every launch must compute the complete operator on the supplied
current tensor contents; it must not cache answers or alter input tensors.
State kept across launches may be derived only from the weights (for example a
one-time re-layout keyed on the weight tensor); it must never be derived from
activations, routing or outputs. Recognizing inputs seen before and returning a
stored or partial result games the measurement; it is not an optimization.
All 13 variable-axis sizes (1 through 4096, powers of two) are scored. Tiling,
fusion, split reductions and per-shape dispatch are implementation choices.

## Baseline, reference and dependencies

The production baseline calls the **installed AITER package**. The framework
materializes the image's `/sgl-workspace/aiter/aiter` package subtree at
`aiter_source/aiter`, preserving every declared `baseline.source_files` path.
This is read-only explanatory code, not a standalone AITER checkout or an
installation/build input; it must not shadow the installed Python package.
All package Python sources, including `jit/__init__.py`, `jit/core.py`, operator
implementations and configuration data, are retained. Acquisition excludes only
`jit/build`, `jit/flydsl_cache` and the package-root `__pycache__` from this copy.
Repository-level third-party CK headers, examples and packaging files outside
this package are not task inputs. Baseline callbacks execute the installed
package, whose complete repository/build dependencies and runtime artifacts
remain in the pinned Docker image. Task-local references and inputs are bundled
under `scripts/`; they do not import or build the explanatory snapshot.
The selected Docker runtime must provide AITER, FlyDSL and ROCm PyTorch on gfx942.
The runner records runtime versions, the executed baseline module's source hash
and dispatch evidence. No SIKL repository clone is needed: the original
initialize, reference, compare and baseline callbacks are included under
`scripts/`. These callbacks are unchanged by the Arena v2 migration.

Candidate computation must be implemented in FlyDSL. Allowed imports are FlyDSL,
PyTorch for tensor allocation/views/dtypes/launch plumbing, and these host-only
Python modules: `__future__`, `typing`, `collections`, `dataclasses`, `enum`,
`functools`, `itertools`, `math`, `operator`, `numbers`, `abc`, `types`.
Calling a library implementation of the operator, including PyTorch matrix
products, is forbidden. Importing AITER, protected task modules, other local
modules or another GPU computation library is forbidden. Loading code or task
files dynamically is also forbidden. The AST guard checks direct imports,
imported members/aliases and common matrix-product forms. It is a guard against
ordinary violations, not a proof against arbitrary Python reflection. Numerical
and timed-path checks and evaluator review also remain required.

## Evaluation

From a materialized task workspace, the public CLI supports:

```bash
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Each command emits exactly one `ARENA_EVAL_RESULT=` JSON line and exits nonzero
on failure. All case rows retain the same ID, shape, dtype and semantic params.
`validate-task` enumerates the complete protected manifest before executing any
candidate. It checks dependencies, inputs and reference validity; in framework
`task_validation` phase it verifies the declared implemented builder and reports
`metadata.candidate_state`. Compilation executes each specialization, including
lazy JIT compilation on its first launch. Correctness uses the task's original
comparison, independently for the requested role. On each GEMM case, the same
launch is also checked outside scoring with two freshly initialized weights:
first by changing `b` in its original storage, then by passing a different
same-shape `b` buffer. The original reference and numerical gate apply to both
replays, and input buffers must remain unchanged. The scored performance path
continues to hold its weight fixed across samples for both roles. An absent
candidate, missing
builder or `NotImplementedError` always fails candidate actions, in both phases;
only the framework can defer candidate checks for an initially empty task.

Both roles use `ReadOnlyCacheFlush` and `benchmark_cuda_graph_cold_cache_samples` from the materialized
canonical `_aka_benchmark.py`. Each case allocates one initialized 1 GiB eviction buffer outside capture and
timing. Every invocation follows the same stream order: prepare inputs and inspect previous read-only buffers,
read the eviction buffer, start the GPU Event, replay one complete operator graph, end the Event, then observe
its output. Eviction, input preparation, oracle work and compilation are outside timing. Capture failure fails
closed; there is no warm replay, batching, Event-only or CPU-clock fallback. `torch.cuda.empty_cache()` is not
used as data-cache eviction. The reported value is the arithmetic mean of the 100 raw GPU Event samples;
`metadata.samples_ms` retains them, and cache policy/size and the one-invocation graph identity accompany each row.
Baseline and candidate must report the same cold-cache identity.

Before each sample, the call-varying operand `a` is overwritten with one of several draws from the bundle's
initializer, while `b` stays fixed. Draw seeds are selected from operating-system entropy when timing begins.
After the samples, the same captured graph is replayed over further unseen draws with the same preparation
and eviction policy. The fastest unseen invocation must take at most `UNSEEN_DRAW_MARGIN` times the reported
mean. Outputs of randomly selected reported samples and every unseen invocation are compared with the original
reference on the draw each one consumed. Before replacing a draw, the task checks the live activation and fixed
weight byte for byte against their previous expected inputs. Final reported and unseen invocations are checked
before acceptance too. These checks cover warmup/capture, reported samples and unseen reruns, and happen before
the next eviction sweep. Input mutation, nonfinite output, missing work and runtime failures always fail.
Historical warm-cache timings and gfx950 results are not comparable with this benchmark identity.
No timing from an instrumented sanitizer build may become an official score.

A task does not require any agent-specific driver or environment variable.
Forge and other adapters invoke these same commands; their private search
protocols do not change the task's comparison or measurement policy.

## Export

The common Arena post-processing stage invokes `exports[].command` after final
acceptance. `scripts/export_solution.py` reads the framework-owned
`task_result.yaml`, the declared candidate and workload paths, and the protected
`solution.json` template; it writes only the declared artifact output. It
requires compilation, correctness, tool policy and complete comparable timings.
It rejects failed/stub candidates and never computes or writes Arena scores.
The artifact includes the candidate and a tensor-call wrapper for its declared
builder; it does not require consumers to derive a builder name from task ID.
The original template slot name is retained for provenance; the artifact does
not imply that Forge produced it. There is no external publication or sync.
Consumers must support the artifact's `flydsl` language; exporting alone does
not prove compatibility or acceptance by a separate SIKL installation.

## Validation controls

`validate-task` executes every original workload case's input generator and
reference, checking the declared output shape, BF16 dtype, device and finiteness.
In addition it executes independent small known-answer controls and comparator
positive/negative controls once per task. They appear in
`metadata.validation_controls`, with observed values and device, and do not add
or remove scored cases. A failed control fails task validation and cannot be
covered by baseline diagnostics.

For GEMM, an asymmetric 2x3 and 4x3 integer fixture has the literal product
`[[1, 3, 0, 4], [3, 0, -1, -4]]`. Its reference output is checked exactly,
independently of the task comparator. The comparator must accept a nonzero BF16
error within its existing gate (1.0078125 versus 1) and reject a larger one
(1.03125 versus 1), wrong signs, shape/dtype mismatches and nonfinite output.

For MoE, controls check the literal E2M1 codebook, E8M0 powers, nearest-even
midpoint codes, scalar-address weight unshuffling and the gate/up formula using
Python scalar `exp`. A three-expert quantized fixture routes two experts and
leaves one unselected. With D=I=256, sparse activations 1 and 2, expert W1 values
1 and -1/2, and W2 values 1/4 and -1/2, quantized intermediate values are
(.75, .09375) and (4, .25). Routing weights (.25, .75) yield final output rows
filled with 3 and 40; these are hand-derived, not generated by the reference or
initializer. Constant per-expert weights/scales make this fixture independent
of the layout shuffle, while a separate nonconstant layout control checks that
step. The comparator must accept 1.125 versus 1 (about 18 dB SQNR) and reject
1.5 versus 1 (about 6 dB), plus contract violations. Positive controls prevent
an always-reject comparator from being mistaken for a working guard.

These controls exercise specific oracle/comparator properties; they are not
proof of every reference operation for every possible input. They do not
replace complete workload evaluation, an actual accepted candidate, or GPU
validation. The fixture checks' exact/FP32 rounding bounds do not alter the
original candidate tolerances or scored workload data.
