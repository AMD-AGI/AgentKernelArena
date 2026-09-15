# BF16 GEMM to FlyDSL

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

The baseline declares `correctness_policy: diagnostic`: the production tuned
implementation has known finite numerical deviations on some cases and remains
a useful runtime comparison. Its correctness command still reports FAIL with
complete per-case `numerical_mismatch` evidence and exits nonzero; the framework
owns the initial-validation exception. Baseline replay records its full verdict
under `metadata.replay_correctness` while permitting only such finite numerical
mismatches during baseline timing. Candidate checks have no such exception.

## Task contract

`config.yaml` is the only task configuration. `evaluation.workloads` locates the
protected case data; `candidate.entrypoints` declares the candidate file and the
exact builder symbol. The symbol is not derived from the operator identity.
The initial candidate is an empty module. The agent must implement the operator
in FlyDSL in the declared editable file. All other task files are protected.

The builder receives shape arguments only, once per case, and returns a callable
launch. Prepare compilation, shape-dependent tile choices and reusable scratch
in the builder. Every launch must compute the complete operator on the supplied
current tensor contents; it must not cache answers or alter input tensors.
All 13 variable-axis sizes (1 through 4096, powers of two) are scored. Tiling,
fusion, split reductions and per-shape dispatch are implementation choices.

## Baseline, reference and dependencies

The production baseline calls the **installed AITER package**. The framework
materializes read-only explanatory source at the location in
`baseline.source_files`; this copy must not shadow the installed Python package.
The selected Docker runtime must provide AITER, FlyDSL and ROCm PyTorch on gfx950.
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
`task_validation` phase it verifies the actual initial stub and reports
`metadata.candidate_state`. Compilation executes each specialization, including
lazy JIT compilation on its first launch. Correctness uses the task's original
comparison, independently for the requested role. An absent candidate, missing
builder or `NotImplementedError` always fails candidate actions, in both phases;
only the framework can defer candidate checks for an initially empty task.

Both roles use the materialized canonical `_aka_benchmark.py` helper with the
unchanged workload warmup, repetition and target duration. Preparation and input
allocation occur outside timing; the timed callable is the complete operator
invocation, including its device work and output/scratch use. Graph timing is
preferred; event fallback is recorded and the framework checks that baseline
and candidate timing methods match. The actual timed invocation is replayed
with freshly initialized inputs and poisoned outputs, then checked against a
new reference using the original comparator. Input mutation, nonfinite output,
missing work or runtime failures cannot be treated as numerical diagnostics.
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
