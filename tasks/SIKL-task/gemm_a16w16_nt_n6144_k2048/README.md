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

The baseline policy is explicit in this task's `config.yaml`. A historical
upstream report in commit `bfe1ef1e0b4f2b391f426de31387c3a703d08eb5`
(2026-09-14, MI355X/gfx950) states that 9 of 17 GEMM tasks failed on 64 of
221 points, but does not enumerate those nine tasks or provide a complete
per-case artifact. This is a family-level report, not evidence that all 17
tasks fail, nor a prediction about another runtime image.

Commit `329bc9861f7199c4df4d6fc0fc0eb16353cfe995` specifically identifies
`gemm_a16w16_nt_n4096_k2048` among tasks rejected by reference replay checking,
and reports repeat-call disagreements at `m_1` and `m_8`. Only that named task
retains `correctness_policy: diagnostic` on this historical basis. The other
16 GEMM tasks use `required` pending task-specific GPU evidence; the aggregate
count alone is not sufficient to declare their deviations known.

These commit messages are provenance, not fresh GPU validation. The actual
runtime must record every case's comparison, executed baseline source hash,
package versions and dispatch. A passing comparison is always reported PASS.
A mismatch is reported FAIL with its real evidence; only a completed finite
`numerical_mismatch` may use the configured diagnostic exception. Crashes,
shape/dtype errors and nonfinite outputs never qualify. Baseline replay keeps
its full verdict under `metadata.replay_correctness`. Unknown deviations need
GPU evidence and a justified policy update, not automatic reclassification.
Candidate correctness and candidate replay have no diagnostic exception.

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

## Production baseline numerical evidence

On 2026-09-15, MI355X/gfx950 job `139315`, `validator-15` ran every original
case under image `sha256:106a7adbeec5554b6e66a4bda0b3694af442717b9fe92754a9885520077b6f93`
with PyTorch `2.11.0+rocm10.0.0` and FlyDSL `0.3.2`. The installed production
baseline was `aiter.tuned_gemm.gemm_a16w16`; its module content SHA256 was
`1fafc9782b43f8c6e198e1d39ce83e01ea3b5892f8b5e1a26c4a5eeb74252b51`.
The complete baseline correctness action SHA256 is
`95770d8f910e865b28d493aa234cc3205a61fe4e1709cd84443761619c950e76`.

Task/reference controls and baseline compilation passed all 13 cases. Every
baseline output passed shape, dtype, device and finiteness checks. The original
comparator reported three finite numerical mismatches:

| M | Status | Elements outside gate | Max absolute error |
| ---: | --- | ---: | ---: |
| 1 | FAIL | 658 | 1 |
| 2 | FAIL | 690 | 1 |
| 4 | FAIL | 1408 | 1 |
| 8 | PASS | 0 | 0.5 |
| 16 | PASS | 0 | 0.5 |
| 32 | PASS | 0 | 0.5 |
| 64 | PASS | 0 | 0.5 |
| 128 | PASS | 0 | 0.5 |
| 256 | PASS | 0 | 1 |
| 512 | PASS | 0 | 1 |
| 1024 | PASS | 0 | 1 |
| 2048 | PASS | 0 | 1 |
| 4096 | PASS | 0 | 1 |

The original required policy stopped before performance and the finalized
validator result was FAIL. This evidence does not qualify the task; the policy
change needs a fresh full validation including actual timed replay.

The supplied production operator remains the independent performance baseline.
Only its completed finite numerical mismatches are diagnostic. Missing cases,
crashes, compile errors, invalid output contracts and input mutation still fail.
Baseline replay retains its full comparison evidence; candidate correctness and
actual timed replay must pass the original comparator without this exemption.
No baseline implementation, reference, tolerance, workload or timing is changed.
This observation does not identify a particular compiler/runtime version as the
cause, nor replace the need to demonstrate an accepted candidate implementation.
