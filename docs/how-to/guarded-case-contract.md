# Guarded case contracts and fresh-container evaluation

New task packages can use the portable contract in `src/task_contract.py` to
check exact expected-case coverage and replay work. Package it unchanged as
`ut/evaluation_contract.py`; isolated tasks import that local file and do not
import this repository's `src` package. This opt-in interface does not change
the already qualified native-quant task or its dedicated evaluator.

The task still owns its kernel loader, real compilation check, independent
oracle, source-edit boundary, and synchronized device timer. The common helper
verifies their case metadata and orchestrates resets and checks. A metadata
flag alone does not demonstrate correct kernel execution. Qualification must
show that a no-op edit of the submitted kernel and corrupted output fail through
the same loader and evaluator used for valid source.

## Package configuration

Use the existing source list and the protected three-phase entrypoint:

```yaml
source_file_path: [source/kernel.py]
compile_command: [python3 scripts/task_runner.py compile]
correctness_command: [python3 scripts/task_runner.py correctness]
performance_command: [python3 scripts/task_runner.py performance]
harness_protection:
  reject_new_source_symlinks: true
trusted_evaluation:
  schema_version: 1
  case_manifest: cases.json
  contract_file: ut/evaluation_contract.py
  source_guard: ut/source_guard.py
  reference_sources:
    source/kernel.py: ut/reference/kernel.py
  requires_aiter_jit_cache: true
```

Set `requires_aiter_jit_cache` only when the task imports AITER or needs its JIT
tree. The frozen source copy must equal the shipped editable source. Every
declared source needs its own reference entry. Editable paths cannot also be
the manifest, contract, guard, runner, or reference files.

The ordinary optimizer freezes every non-editable file shipped by the task
before launching the agent. For `trusted_evaluation` packages, this protection
also applies when only a materialized workspace is available, and the
validator's authoritative `protected_paths` reports that same boundary. It
includes the complete task-local helper tree, case and fixture manifests,
materialized fixture payloads, provenance, and non-editable source wrappers.
Declared source files remain subject to the task's GPU-body guard. Top-level
build, log, validator, and environment caches remain runtime output; nested
same-name directories containing task inputs stay protected.

The guard must expose `validate_sources(candidate_root, reference_root)` and
raise on edits outside the intended GPU implementation boundary. It runs on
the host from the trusted Git package before any container starts. Keep it
CPU-only and freeze decorators, host launchers, import-time effects and other
harness code as appropriate to the task. A C++ kernel body guard and a Triton
function guard need different language-specific checks.

Whole editable host wrappers are not an acceptable optimization boundary:
they can start streams or threads, move work outside the timing interval, or
interfere with the coordinator. A producer implemented as a Python DSL builder
needs a conservative AST allowlist for its specific GPU-building operations.
Review the actual guard and the actual GPU source, including templates or
assembly when those define the kernel. Selecting a different prebuilt binary
from a host wrapper does not establish an editable GPU implementation.

After writing the configuration, package and check the shared helper:

```bash
python3 src/tools/materialize_task_contract.py --task tasks/headkernel_sg520/TASK
python3 src/tools/materialize_task_contract.py --task tasks/headkernel_sg520/TASK --check
```

The materializer only accepts explicitly opted-in tasks and uses the required
rclone transfer flags. Commit the task-local helper with the package.

## Expected cases

Replacement-port tasks can declare an explicit native-production scoring policy:

```yaml
scoring_baseline:
  schema_version: 1
  kind: native_production
  native_source_manifest: provenance/NATIVE-BASELINE.json
```

This policy uses the already measured native-production and candidate-port graph
legs on matched inputs. The protected performance report embeds a
`native-production-comparison-v1` record bound to its complete request, current
source hashes, case manifest, image, and native-source provenance. The host
revalidates every leg with the case contract and derives timings from raw samples.
Absent or mismatched evidence rejects scoring; it never falls back to a port
baseline. No implementation is selected by source hash.

Arena results and trusted retests report native/candidate as the primary ratio,
and keep frozen-port/candidate-port improvement as `port_to_port_speedup_ratio`.
The aggregate `production_kernel_improvement` flag requires a primary ratio
greater than one. Per-case ratios, regressions, and `all_cases_faster_than_native`
remain explicit. Slower correct starters remain measurable and can qualify as
valid tasks. These isolated operator measurements do not establish serving gains.

`cases.json` contains `schema_version: 1`, `runtime_image` with the exact image
digest, a nonempty `cases` array, and a `measurement` policy. The image must also
match `headkernel.docker` in the task configuration.

Each case has a unique `case_id`, positive `occurrences` from the observed
workload, positive `calls_per_sample`, an explicit `scalars` object, and a
`tensors` object keyed by argument name. Each tensor records:

```json
{
  "role": "input",
  "shape": [8192, 1536],
  "strides": [1536, 1],
  "storage_offset": 0,
  "dtype": "bfloat16",
  "device_type": "cuda"
}
```

Roles are `input`, `output`, or `inout`. Include every semantically relevant
scalar, including nulls, booleans, scales, bounds, activation choices and
layout switches. The helper preserves JSON types: `true`, `1`, and `1.0` are
different arguments. Preserve observed multiplicity even when several calls
share a deduplicated ABI. `calls_per_sample` specifies the work within one timed
replay; `occurrences` describes its observed workload frequency. The default
Arena aggregate remains the arithmetic mean of per-case speedup ratios.

The measurement policy is explicit and frozen:

```json
{
  "method": "cuda_graph",
  "warmup_iterations": 10,
  "benchmark_iterations": 100,
  "correctness_seeds": [0, 1],
  "refresh_inputs": "each_replay",
  "initialize_outputs": "each_replay",
  "validate_outputs": "each_replay",
  "negative_controls": ["no_op", "wrong_output"]
}
```

Counts can differ by task when justified before qualification. Reference and
candidate use the same committed policy. Additional negative controls are
allowed and must all reject invalid work.

Use `observe_case(expected, tensors, scalars)` on the actual runtime objects.
It reads tensor shape, dtype, strides, storage offset and device type and checks
them against the complete expected case. Do not manufacture an observed ABI by
copying only the case ID from the manifest.

## Worker and report protocol

The protected runner accepts a positional phase (`compile`, `correctness`, or
`performance`) and optional `--request PATH`. The trusted host passes a fresh
read-only request file. For ordinary framework invocation without that flag,
the protected task runner creates its own fresh request with source/package
hashes. Reports are written to `build/PHASE_report.json` and nonzero exits must
propagate to the caller.

Every report includes `schema_version: 1`, `status: "ok"`, the complete
`request` object, and `cases`. Compilation additionally reports `compiled: true`
after compiling the current source and uses an empty case array.

Correctness rows contain `case` (the complete observed ABI), `correct: true`,
the exact `seeds` list and a `negative_controls` mapping whose required values
are all `true` after invalid work was rejected. All expected cases must appear
exactly once, including cases with identical tensor shapes but different
scalar arguments or call multiplicity.

For performance, `checked_replays` accepts protected callbacks:

- `reset_inputs(seed)` copies fresh input values and restores every mutable
  input/state used by the kernel. Return CPU-owned truth: either CPU reference
  outputs or an input snapshot for a deferred reference. Do not compute or keep
  the current golden output on the GPU before the candidate executes.
- `initialize_outputs()` poisons pure outputs or restores the required initial
  state of output/inout buffers. A no-op must be rejected by the oracle.
- `observe()` describes the actual complete case through `observe_case`.
- `replay()` runs exactly one captured graph with `calls_per_sample` operations.
- `measure(call)` invokes `call` exactly once and returns synchronized device
  milliseconds. Reset, reference calculation and oracle checks remain outside
  this timing interval.
- `verify(truth)` first snapshots candidate outputs and immutable input storage
  to CPU. When the independent reference needs the GPU, compute it only after
  those observations are frozen, using the CPU input snapshot. Compare the
  frozen observations and raise on wrong output or illegal input mutation.

The helper executes resets and validation for warmups and every measured replay.
It rejects timers that omit or repeat the replay and returns a performance row
with raw `samples_ms`, observed case, timing method and validation counts.
The host derives means from finite positive samples; candidate-supplied means
and speedup fields do not enter scoring.

Call `finalize_report(report, manifest, request)` before publishing a task-local
report. It validates the report and adds Arena-compatible `test_cases` derived
from the raw samples. `validate_report` rejects a preexisting `test_cases` array
that disagrees with those samples. `strict_json` rejects duplicate JSON keys and nonfinite constants.
Case hashes are diagnostic identities; actual oracle checks remain mandatory.

## Trusted host retest

After stopping the optimization worker, run from trusted host infrastructure:

```bash
python3 src/tools/trusted_task_eval.py \
  --repo TRUSTED_CHECKOUT --commit FULL_TRUSTED_COMMIT \
  --task tasks/headkernel_sg520/TASK \
  --candidate-workspace AGENT_WORKSPACE \
  --render-device /dev/dri/renderD128 \
  --scratch-dir LOCAL_SCRATCH --output NEW_OUTPUT_DIRECTORY
```

The trusted retest preserves task-generated JSON, JSONL, and log diagnostics
before removing disposable build directories. JSONL replay receipts retain the
case, leg, challenge seed, and iteration when a native comparison fails. Copies
are hashed after transfer; image-cache file lists are bounded to 64 entries and
disable multithreaded streams while retaining the configured transfer count.

The evaluator extracts the task from Git and accepts only regular files named
by `source_file_path` from the candidate workspace. It discards agent reports,
caches and harness files by never staging them. It uses the same protected
entrypoint for original and submitted sources in six fresh containers, with
read-only task/image mounts, private caches, no network or agent credentials,
and the unprivileged runtime UID. AITER tasks use the verified complete image
cache initializer. Requests and complete case identities are checked on the
host, including multiplicity and scalar types.

For tasks with `requires_aiter_jit_cache: true`, each phase retains a complete,
byte-verified copy of the image cache at `AITER_JIT_DIR=/aiter-jit`. The portable
evaluator separately sets
`FLYDSL_RUNTIME_CACHE_DIR=/aiter-jit/fresh_flydsl_<request_id>` and rejects that
path if it already exists, including as a symlink. The fresh request ID gives
each reference/candidate compile, correctness and performance phase its own
initially absent FlyDSL cache. Image-built FlyDSL artifacts must not mask
submitted source changes. Existing AITER modules, lookup paths and copied
image-cache files remain intact. This changes only the FlyDSL cache selection;
task timing, warmups, sample counts and correctness checks are unchanged. Tasks
without the AITER cache opt-in keep their existing cache environment.

GPU selection maps the requested render node to KFD's `drm_render_minor` and
`unique_id`, then sets `ROCR_VISIBLE_DEVICES` to the corresponding `GPU-` UUID.
It does not assume render order, KFD-node order, or physical GPU ordinal zero.
Every phase first verifies that HSA exposes exactly that GPU UUID and HIP
exposes exactly one device at the expected PCI address. The preflight runs
before the task in the same process and its JSON proof is retained with phase
diagnostics. Missing UUIDs or ambiguous partition mappings are rejected.
`/dev/kfd` access is not a complete hardware security boundary.

The UUID format follows ROCr's `HSA_AMD_AGENT_INFO_UUID` implementation and
Linux KFD topology properties. Inspect a host mapping without launching work:

```bash
python3 src/tools/gpu_binding.py --render-device /dev/dri/renderD129
```

`trusted_measurement.json` is measurement evidence, not an Arena task-result or
task-validator completion report. The host, Git database, Docker daemon, pinned
image and GPU driver remain trusted. Task qualification and fresh GPU evidence
are still required; this protocol does not claim to prevent every native-code
or driver attack.
