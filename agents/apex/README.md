# Apex agent integration

This adapter runs Apex as a bundle-producing kernel optimizer
while keeping AgentKernelArena as the only benchmark and scoring authority.
Apex receives a caller-neutral `TaskSpec`; it does not receive or write an Arena
score.

## Trust boundary

For each task, AgentKernelArena measures the baseline and freezes its harness
before calling Apex. The adapter then:

1. maps the declared source, symbols, GPU architecture, prompt, backend options,
   phase commands, and budget into Apex's flat `task_spec.json` contract;
2. invokes `Apex/main.py optimize kernel --task-spec ... --result-json ...
   --non-interactive` in a separate process group;
3. validates the result and bundle schema, sizes, paths, symlinks, source hashes,
   changed-file allowlist, patch hashes, and bundle digest;
4. runs `git apply --check` before applying only the declared source patch; and
5. returns control to AgentKernelArena's harness guard and centralized
   compilation, correctness, performance, and scoring pipeline.

The adapter ignores Apex's internal performance and safety fields. A returned
`candidate_ready` patch is still only a candidate until AgentKernelArena checks
it. `no_gain` leaves the workspace at its frozen baseline. Invalid, unsupported,
or infrastructure outcomes fail the task instead of being reported as a 1.0x
optimization.

Run artifacts are written beside, never inside, the scored workspace under
`.<task-workspace>_apex/<run-id>/`.

## Setup

Bootstrap Apex and its pinned Magpie and TraceLens dependencies once from the
Apex checkout:

```bash
cd /absolute/path/to/Apex
python3 scripts/bootstrap_dependencies.py install
```

Then select the Apex backend in `agents/apex/agent_config.yaml`. The default is
Codex; `claude` and `cursor` are also accepted. When switching backend, also set
`model` and `effort` to values supported by that CLI (or `null` to use its
defaults). Install and authenticate that one host CLI, then point the Docker
runner at the Apex checkout:

```bash
export AKA_APEX_ROOT=/absolute/path/to/Apex
make docker-check-agents CONFIG=example_configs/quickstart_apex_mi355x.yaml
make docker-run CONFIG=example_configs/quickstart_apex_mi355x.yaml
```

The runner bind-mounts the Apex checkout read-only at the same absolute path it
has on the host and executes its bootstrapped `.venv/bin/python`. Keeping the
path identical preserves the editable-install receipt without a `PYTHONPATH`
override. It mounts only the selected backend's CLI and login state; it does not
expose all three backend credentials to one run.

## Supported tasks

The first integration slice accepts `triton2triton` tasks whose editable Triton
source is separate from the protected compile/correctness/performance harness.
Each phase must currently contain exactly one argv-representable command; tasks
with multiple commands or shell operators are rejected before Apex starts.
`hip2hip` remains disabled until a task can name a trusted fixed build and
verification recipe. Repository, image-kernel, authoring, translation, and
FlyDSL task types are rejected before the Apex subprocess starts.

## Matched Apex versus Codex benchmark

The two checked-in MI355X configurations contain the same ordered ten-task
vLLM Triton cohort:

```bash
export AKA_APEX_ROOT=/absolute/path/to/Apex

make docker-run \
  CONFIG=example_configs/benchmark_codex_mi355x_10.yaml \
  RUN_ARGS="--run-suffix codex_baseline"

make docker-run \
  CONFIG=example_configs/benchmark_apex_mi355x_10.yaml \
  RUN_ARGS="--run-suffix apex_treatment"
```

The checked-in settings pin both paths to Codex `gpt-5.5`, `xhigh`, and a
3600-second agent budget; Apex gets one outer attempt and its backend may iterate
within that session. Also hold the Docker image, GPU, task order, timeout, and
repetition policy constant. Compare only the resulting AgentKernelArena
`task_result.yaml` files and aggregate reports.

## Bundle contract

`result.json` uses schema version 1 and includes `task_id`, `status`,
`reason_code`, `applied=false`, `external_verification_required=true`,
`bundle_path`, `bundle_digest`, and `changed_files`. A candidate bundle directory
contains only `bundle.json` and its declared patch files. `bundle.json` binds the
baseline file hashes, changed files, and ordered `patches[{path,sha256}]`.

The bundle digest is SHA-256 over canonical JSON for the parsed manifest
(`sort_keys=true`, compact separators, UTF-8) followed by each patch's raw bytes
in manifest order. Digests are lowercase 64-character hexadecimal strings with
no `sha256:` prefix. This binds both metadata and payload without trusting a
self-reported score.
