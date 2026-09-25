# Apex

This integration runs the standalone kernel optimizer from Apex's
[`codex/recovery-integration`](https://github.com/AMD-AGI/Apex/tree/codex/recovery-integration)
branch. The exact source revision is pinned in [runtime.py](runtime.py);
the Apex `main` branch is not compatible with this adapter.

## Setup and run

Prepare a **standalone clone** outside the Arena checkout on the GPU host.
The runner mounts this directory read-only. Linked Git worktrees are not
accepted because their Git metadata can live outside the mounted directory.

Apex's standalone kernel path uses ordinary process groups and caller-assigned
GPUs. It requires neither bubblewrap, root execution nor host PID sharing. Use
Arena's architecture-matched scoring image directly. For clusters that require
ordinary container permissions, set `AKA_DOCKER_PRIVILEGED=0`; the default remains
the existing privileged GPU configuration. The scheduler must reserve the GPUs
and prevent overlapping jobs. Apex checks GPU access with a real operation,
without attempting to identify host GPU processes.

```bash
git clone --branch codex/recovery-integration --single-branch \
  https://github.com/AMD-AGI/Apex.git ../apex-runtime
APEX_REVISION=$(python3 -c 'from agents.apex.runtime import APEX_REVISION; print(APEX_REVISION)')
git -C ../apex-runtime checkout --detach "$APEX_REVISION"
export APEX_ROOT="$(cd ../apex-runtime && pwd)"

# Authenticate the selected backend CLI on the host first.
# Select the architecture matching the physical GPU for setup/smoke.
export AKA_GPU_ARCH=gfx950
export AKA_DOCKER_PRIVILEGED=0
make docker-smoke
make docker-setup-apex
make docker-check-agents CONFIG=example_configs/quickstart_apex_mi355x.yaml
make docker-run CONFIG=example_configs/quickstart_apex_mi355x.yaml
```

For MI300/MI300X use `gfx942` and `quickstart_apex_mi300.yaml`.
`docker-setup-apex` installs the small standalone Python dependencies from
[requirements.txt](requirements.txt) into Arena's persistent container dependency
directory. It does not install Apex's separate E2E benchmark dependencies.
Normal preflight checks the clean source pin, runtime import and ordinary
subprocess cleanup. Agent preflight additionally runs a small GPU operation.
It does not download or switch Apex source. The selected backend must support
the pinned runtime's CLI arguments. For three different workload types, use
`example_configs/qualification_apex_mi355x.yaml` (normalization, elementwise
subtraction and matrix multiplication).

## Configuration

Select `agent.template: apex`. Supported overrides, with defaults in
[agent_config.yaml](agent_config.yaml), are:

| Field | Meaning |
| --- | --- |
| `backend` | `codex`, `claude`, or `cursor`; only this CLI and its authentication are mounted |
| `model`, `effort` | Optional backend controls; null leaves the choice to Apex/backend defaults |
| `max_iterations` | Number of upstream optimization attempts within one Arena invocation |
| `max_turns` | Upstream backend turn budget |
| `timeout_seconds` | Total invocation deadline, including preparation and delivery |

For example, set `backend: claude` and the desired model in the run config's
`agent` mapping. Task files contain no Apex-specific settings.

## Task and delivery contract

The adapter supports schema-v2 tasks with an implemented Python or Triton
candidate, explicit editable files, and a declared callable entrypoint.
Translation, generation, HIP/FlyDSL and editable tree scopes fail explicitly
before the agent starts. File and symbol scopes use Arena's existing harness
guard. Full task instructions remain in the copied task package.

1. Arena validates the task and freezes its independent baseline and case manifest.
2. The adapter captures a separate task copy and a local Git snapshot for Apex's
   repository identity. This commit identifies materialized input, not an upstream
   source revision, and is never pushed.
3. The adapter supplies a caller-neutral TaskSpec and binds evaluator authority
   to the exact previewed contract. The upstream optimizer controls search,
   knowledge, assigned-GPU startup checks and backend invocation.
4. The action bridge runs the task's declared commands through Arena's shared
   action executor, preserving action timeouts, numerical checks, case coverage,
   and candidate-evaluation semantics. Checks run in temporary copies so build
   products and reports cannot become undeclared candidate edits.
5. A successful source bundle is checked for task identity, baseline hashes,
   allowed paths, patch hashes, resulting source hashes and protected harness
   content in a scratch copy. Only then are candidate files installed.
6. Arena independently compiles, checks and times the delivered candidate. Apex
   grades and agent text do not supply Arena correctness or scores. A failed
   process or non-deliverable result remains an agent failure; `no_gain` leaves
   the starting candidate intact.

Per-invocation artifacts are retained beside the scored task workspace, in a
directory named `<workspace>-apex-*`. They include the task contract, upstream
results, evaluation contract and bounded process output. These are experiment
artifacts, not source files to commit or publish automatically.

## Execution boundary

The additional source mount is read-only and only enabled for Apex. The container
runs as the caller's UID/GID, with a private PID namespace and an init process to
reap orphaned children. The upstream supervisor bounds and cleans its ordinary
process groups. The outer container is the execution boundary; the default
backend does not start another filesystem/process sandbox inside it. The adapter
starts the pinned package in a separate process with a deadline and bounded
output; it cleans up observed descendants and checks returned patches before
writing the scored workspace. Authentication comes from the existing selected
backend mount and is not placed in task specifications or command arguments.
The runner removes its own container on exit or cancellation. Its private task
copy and process supervision are reproducibility measures, not a hostile-code
security sandbox. Detached processes are ultimately bounded by the container
lifetime. GPU metadata explicitly leaves host ownership unverified. The existing
evaluator and harness guard remain authoritative.

## Tests

```bash
python3 -m pytest -q tests/test_apex_agent.py
make check-docker-runner
```

With `APEX_ROOT` set, the CPU suite also parses the generated TaskSpec and
previews its evaluation contract using the real pinned Apex package. These
checks use no model or GPU and do not establish live optimization performance.
