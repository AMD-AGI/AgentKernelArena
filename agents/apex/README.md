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
   --non-interactive` inside an outer per-attempt PID namespace; Apex creates the
   second private PID/proc boundary around its backend;
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

The shared Arena prompt builder appends large generic architecture and language
cheatsheets after the task-specific objective, source declaration, phase commands,
protected-harness rules, and completion contract. Apex's `ContextPacket` limits any
single mandatory text field to 8,192 characters and already retrieves its own scoped,
provenance-aware architecture and optimization knowledge. For an oversized MI355X
prompt, this adapter therefore preserves the complete task-specific prefix and omits
only the known generic MI355X/Triton suffix. It reconstructs a concise workspace,
editable-source, and target-function handoff from validated TaskSpec fields because
the shared prompt places its workspace prose after the cheatsheets.

The raw TaskSpec records original and adapted byte and character lengths, SHA-256
hashes, the exact boundary, and the transform version. Short prompts pass through
byte-for-byte. An oversized prompt with a missing or ambiguous boundary, or one that
remains over the bound, fails closed; the adapter never applies arbitrary truncation.
Formal TaskSpecs also carry `caller_run_control`, a structured copy of the matched
execution contract. It fixes one deliverable version, 50
`structured_agent_turn_checkpoint_v2` turns (each assistant message and each tool-call start
counts once), and requires the agent to leave its best source in the editable files
before the boundary. The caller-selected `AGENT_KERNEL_ARENA_PYTHON` must be an
absolute, executable interpreter; a formal launch fails closed if it is absent. The
adapter binds Python-based compile, correctness, and performance argv to that exact
path, records all three argv vectors, and renders a compact copy into the instructions.
Receipt validation independently regenerates this text and requires it as the exact
instruction suffix; the structured field alone is not accepted because Apex treats it
as caller metadata. The checked-in ten-task prompt test proves every result stays
within 8,192 characters.

For a formal attempt, the raw TaskSpec is written before launch under a sealed sibling
contract directory and mounted read-only, while Apex's result/artifact directory stays
separately writable. The adapter retains the prelaunch bytes and digest, verifies the
same sealed file after the subprocess exits, and copies those original bytes—not a
post-process reread—into the immutable attempt receipt. The untouched Arena prompt is
also copied into a separate `0444` `original_arena_prompt.txt` receipt artifact. Its
bytes and digest are recomputed against `instruction_adaptation.original`; a mismatch
invalidates the receipt rather than leaving an unbound transformation claim.

The outer attempt namespace replaces `/tmp` and the complete campaign data root with
private filesystems. Before launch, the adapter therefore revalidates the exact
`APEX_ROOT` Git commit, clean-status digest, regular `main.py`, and in-checkout
`APEX_PYTHON` launcher against the runner's sealed campaign provenance. It then
read-only rebinds that one canonical Apex checkout after `/tmp` is hidden. The mount
API rejects `/tmp`, `/var/tmp`, `/dev/shm`, symlinked roots, overlapping roots, and
paths inside campaign data; it never exposes an entire temporary directory. Both the
exact bind list and Apex runtime identity are content-digested into the attempt
receipt, and the campaign auditor binds them back to `repositories.apex` in
`campaign_manifest.yaml`. The scored workspace remains a separate read-only bind,
and only the attempt artifact and fresh backend-home roots remain writable.

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
expose all three backend credentials to one run. For a formal campaign, `.venv`
must be a real directory inside the checkout; a symlinked `.venv` parent is rejected
because its target would not be covered by the exact read-only runtime bind. A normal
venv's final `bin/python` symlink to a system interpreter is accepted because system
paths remain visible read-only.

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

make docker-parallel-run \
  CONFIG=example_configs/benchmark_codex_mi355x_10.yaml \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--run-suffix codex_baseline"

make docker-parallel-run \
  CONFIG=example_configs/benchmark_apex_mi355x_10.yaml \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  RUN_ARGS="--run-suffix apex_treatment"
```

The checked-in settings pin both paths to Codex `gpt-5.5`, `xhigh`, a 50-turn
structured-agent limit, and a 3600-second inner-agent budget. The campaign
controller creates three fresh workspaces and invokes three independent Codex
sessions per task on both paths;
`max_iterations: 1` prevents an additional hidden inner campaign. Every attempt
returns to AgentKernelArena for centralized compilation, correctness, and performance
evaluation. Selection is deterministic: correctness-qualified attempts rank by their
measured rate, with the lower attempt number breaking an exact tie.

Use the exact same ordered `GPU_IDS` pool for both commands. Policy
`deterministic_task_gpu_v1` maps ordered task index `i` to
`GPU_IDS[(i - 1) % len(GPU_IDS)]`; queue descriptors carry that physical host ID and
only the matching worker may claim them. This supports a single GPU or an ordered
pool such as `0,1,2,3,4,5,6,7` without dynamic-scheduler drift between treatments.
Each Apex attempt has a separate 3,600-second allowance for Apex-owned freeze,
compile, correctness, safety, measurement, and bundle work after the inner Codex
budget. The outer 25,200-second task budget therefore covers three
`(3,600 agent + 3,600 Apex internal)` reservations plus a 3,600-second central
evaluator allowance. Direct Codex does not consume the Apex allowance, but both
treatments retain the same immutable outer policy and hard task deadline.

Each run writes `campaign_manifest.yaml`, pinning the AgentKernelArena and Apex Git
commits and clean-state digests, Codex binary hash/version/model/effort/permissions,
the Docker reference plus its daemon-inspected content ID and repo digests, complete
evaluator/task-package hashes, every GPU's unique ID/serial/model/gfx plus task
mapping, and a live runtime-isolation receipt. The receipt proves a non-root UID,
zero inheritable/permitted/effective/bounding/ambient capabilities, NNP, the pinned
seccomp/AppArmor/Yama state, the exact `bwrap` and Codex binaries, the managed
Codex policy hash, private PID/IPC namespaces, private shared memory, rebuilt Docker
system-path masks, and pidfd-bound teardown. Direct Codex receives AKA's private
procfs. The Apex arm instead leaves procfs inherited only around the trusted outer
orchestrator, because Apex owns the private procfs around its backend and an earlier
procfs layer prevents the Apex-to-Codex nesting. A separate live probe must prove that the
managed profile permits workspace writes while denying credential reads and
command network. A content-pinned bubblewrap compatibility shim is copied into a
sealed memfd and mounted from that exact descriptor before restoring only the
`/dev/kfd` and render nodes already admitted by Docker after Codex creates its
private `/dev`. Its parent is a dedicated read-only mountpoint, and live
rename/unlink/replace/write attacks must fail; the same probe sees exactly one ROCm
device and completes a Torch allocation plus reduction on it. It also
proves the backend command is outside the worker PID namespace and blocks PID-1
root/environ/mem credential aliases. Init,
every worker, and postprocess independently reproduce this receipt before accepting
the immutable manifest. The comparison contract explicitly names
`aka.task-package-objective-and-protected-harness/v1` as its objective policy and
`aka.shared-objective-backend-native-context-receipted/v1` as its prompt policy.
The latter keeps the task objective and protected harness common while allowing the
documented, receipt-bound Apex context adaptation. `comparison_contract_sha256`
excludes the treatment template/config and
run-specific GPU lease fields (run name, PID, timestamp, receipt hash, and lock path),
while retaining the common lease policy, physical unique IDs, protected device paths,
and GPU boundary-plan digest; it must match across the Apex and direct-Codex runs.
Per-task attempt evidence remains
under `.campaign_attempts/`. Both treatments retain a read-only session receipt
with exact backend/model/effort and invocation identity, bounded process output,
the same 50-turn structured-agent policy and 16 MiB inner Codex stream bound, and
verified namespace teardown. Each receipt directly carries the comparison-contract
digest; postprocess recomputes the immutable manifest contract and rejects a missing
or different attempt binding. The direct-Codex receipt also freezes the exact rendered
prompt bytes, so its invocation prompt hash is independently reproducible rather than
self-attested. The Apex adapter separately caps its outer transport
at 4 MiB. Apex receipts additionally snapshot the
TaskSpec, TaskResult, checksummed event journal, canonical agent transcript, and
terminal verdict lineage. The event-bound inner agent prompt is also copied into the
outer immutable receipt, so later audits do not depend on a still-live Apex CAS path.
New formal receipts use `agentkernelarena.apex-attempt-receipt/v4`; the campaign
manifest freezes that schema before any attempt starts. The auditor keeps explicit
read-only support for sealed history, but a v4 receipt cannot drop these fields or
change its schema to select the legacy validation path. Receipt dispatch is selected
from the sealed manifest's agent template and schema, never from the receipt's own
type claim, so an Apex and direct-Codex receipt cannot be substituted for each other.
The evaluator metadata gate and receipt auditor use the same campaign-owned schema
registry; adding a generation in one place cannot leave a stale evaluator allowlist.
The adapter requires
exactly one `prompt_sent` CAS binding
and a direct journal parent edge from that event to the sole `agent_completed` or
`agent_failed` event. The outer receipt records the prompt event ID, digest, and size
as an event-bound CAS fact; it explicitly does not claim that these bytes were
independently attested at stdin transport. Both adapter and postprocess parse the
canonical `Identity and role` JSON in those exact bytes and require
`role.objective` to equal the sealed TaskSpec instructions, including the full formal
run-control suffix.

Successful `candidate_ready` and `no_gain` results require one `agent_completed` and
no `agent_failed`, plus one through 50 recomputed structured turns. A
`budget_exhausted` result instead requires exactly one
`agent_failed`, an `agent_turn_budget_overrun` result/error/decision chain, typed
`turn_overrun` termination above the 50-turn bound, and matching invocation,
transcript, event, and private-PID-namespace receipts. The observer destroys the
pidfd-pinned namespace init, so the bound process exit is 137; SIGTERM or another
numeric exit cannot substitute for that proof. Formal lineage is validated before the
outer Apex return code is rejected, so a failed session retains audit evidence while
still raising and keeping `session_succeeded=false`. A nonzero `no_gain` is invalid.
At the exact 50-turn boundary, both arms bind
`private_pid_namespace_init_pidfd_v1` through comparison-contract v4, invocation,
transcript, event, and attempt receipts. Each backend runs behind a gated private PID
namespace with a private procfs. The trusted supervisor pins namespace PID 1 with a
pidfd and freezes candidate bytes only after init exit, wrapper status/EOF, stream
EOF, and visible-membership corroboration. The receipt counts inaccessible sibling
`/proc` entries instead of claiming a complete scan; exact namespace-init pidfd exit
remains the authoritative Linux teardown proof. Killing namespace PID 1 contains
`setsid`, double-fork, clear-environment, closed-stdio, immediate-exec, and late-write
descendants without trusting PGIDs or reusable numeric PIDs. Apex's inner receipt is
then bound to AKA's separate outer namespace teardown. Its exact-boundary candidate
must additionally traverse the frozen-source, compile, correctness, safety,
measurement, reward, decision, and immutable-bundle gate chain. A count of 49 is not
an exact-boundary checkpoint, 51 is always an overrun, and timeout, truncation,
unverified containment or cleanup failure is ineligible. Older schemas cannot claim
this checkpoint path. These receipts only govern source persistence; AgentKernelArena's
outer evaluator remains the sole authority for scored correctness and performance.
The AKA outer layer exposes writable inherited `/proc` only to the trusted Apex
orchestrator so nested uid/gid maps can be established; Apex replaces and remasks
that procfs before the Codex backend starts. A live three-layer CPU test exercises
AKA outer, Apex inner, and a nested managed-command namespace together, including
direct credentials and `/proc/1/{root,environ,mem}` alias attacks.

An exact 50-turn stop is a candidate checkpoint; a provider event beyond that bound
is typed `max_turns_overrun` and cannot persist candidate bytes. A valid `no_gain` is
an audited successful session but its central baseline replay is marked
`no_candidate_baseline_replay_v1` and can never enter campaign selection. Direct
Codex receives the same treatment when its verified final declared-source delta is
empty: both arms keep `attempt_completed=true`, keep the session receipt valid, and
set `selection_eligible=false`, so neither can score the unchanged baseline. Three
`no_gain` sessions produce no canonical projection; a mixed task may still select a
valid candidate session. Failed sessions and untrusted evidence remain
diagnostic-only: they cannot create the ordinary task workspace or count as
completion. Only a complete campaign gets a canonical copied projection of its
selected attempt.

The checked-in campaign workspaces and logs live under `/data/viouyang/apex/aka`, not
the smaller home filesystem. Both repositories must be clean before formal campaign
initialization. The runner fails closed if image inspection, repo digests, GPU identity,
source manifests, or worker affinity cannot be proven. During a formal campaign the AKA
checkout is mounted read-only. Bubblewrap hides the shared campaign result tree
from each agent. Direct Codex receives only its current workspace and fresh auth-only
home writable; its raw changes are receipted and then reduced to declared
`source_file_path` content. Apex instead receives the scored Arena workspace through an
explicit read-only bind: only its separate result/artifact root and fresh auth-only home
are writable. The adapter verifies a full pre-apply workspace manifest before it may
apply a validated Apex bundle, so `no_gain` cannot retain direct or undeclared edits.
Both treatments use strict `approval_policy=never`, ignored user config and exec-policy
rules, an ephemeral session, a private outer IPC namespace, and private `/dev/shm`.
The content-pinned `/etc/codex/requirements.toml` selects the only allowed managed
permission profile, disables hooks and command network, and denies sandboxed
commands access to `~/.codex/auth.json`; the Codex supervisor can still authenticate
before launching those commands. A missing or unusable outer `bwrap`, changed GPU
shim, different requirements
file, or any failed live property probe aborts campaign preflight. Formal Docker
workers remain non-root and receive no added Linux capabilities. The outer attempt
boundary always creates a private PID namespace. For the Apex arm only, that outer
trusted-orchestrator layer inherits the worker procfs; Apex's inner supervisor creates
and remasks the backend-visible private procfs before Codex starts. Direct Codex never
receives the inherited worker procfs.

Formal GPU workers do not use `--privileged`, `/dev/mem`, or the complete `/dev/dri`
tree. A host-resolved plan maps each physical `unique_id` through KFD topology to its
exact (possibly non-contiguous) render nodes; Docker receives only `/dev/kfd` and those
nodes. Container preflight requires ROCm, Torch, and KFD to agree that exactly the
assigned physical GPU is usable. The host holds one nonblocking `flock` per physical
unique ID for the entire runner lifetime. After acquiring those leases, it calls
`rsmi_compute_process_info_get` directly through `librocm_smi64`, checks init, count,
fetch, and shutdown return codes, and publishes an immutable structured KFD process
inventory. Any reported KFD PID or any API uncertainty fails closed; `/proc/*/fd` is
retained only as supplementary render/KFD evidence because host permissions can hide
descriptors. It never terminates an unrelated GPU process. Boundary, process-inventory,
and exclusivity receipts are bound into the run manifest and every attempt receipt.

This comparison uses each task's AgentKernelArena-native 100-repetition timing and
external score. It is **not** an `apex.kernel-measurement/v1` report, does not satisfy
Apex's canonical 300-raw-sample p50/p99 grade, and must not be presented as one.
Eligibility additionally requires identical testcase identities/counts, positive finite
timings, 100 configured repetitions per testcase, consistent timing methods, and no
speedup-calculation error. Compare selected results only when the two
`comparison_contract_sha256` values match and all three sessions succeeded.

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

## Focused tests

The adapter tests are CPU-only. Run `python3 -m pytest -q -p no:cacheprovider
tests/test_apex_agent.py` in an environment with the project test dependencies.
For the checked MI355X runtime, the equivalent repository-wide command is:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --entrypoint /bin/bash -e PYTHONDONTWRITEBYTECODE=1 \
  -v /usr/bin/bwrap:/usr/bin/bwrap:ro \
  -v "$PWD":/workspace:ro -w /workspace \
  lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260705 \
  -lc 'python3 -m pytest -q -p no:cacheprovider tests'
```
