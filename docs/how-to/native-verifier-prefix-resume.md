# Explicit native verifier prefix resume

This optional route reuses an already completed **compile + correctness** prefix
after an interrupted direct-verifier run. It is limited to one task and an
unsharded `verify` invocation. It does not change a task harness, reduce cases,
draws or timing samples, or create a framework PASS or `task_result.yaml`.

The new workspace always comes from the current repository. Its complete
materialized `source_identity` must equal the pinned prior per-task report.
The same public manifest/config digests and gfx950 architecture are required;
the current physical GPU architecture is checked through PyTorch. Physical GPU
indexes and user-owned cache paths may differ. Tasks with `.pt` inputs are
refused because v1 source inventories exclude those payloads. No old source,
overlay, cache, executable, or tensor pickle is loaded from the evidence bundle.

## Evidence bundle

Explicitly stage authentic collected evidence under the local repository, then
write a descriptor like this. All paths are relative to the descriptor directory
and must stay within it, without symlinks or `..` traversal:

```json
{
  "schema": "aka-native-prefix-resume-v1",
  "repository_root": "repo",
  "workspace": "repo/prior-run/worker-000/000-elementwise_copy_cluster",
  "task_report": {
    "path": "repo/prior-run/worker-000/000-elementwise_copy_cluster.direct.json",
    "sha256": "<SHA-256 of that exact report>",
    "bytes": 12345
  },
  "run_report": {
    "path": "repo/prior-run/worker-000/direct-verification.json",
    "sha256": "<SHA-256 of that exact report>",
    "bytes": 12345
  },
  "runtime_report": {
    "path": "repo/prior-run/worker-000/000-elementwise_copy_cluster/build/runtime_preflight.json",
    "sha256": "<SHA-256 of that exact report>",
    "bytes": 12345
  },
  "origin": {
    "job_id": "159222",
    "source_commit": "<original code snapshot>"
  }
}
```

Replace every hash/size placeholder with the actual bytes. Include the original
retained `direct-native-reports/compile_report.json` and
`direct-native-reports/correctness_report.json` beneath the declared workspace.
Their hashes and byte sizes are checked against the pinned per-task report.
The runtime report must be successful, `phase: complete`,
`native_resolution_complete: true`, and report actual `architecture: gfx950`.

The full per-task `workspace` path must resolve beneath the declared historical
`repository_root` to the explicitly selected workspace; matching only a basename
is insufficient. The prior worker shard must include this task. Runtime profile,
run target, and required environment values such as the TVM FFI flag must match
the current plan. Incidental cache paths and GPU ordinals may differ.

The run summary may have `status: running` and an empty `tasks` list when
preemption interrupted performance before the task returned. The separate
per-task `.direct.json` must still contain the exact contiguous successful
compile/correctness entries, all original commands in order, matching timeouts,
integer zero return codes, `timed_out: false`, and native report status `ok`.
Failed, partial, or previously reused prefixes are rejected. Old performance
reports are never read or reused.

Hash the descriptor itself with `sha256sum`; its SHA-256 must be supplied on
every resume invocation. There is no latest-run search or automatic discovery.
The hash pins the selected bytes; it does not authenticate an unknown producer.
The caller must choose authentic collected native evidence. `origin` is retained
as caller-supplied provenance under that same descriptor pin.

## Run a fresh performance phase

Use a single-task config with the task's normal public runtime and unchanged
commands. For example, with a descriptor staged at `resume_inputs/prefix.json`:

```bash
AKA_VISIBLE_GPU=0 python3 src/scripts/top5_head_kernels.py verify \
  --config example_configs/<single-task-config>.yaml -- \
  --resume-prefix resume_inputs/prefix.json \
  --resume-prefix-sha256 <descriptor-sha256>
```

The standard agentless Docker route still verifies the current image identity.
Any evidence/source/runtime mismatch fails explicitly before task commands;
the tool does not silently fall back to a different run. A fresh output directory
is created and existing output is never overwritten. Independent single-task
invocations may use different explicitly selected GPUs; `parallel-verify` does
not accept resume-prefix arguments.

Only the unchanged full `performance_command` runs in the fresh workspace. The
task's performance path still repeats its complete correctness gate internally.
The old compile/correctness reports are retained only under `resume-evidence/`,
never installed as fresh `build/` reports. The complete pinned JSON evidence
and its relative layout are retained there for independent rechecking.

## Result semantics

Successful resumed runs use `native_prefix_reused_performance_succeeded`, rather
than claiming all phases executed in this run. Reused phase entries have
`status: native_phase_reused`, `executed_here: false`, and
`executed_in_this_run: false`. Their original commands, timestamps, return codes,
and report provenance remain in `prior_phase`; fresh performance is marked as
executed here. `resume_provenance` records the descriptor hash, origin labels,
original runtime identity, and original incomplete-run status.

Downstream admission must explicitly verify this resumed status and its pinned
prefix evidence. It must not treat the new tag as ordinary fresh execution or
as framework validation. `framework_task_validator` remains `NOT_RUN` and
`framework_PASS_claimed` remains false.
