# mini_swe_triton compatibility status

`mini_swe_triton` remains registered for legacy external tasks. **It does not
implement the public v2 task contract.** All retained Arena tasks use v2, so
calling this launcher for them raises `MINI_SWE_V2_UNSUPPORTED` before
looking for candidate files, creating logs, initializing Git, or starting mini.
A framework `ARENA_TASK_CONTEXT` also prevents a stripped version field from
falling through to the legacy path. This is an agent capability error, not a
task correctness rejection or a successful optimization.

## Runtime evidence and limits

Read-only inspection of the available GEAK checkout at
`c0c0e2aee5e2bec70583253058382523bdf7a3ab` found no `minisweagent` source tree.
Its `pyproject.toml` packages `geak`; its README describes Claude Code Workflow
engines. Neither the inspected host Python nor the CPU test environment had
`minisweagent` installed or a `mini` executable available.

The local GEAK Git history does contain a different, older implementation.
At `f85d0b2275f64587d5167d4a4f1c305e2f634082`,
`src/minisweagent/run/mini.py` declares the CLI options used by this launcher:
`--task`, `--test-command`, `--repo`, `--num-parallel`, `--gpu-ids`, `--model`,
`--yolo`, `--exit-immediately`, `--output`/`-o`, and `--cost-limit`.
Its finalizer in `src/minisweagent/run/postprocess/finalize_apply.py` applies
the selected result to `--repo`. Directory names and arbitrary patch files are
not an authoritative best-result protocol for Arena.

This historical source inspection is **not** an installed runtime, a dependency
qualification, or an endorsement of that revision. No upstream checkout was
changed and no model/GPU run was made for this correction. Installing the current
GEAK package does not restore the old mini CLI; a generic mini-swe-agent release
must not be assumed to implement the fork's options or delivery behavior.
The launcher never substitutes native GEAK for mini.

Docker provisioning is a separate limitation: the shared `docker-check-agents`
allowlist does not include mini, and the shared runner has no dedicated
`GEAK_SRC` provisioning path. Registration and the legacy code below therefore
do not establish a working standard Docker deployment. Adding that support
requires an explicit runtime/provider integration and separate qualification.

## Retained legacy behavior

For an external legacy task without a versioned/v2 declaration or framework
v2 context:

- `GEAK_SRC` must identify an absolute source directory containing
  `minisweagent/run/mini.py`. Missing source raises
  `MINI_SWE_RUNTIME_UNAVAILABLE` before workspace mutation. File presence is
  only a dependency check, not proof that a particular fork is compatible.
- Exactly one `source_file_path` is supported. The historical defaults remain
  `kernel.py` and `test_kernel_harness.py`; nested paths are preserved and must
  stay within the workspace. The kernel and harness must be separate files.
  Multi-file and colocated symbol-scoped tasks are not supported by this path.
- The legacy harness must implement `--correctness`, `--benchmark`, and
  `--full-benchmark --iterations`. These flags are not the v2 action protocol.
- Existing agent configuration and `GEAK_GPU_IDS`/run-level `gpu_ids` remain
  the legacy controls. GPU IDs must match the process-visible allocation.
- The CLI receives literal argv. Its shell-valued `--test-command` retains the
  historical two-command sequence with quoted task-relative paths.
- Each invocation gets a new sibling output directory. Git setup failures,
  CLI nonzero exits, and timeouts raise errors. Timed-out process groups are
  killed and reaped; available stdout/stderr remain in invocation logs.
- Delivery consists only of files the CLI leaves in the exact `--repo`
  workspace. Arena does not apply a guessed patch or scan sibling workspaces.
  Patch-only forks need a separately verified delivery adapter. A zero exit
  with unchanged files does not imply an optimization or an accepted candidate;
  the ordinary evaluator still checks the retained implementation.

## Requirements for future v2 support

A complete adapter needs an available, pinned mini runtime with inspected CLI
and result semantics. It must consume the shared `TaskSpec` and immutable
`ARENA_TASK_CONTEXT`, honor declared nested/multiple candidate files and file,
symbol, or tree boundaries, and run the task-owned actions against the captured
manifest and independent baseline. It must bind delivery to its own invocation
and preserve framework guard, runtime, timeout, and scoring authority.

Supplying the old mini module alone does not meet those requirements. CPU tests
with a small local command fixture cover rejection, argv, path isolation, and
process failure handling only. They do not qualify a mini provider, model,
GPU task, or optimization result.

```bash
python3 -m pytest -q tests/test_mini_swe_v2.py
```
