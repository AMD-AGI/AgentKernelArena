---
myst:
    html_meta:
        "description": "Generate private held-out task shapes and evaluate completed AgentKernelArena runs for kernel generalization."
        "keywords": "AgentKernelArena, held-out evaluation, generalization, GPU kernel, private tests, unseen shapes"
---

# Evaluate kernels on held-out shapes

The optional `src.held_out` module evaluates whether an optimized kernel
generalizes to input shapes that were not visible during the original run.
This is a manually invoked post-run workflow; normal `docker-run` and
`docker-parallel-run` commands do not execute it automatically.

Generated shape configurations are written to `held_out_tests/`, which is
excluded from Git so private evaluation inputs are not committed.

## Supported task scopes

| Task type | Supported scope | Injection target |
| --- | --- | --- |
| `triton2triton` | `vllm` | `TEST_SHAPES` in `scripts/task_runner.py` |
| `triton2triton` | `rocmbench` | `@pytest.mark.parametrize` declarations |
| `hip2hip` | `gpumode` | `get_inputs()` in the module and functional references |
| `torch2hip` | `gpumode` | `get_inputs()` in the module and functional references |

## Enter the runtime

Generation and evaluation require the same agent, Python, ROCm, and GPU access
as a normal experiment. From the repository root, enter the Docker runtime:

```bash
make docker-shell
```

Run the remaining commands inside that shell from the mounted repository root.

## Generate held-out shapes

Generate configurations with one authenticated first-class agent CLI:

```bash
python3 -m src.held_out.generate_heldout \
  --tasks-dir tasks/ \
  --output-dir held_out_tests/ \
  --backend claude_code \
  --timeout 600
```

Use `--dry-run` to list supported tasks without launching an agent. Use
`--tasks hip2hip/gpumode/SiLU` to select a specific task. Supported backends are
`claude_code`, `codex`, and `cursor`.

## Evaluate a completed run

Pass a completed run directory and the generated held-out configurations:

```bash
python3 -m src.held_out.run_heldout_eval \
  --run-dir workspace_MI300_claude_code/run_<timestamp> \
  --heldout-dir held_out_tests/ \
  --tasks-dir tasks/
```

The evaluator creates a sibling directory ending in `_heldout`. For each task,
it restores and evaluates the original kernel under `orig/`, evaluates the
optimized kernel under `opt/`, and injects the same held-out shapes into both.

Results distinguish four outcomes:

| Original | Optimized | Status |
| --- | --- | --- |
| Pass | Pass | `both_pass` |
| Pass | Fail | `opt_regression` |
| Fail | Fail | `both_fail` |
| Fail | Pass | `opt_improvement` |

Each task writes `heldout_task_result.yaml`; the run-level output is
`heldout_summary.yaml`. The aggregate includes conditional correctness,
`P(optimized correct | original correct)`, and held-out speedup retention.

For injection formats and implementation details, see the
[module README](https://github.com/AMD-AGI/AgentKernelArena/blob/main/src/held_out/README.md).
