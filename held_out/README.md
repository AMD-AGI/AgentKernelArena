# Held-Out Evaluation

Evaluates whether agent-generated kernels generalize to unseen input shapes,
or are overfit to the test cases visible during development.

## Overview

1. **Generate** held-out shapes per task via an LLM agent (`generate_heldout.py`)
2. **Evaluate** a completed run against those shapes (`run_heldout_eval.py`)
3. **Compare** held-out vs original correctness and speedup to measure the
   "generalization gap"

The **code** (this directory) is checked into the repo. Newly generated test
data (`held_out_tests/`) is `.gitignore`'d so that your own held-out shapes
stay private.

## Released NeurIPS 2026 held-out shapes

For reproducibility, we release the exact held-out shapes used in the NeurIPS
2026 paper in [`neurips2026_shapes/`](neurips2026_shapes/) (one
`held_out_shapes.yaml` per task, 196 tasks). They were generated with Cursor
Agent and `claude-4.6-opus-high` using `generate_heldout.py`.

These shapes were generated **after** the agents had optimized the kernels, so
the agents evaluated in the paper never saw them. Now that they are public,
they are no longer hidden from future agents or models. To keep the held-out
evaluation meaningful, either:

- **Hide them during optimization:** make sure this directory is not
  reachable from the agent's environment (agents have shell access), and use
  the shapes only in the final held-out validation stage; or
- **Regenerate them:** produce a fresh set with `generate_heldout.py` (see
  below) and keep it private.

To reproduce the paper's held-out results on a completed run:

```bash
python held_out/run_heldout_eval.py \
    --run-dir <run_dir> \
    --heldout-dir held_out/neurips2026_shapes/ \
    --tasks-dir tasks/
```

## Scope

| Task type | Subdirectory | Injection target |
|-----------|-------------|------------------|
| triton2triton | vllm | `TEST_SHAPES` in `scripts/task_runner.py` |
| triton2triton | rocmbench | `@pytest.mark.parametrize` decorators in `test_*.py` (raw_replace) |
| hip2hip | gpumode | `get_inputs()` in both `pytorch_code_module/*.py` and `pytorch_code_functional/*_func.py` |
| torch2hip | gpumode | Same as hip2hip |

## Quick start

### 1. Generate held-out shapes

```bash
python held_out/generate_heldout.py \
    --tasks-dir tasks/ \
    --output-dir held_out_tests/ \
    --backend claude_code \
    --timeout 600
```

Use `--dry-run` to list tasks without launching agents.
Use `--tasks hip2hip/gpumode/SiLU` to generate for a single task.
Supported backends: `claude_code`, `codex`, `cursor` (same backends as the main arena agents).

### 2. Evaluate a run

```bash
python held_out/run_heldout_eval.py \
    --run-dir workspace_MI300_cursor/run_20260417_142419 \
    --heldout-dir held_out_tests/ \
    --tasks-dir tasks/
```

This creates `run_20260417_142419_heldout/` alongside the original run.
Each task workspace contains two subdirectories:

- `orig/` — the original (unoptimized) kernel restored from `tasks/`, with
  held-out shapes injected. Evaluated for **compilation, correctness, and
  baseline performance**.
- `opt/` — the agent's optimized kernel, with the same held-out shapes
  injected. Evaluated for **compilation, correctness, and optimized
  performance**.

Both kernels are evaluated on the same held-out shapes, producing a
**generalization quadrant** for each task:

| orig | opt | Status | Meaning |
|------|-----|--------|---------|
| ✓ | ✓ | `both_pass` | Normal — compare speedups |
| ✓ | ✗ | `opt_regression` | Optimization broke generalization |
| ✗ | ✗ | `both_fail` | Shape exceeds kernel design spec |
| ✗ | ✓ | `opt_improvement` | Agent improved robustness |

The key paper metric is **conditional correctness**: P(opt correct | orig
correct), which measures generalization by excluding shapes that are
inherently beyond the kernel's capability.

Output files:
- Per-task `heldout_task_result.yaml`
- Aggregate `heldout_summary.yaml` (with quadrant counts and conditional
  correctness rate)

## held_out_shapes.yaml format

Each task gets a YAML file storing replacement Python code for injection.

### triton2triton example

```yaml
task_type: triton2triton
num_original_shapes: 5
injections:
  - file: scripts/task_runner.py
    find_marker: "TEST_SHAPES"
    replacement_code: |
      TEST_SHAPES = [
          (64, 256, 128, True, False, True),
          ...
      ]
```

### triton2triton/rocmbench example

```yaml
task_type: triton2triton
num_original_shapes: 2
injections:
  - file: test_add_kernel.py
    find_marker: "raw_replace"
    old_code: |
      @pytest.mark.parametrize('SIZE,BLOCK_SIZE,dtype_str',
                               [(98432, 1024, dtype_str) for dtype_str in ['float16', 'float32']])
    replacement_code: |
      @pytest.mark.parametrize('SIZE,BLOCK_SIZE,dtype_str',
                               [(65536, 512, dtype_str) for dtype_str in ['float16', 'float32']])
  - file: test_add_kernel.py
    find_marker: "raw_replace"
    old_code: |
      @pytest.mark.parametrize('SIZE,BLOCK_SIZE_ARG,dtype_str',
                               [(98432, 1024, dtype_str) for dtype_str in ['float16', 'float32']] +
                               [(1048576, 2048, dtype_str) for dtype_str in ['float16', 'float32']])
    replacement_code: |
      @pytest.mark.parametrize('SIZE,BLOCK_SIZE_ARG,dtype_str',
                               [(65536, 512, dtype_str) for dtype_str in ['float16', 'float32']])
```

### hip2hip / torch2hip example

```yaml
task_type: hip2hip
num_original_shapes: 11
init_constraints:
  features: 4
injections:
  - file: pytorch_code_module/py_11754_layer_normalization.py
    find_marker: "def get_inputs"
    replacement_code: |
      def get_inputs():
          ...
  - file: pytorch_code_functional/py_11754_layer_normalization_func.py
    find_marker: "def get_inputs"
    replacement_code: |
      def get_inputs():
          ...  # identical to modular
```

The `init_constraints` field documents shape constraints from `get_init_inputs()`.

## How injection works

The eval script copies the agent's workspace and replaces test shapes via
text-based codegen replacement (`injection.py`):

- **TEST_SHAPES**: Regex finds `TEST_SHAPES = [`, bracket-balances to find
  the end, replaces the entire block
- **get_inputs**: Regex finds `def get_inputs(`, indentation-based detection
  finds the function end, replaces the entire function
- **raw_replace**: Exact string match on `old_code`, replaces with
  `replacement_code` (used for rocmbench `@pytest.mark.parametrize` decorators
  and any other ad-hoc patterns)

The agent's optimized kernel code is untouched — only the test harness shapes
change.
