# AgentKernelArena held-out evaluation module

`src.held_out` evaluates whether agent-generated kernels generalize to unseen
input shapes or are overfit to the test cases visible during development. It
is an optional, manually invoked post-run workflow rather than part of a normal
AgentKernelArena run.

## Overview

1. **Generate** held-out shapes per task via an LLM agent (`generate_heldout.py`)
2. **Evaluate** a completed run against those shapes (`run_heldout_eval.py`)
3. **Compare** held-out vs original correctness and speedup to measure the
   "generalization gap"

The **code** (this directory) is checked into the repo. The **generated test
data** (`held_out_tests/`) is `.gitignore`'d so that held-out shapes remain
private for paper evaluation. Others can regenerate their own sets.

## Scope

Every schema-v2 task can use the same command/result and manifest protocol.
The generator reads its declared runner, workloads, instructions and edit
boundaries. It runs the authoring agent in a disposable task copy and preserves
existing held-out data. Generated injections need review: new shapes must stay
inside the operator's documented domain and retain its reference and timing
rules. This is separate evaluation data, not another task definition file.

The following legacy helpers remain for older configurations:

| Task type | Subdirectory | Injection target |
|-----------|-------------|------------------|
| triton2triton | vllm | `TEST_SHAPES` in `scripts/task_runner.py` |
| triton2triton | rocmbench | `@pytest.mark.parametrize` decorators in `test_*.py` (raw_replace) |
| hip2hip | gpumode | `get_inputs()` in both `pytorch_code_module/*.py` and `pytorch_code_functional/*_func.py` |
| torch2hip | gpumode | Same as hip2hip |

## Quick start

Held-out generation and evaluation use the same Python/ROCm dependencies and
GPU access as normal experiments. From the repository root, enter the supported
Docker runtime and run the remaining commands inside that shell:

```bash
make docker-shell
```

### 1. Generate held-out shapes

```bash
python3 -m src.held_out.generate_heldout \
    --tasks-dir tasks/ \
    --output-dir held_out_tests/ \
    --backend claude_code \
    --timeout 600
```

Use `--dry-run` to list tasks without launching agents.
Use `--tasks hip2hip/gpumode/SiLU` to generate for a single task.
Supported backends: `claude_code`, `codex`, and `cursor` (the three first-class
host-CLI integrations).

### 2. Evaluate a run

```bash
python3 -m src.held_out.run_heldout_eval \
    --run-dir workspace_MI300_cursor/run_20260417_142419 \
    --heldout-dir held_out_tests/ \
    --tasks-dir tasks/
```

This creates `run_20260417_142419_heldout/` alongside the original run.
Each task workspace contains two subdirectories:

- `orig/` — the original materialized task from the run's frozen v2 baseline, with
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
| ✗ | ✗ | `both_fail` | Both implementations failed; inspect the case and operator domain |
| ✗ | ✓ | `opt_improvement` | Agent improved robustness |

V2 evaluation requires an accepted original candidate, intact framework
completion/source evidence, and the same captured GPU/runtime identity. It does
not reconstruct the baseline from today's repository checkout. Old runs without
v2 provenance retain the legacy restoration path. Existing output workspaces
are never replaced by a v2 rerun; choose a new output suffix.

The task's `validate-task` action creates the new manifest in the original
snapshot. Baseline and candidate then use the same public compile, correctness,
and performance actions as normal evaluation. Each new performance input must
be absent from the original manifest; changing only its case ID is rejected.
Changing candidate code/imports through an injection, source changes during an
action, missing case coverage, and corrupt evidence produce `evaluation_error`,
not a generalization quadrant or speedup. Per-action evidence is retained under
`actions/` with the manifest, injection specification and runtime identity.

Only numerically correct baseline/candidate pairs with identical case identities
and benchmark methods receive a held-out speedup. A baseline allowed as
`diagnostic` in the original experiment is still reported as incorrect if it
fails the held-out numerical rule; it is not relabeled as a passing reference.
This optional workflow does not run sanitizer tools or claim their coverage.

The key paper metric is **conditional correctness**: P(opt correct | orig
correct), which measures generalization by excluding shapes that are
inherently beyond the kernel's capability.

Output files:
- Per-task `heldout_task_result.yaml`
- Aggregate `heldout_summary.yaml` (with quadrant counts and conditional
  correctness rate)

## held_out_shapes.yaml format

Each task gets a YAML file storing replacement Python code for injection.
For v2, raw replacements can also update task-owned JSON workload data. If a
task maintains its case manifest separately from the case generator, update
both. The shared executor rejects stale manifests instead of silently measuring
only an intersection of cases.

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
