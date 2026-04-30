#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Generate held-out test shapes for AgentKernelArena tasks.

Launches a coding agent (claude_code, codex, or cursor) per task to inspect
the existing test infrastructure, understand shape constraints, and produce
a held_out_shapes.yaml with replacement code that can be injected by
run_heldout_eval.py.

Supported task types (scope):
  - triton2triton/vllm
  - triton2triton/rocmbench
  - hip2hip/gpumode
  - torch2hip/gpumode

Usage:
    python held_out/generate_heldout.py \
        --tasks-dir tasks/ \
        --output-dir held_out_tests/ \
        [--backend claude_code] \
        [--timeout 600]
"""
import argparse
import json
import logging
import shlex
import shutil
import subprocess
import sys
import threading
import yaml
from pathlib import Path
from typing import Any, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("generate_heldout")

SUPPORTED_SCOPES = {
    "triton2triton": ["vllm", "rocmbench"],
    "hip2hip": ["gpumode"],
    "torch2hip": ["gpumode"],
}

NUM_HELDOUT_SHAPES = 8

# Shared generalization methodology referenced by all prompt builders.
# Each held-out shape targets a specific generalization axis so the paper
# can report per-category pass rates (e.g., "92% on production shapes but
# only 64% on alignment-stress shapes").
GENERALIZATION_CATEGORIES = """
## Generalization Categories

You must generate exactly {n} held-out shapes across 6 categories (some
categories get 2 shapes for statistical robustness).  Tag each shape with
its category in an inline comment.

### Category 1: Edge-case / boundary  (1 shape)
Minimal or boundary sizes that stress partial-tile handling, single-element
batches, or degenerate dimensions.  Examples: batch=1, sequence_len=1, a
dimension equal to the kernel's BLOCK_SIZE, or a size just above/below a
tile boundary.

### Category 2: Scale-up  (1 shape)
Dimensions significantly LARGER than the original test shapes (at least
2-4× in the dominant dimension).  Tests whether the optimization scales:
tiling strategies, shared-memory reuse, grid sizing.
Keep within GPU memory (≤ 2 GB total allocation).

### Category 3: Scale-down  (1 shape)
Dimensions significantly SMALLER than the original test shapes (at least
2-4× in the dominant dimension).  Many optimized kernels over-provision
shared memory or assume a minimum occupancy; small inputs reveal that.

### Category 4: Alignment-stress  (2 shapes)
Sizes that are NOT multiples of common GPU tile sizes (32, 64, 128, 256).
Use primes or odd composites (e.g., 37, 131, 1019, 4003).  This tests
masking, bounds-checking, and padding logic.  This is the most common
failure mode for optimized kernels — generate 2 distinct shapes here.

### Category 5: Asymmetric / unusual aspect ratio  (1 shape)
Highly skewed dimensions — e.g., M=1 with N=65536, or a very tall-skinny
matrix (M>>N by 100×+).  Tests whether the kernel handles non-square or
extreme-ratio workloads that break assumptions about grid partitioning.

### Category 6: Production-realistic  (2 shapes)
Shapes drawn from real ML workloads.  Generate 2 distinct shapes:
  - For Triton/attention kernels: batch ∈ {{1,2,4,8}}, heads ∈ {{8,16,32,64}},
    seq_len ∈ {{128,512,2048,4096,8192}}, head_dim ∈ {{64,128}}
  - For HIP/element-wise kernels: NCHW with N ∈ {{1,2,4,8,16}},
    C ∈ {{64,128,256,512,1024}}, H=W ∈ {{7,14,28,56,112,224}}
  - For GEMM kernels: M,N,K from transformer FFN sizes
    (e.g., 4096×11008×4096, 2048×8192×2048)
  Pick the pattern that matches this kernel's semantics.

### Tagging

For EACH shape, add a short inline comment identifying its category, e.g.:
```python
(1, 64, 128, True, False),       # edge-case: batch=1
(32, 4096, 8192, True, True),    # scale-up: 4x seq_len
(2, 131, 64, False, True),       # alignment-stress: prime seq_len
(1, 65536, 64, True, False),     # asymmetric: M=1 decode-like
(8, 32, 2048, 128, True, False), # production: Llama-7B prefill
```
"""


def discover_tasks(tasks_dir: Path) -> List[Tuple[str, Path]]:
    """Return list of (task_id, task_dir) for all in-scope tasks."""
    results = []
    for task_type, subdir_filters in SUPPORTED_SCOPES.items():
        for subdir_filter in subdir_filters:
            base = tasks_dir / task_type / subdir_filter
            if not base.exists():
                continue
            for config_path in base.rglob("config.yaml"):
                task_dir = config_path.parent
                rel = task_dir.relative_to(tasks_dir)
                task_id = str(rel)
                results.append((task_id, task_dir))
    return sorted(results)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

HELDOUT_YAML_SCHEMA_TRITON = """
task_type: triton2triton
num_original_shapes: <count of original TEST_SHAPES entries>
injections:
  - file: scripts/task_runner.py
    find_marker: "TEST_SHAPES"
    replacement_code: |
      TEST_SHAPES = [
          <tuple1>,
          <tuple2>,
          <tuple3>,
          <tuple4>,
          <tuple5>,
      ]
"""

HELDOUT_YAML_SCHEMA_ROCMBENCH = """
task_type: triton2triton
num_original_shapes: <count of original parametrize entries for correctness>
injections:
  - file: <test_file.py>
    find_marker: "raw_replace"
    old_code: |
      <exact original @pytest.mark.parametrize decorator text for correctness test>
    replacement_code: |
      <new @pytest.mark.parametrize decorator with held-out shapes>
  - file: <test_file.py>
    find_marker: "raw_replace"
    old_code: |
      <exact original @pytest.mark.parametrize decorator text for performance test>
    replacement_code: |
      <new @pytest.mark.parametrize decorator with held-out shapes>
"""

HELDOUT_YAML_SCHEMA_HIP = """
task_type: <hip2hip or torch2hip>
num_original_shapes: <count of original configs in get_inputs>
init_constraints:
  <key>: <value>   # shape-relevant params from get_init_inputs only
injections:
  - file: <pytorch_code_module/py_XXXX_Name.py>
    find_marker: "def get_inputs"
    replacement_code: |
      def get_inputs():
          configs = [
              <config1>,
              ...
          ]
          for shape in configs:
              shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
              <tensor creation matching the original pattern>
              yield [<tensors>]
  - file: <pytorch_code_functional/py_XXXX_Name_func.py>
    find_marker: "def get_inputs"
    replacement_code: |
      def get_inputs():
          <IDENTICAL code to the modular version above>
"""


def build_prompt(task_id: str, output_path: str) -> str:
    task_type = task_id.split("/")[0]
    sub_scope = task_id.split("/")[1] if "/" in task_id else ""

    if task_type == "triton2triton" and sub_scope == "rocmbench":
        return _build_prompt_rocmbench(task_id, output_path)
    elif task_type == "triton2triton":
        return _build_prompt_triton(task_id, output_path)
    else:
        return _build_prompt_hip(task_id, task_type, output_path)


def _build_prompt_triton(task_id: str, output_path: str) -> str:
    cats = GENERALIZATION_CATEGORIES.format(n=NUM_HELDOUT_SHAPES)
    return f"""# Held-Out Test Shape Generator

You are generating held-out test shapes for evaluating whether an
agent-optimized GPU kernel generalizes to unseen inputs.  The results will
be reported in a scientific paper, so the shapes must be methodologically
motivated, not arbitrary.

## Task
- Task ID: `{task_id}`
- Task type: triton2triton (Triton kernel optimization)

## Your Mission

1. **Read** `scripts/task_runner.py` in this workspace directory.
2. **Also read** the kernel source file(s) in `source/` to understand what
   the kernel computes, what its BLOCK_SIZE / tile parameters are, and
   whether there are algorithmic branches (e.g., causal vs non-causal,
   different code paths for head_dim <= 64 vs > 64).
3. **Understand** the existing `TEST_SHAPES` list:
   - What does each element of the tuple mean?
   - What hard constraints exist (alignment, minimum sizes, dtype, divisibility)?
   - What is the range of each dimension in the original shapes?
   - How many shapes are there currently?
4. **Generate exactly {NUM_HELDOUT_SHAPES} NEW test shape tuples** across
   the generalization categories below (some categories require 2 shapes).
5. **Write** the file `{output_path}` with the exact YAML format below.

{cats}

## Output Format

Write `{output_path}` as a valid YAML file with this structure:
```yaml
{HELDOUT_YAML_SCHEMA_TRITON}
```

The `replacement_code` must be valid Python that can directly replace the
existing TEST_SHAPES definition in task_runner.py.

## Rules
- Do NOT modify any existing files — only create the output YAML
- Do NOT run any commands — this is a read-and-write task only
- The output MUST be valid YAML
- Each shape must satisfy ALL hard constraints you identified
- Each shape MUST be tagged with its generalization category in a comment
- Ensure no generated shape is identical to any original TEST_SHAPES entry
"""


def _build_prompt_rocmbench(task_id: str, output_path: str) -> str:
    cats = GENERALIZATION_CATEGORIES.format(n=NUM_HELDOUT_SHAPES)
    return f"""# Held-Out Test Shape Generator (rocmbench)

You are generating held-out test shapes for evaluating whether an
agent-optimized GPU kernel generalizes to unseen inputs.  The results will
be reported in a scientific paper, so the shapes must be methodologically
motivated, not arbitrary.

## Task
- Task ID: `{task_id}`
- Task type: triton2triton / rocmbench

## Your Mission

1. **Read** the main Python test file in this workspace directory
   (e.g. `test_*.py`, `softmax.py`, `gemm.py`, etc.)
2. **Read the kernel code** to understand what it computes, its tile/block
   parameters, and whether there are algorithmic branches that trigger at
   certain sizes.
3. **Find** all `@pytest.mark.parametrize(...)` decorators on test functions:
   - The **correctness test** (any test NOT named `test_performance` or
     `test_save_performance_results`)
   - The **performance test** (`test_performance`)
4. **Understand** the parametrize arguments:
   - What do the parameters represent? (sizes, block sizes, dtypes, etc.)
   - What hard constraints exist? (block sizes must be powers of 2, etc.)
   - What is the range of each dimension in the original shapes?
   - How many test configurations are there currently?
5. **Generate exactly {NUM_HELDOUT_SHAPES} NEW parametrize entries** for
   BOTH the correctness test and the performance test, across the
   generalization categories below (some categories require 2 shapes).
6. **Write** the file `{output_path}` with the exact YAML format below.

{cats}

## CRITICAL: Exact text matching

For `raw_replace` injections, the `old_code` field must be the EXACT text of the
original `@pytest.mark.parametrize(...)` decorator (including newlines, spaces, and
the closing parenthesis). Copy it character-for-character from the source file.
The `replacement_code` must be a valid `@pytest.mark.parametrize(...)` decorator
that can replace the old one.

You must create separate injection entries for EACH parametrize decorator
(one for the correctness test, one for the performance test).

## Output Format

Write `{output_path}` as a valid YAML file with this structure:
```yaml
{HELDOUT_YAML_SCHEMA_ROCMBENCH}
```

## Rules
- Do NOT modify any existing files — only create the output YAML
- Do NOT run any commands — this is a read-and-write task only
- The output MUST be valid YAML
- The `old_code` must be an EXACT copy of the original decorator text
- Each shape must satisfy ALL hard constraints you identified
- Each shape MUST be tagged with its generalization category in a comment
- Ensure no generated shape is identical to any original parametrize entry
- If there are multiple correctness test functions with parametrize, include injections for each
"""


def _build_prompt_hip(task_id: str, task_type: str, output_path: str) -> str:
    cats = GENERALIZATION_CATEGORIES.format(n=NUM_HELDOUT_SHAPES)
    return f"""# Held-Out Test Shape Generator

You are generating held-out test shapes for evaluating whether an
agent-optimized GPU kernel generalizes to unseen inputs.  The results will
be reported in a scientific paper, so the shapes must be methodologically
motivated, not arbitrary.

## Task
- Task ID: `{task_id}`
- Task type: {task_type} (HIP kernel optimization)

## Your Mission

1. **Read** both Python files in this workspace:
   - `pytorch_code_module/py_*.py` (the modular version)
   - `pytorch_code_functional/py_*_func.py` (the functional version)
2. **Also read** the HIP kernel source in `hip/` to understand what the
   kernel computes, its launch configuration (block/grid dims), and whether
   there are algorithmic branches that trigger at certain input sizes.
3. **Understand** the existing `get_inputs()` function:
   - What tensors does it yield? (single tensor, pair, triple?)
   - What shapes and dtypes are used?
   - How many test configurations exist?
   - What is the range of each dimension across the original configs?
4. **Read `get_init_inputs()`** and identify shape constraints:
   - If it returns `[[], {{'features': 4}}]`, the last dimension of inputs MUST be 4
   - If it returns `[[], {{'input_channel_num': 4}}]`, the channel dimension (dim 1 in NCHW) MUST be 4
   - If it returns `[[], {{'dim_model': 4}}]`, the model dimension MUST be 4
   - If it returns `[[], {{'channel': 256}}]`, channel dimensions must be <= 256
   - If it returns `[[], {{'max_sequence_length': N}}]`, sequence lengths must be <= N
   - If it returns `[[], {{'heads': H, 'd_model': D}}]`, then D % H must equal 0
   - Params like `prob_dropout`, `temp_factor` do NOT constrain shapes
   - If it returns `[[], {{}}]`, there are no shape constraints beyond what the kernel needs
5. **Generate exactly {NUM_HELDOUT_SHAPES} NEW test configurations** across
   the generalization categories below (some categories require 2 shapes).
   - RESPECT all get_init_inputs() constraints
   - Use the SAME tensor structure as the original (same number of tensors, same dtypes)
   - Keep total allocation ≤ 2 GB to avoid OOM
6. **Write** the file `{output_path}` with the exact YAML format below.

{cats}

## CRITICAL: Dual-file injection

The replacement `get_inputs()` function must be **IDENTICAL** in both the
modular and functional injection entries. Find the exact filenames from the
workspace (the `pytorch_code_module/` and `pytorch_code_functional/` directories).

## Output Format

Write `{output_path}` as a valid YAML file with this structure:
```yaml
{HELDOUT_YAML_SCHEMA_HIP}
```

The `replacement_code` must be valid Python that can directly replace the
existing get_inputs() function. Keep the same coding style (configs list,
for loop, yield pattern).

## Rules
- Do NOT modify any existing files — only create the output YAML
- Do NOT run any commands — this is a read-and-write task only
- The output MUST be valid YAML
- The `init_constraints` field should only include params that constrain tensor shapes
- Each shape must satisfy ALL constraints you identified (both hard and init_constraints)
- Each shape MUST be tagged with its generalization category in a comment
- Ensure no generated shape is identical to any original get_inputs() entry
- The modular and functional replacement_code MUST be byte-for-byte identical
"""


# ---------------------------------------------------------------------------
# Agent launchers (adapted from task_validator)
# ---------------------------------------------------------------------------

def _read_stream(stream, output_list, prefix, log_func):
    try:
        for line in iter(stream.readline, ''):
            if not line:
                break
            raw = line.rstrip()
            if raw.strip():
                output_list.append(raw)
                try:
                    data = json.loads(raw)
                    et = data.get("type", "")
                    if et == "stream_event":
                        ev = data.get("event", {})
                        if ev.get("type") == "content_block_delta":
                            continue
                    log_func(f"{prefix} {raw[:200]}")
                except (json.JSONDecodeError, AttributeError):
                    log_func(f"{prefix} {raw[:200]}")
    finally:
        stream.close()


def _launch_claude_code(prompt: str, workspace: str, timeout: int, log: logging.Logger,
                        model: Optional[str] = None) -> str:
    agent = "claude"
    opts = (
        "--print "
        "--verbose "
        "--output-format stream-json "
        "--permission-mode bypassPermissions "
        "--dangerously-skip-permissions"
    )
    if not shutil.which(agent):
        raise RuntimeError(f"'{agent}' CLI not found in PATH")

    if model:
        opts += f" --model {shlex.quote(model)}"
    cmd = f"{agent} {opts} {shlex.quote(prompt)}"
    log.info(f"Launching claude_code (model={model or 'default'}): {cmd[:200]}...")

    proc = subprocess.Popen(
        cmd, shell=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=workspace, bufsize=1,
    )
    if proc.stdin:
        proc.stdin.close()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    t1 = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_lines, "[GEN]", log.info), daemon=True)
    t2 = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_lines, "[GEN-ERR]", log.warning), daemon=True)
    t1.start(); t2.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning(f"Agent timed out after {timeout}s; terminating")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    t1.join(timeout=5); t2.join(timeout=5)
    log.info(f"Agent exited with code {proc.returncode}")
    return "\n".join(stdout_lines)


def _launch_codex(prompt: str, workspace: str, timeout: int, log: logging.Logger,
                  model: Optional[str] = None) -> str:
    agent = "codex"
    if not shutil.which(agent):
        raise RuntimeError(f"'{agent}' CLI not found in PATH")

    cmd = [
        agent, "exec", "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--cd", workspace,
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)
    log.info(f"Launching codex (model={model or 'default'}): {' '.join(shlex.quote(p) for p in cmd[:8])}...")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=workspace, bufsize=1,
    )
    if proc.stdin:
        proc.stdin.close()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    t1 = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_lines, "[GEN]", log.info), daemon=True)
    t2 = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_lines, "[GEN-ERR]", log.warning), daemon=True)
    t1.start(); t2.start()

    try:
        proc.wait(timeout=timeout if timeout > 0 else None)
    except subprocess.TimeoutExpired:
        log.warning(f"Agent timed out after {timeout}s; terminating")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    t1.join(timeout=5); t2.join(timeout=5)
    log.info(f"Agent exited with code {proc.returncode}")
    return "\n".join(stdout_lines)


def _launch_cursor(prompt: str, workspace: str, timeout: int, log: logging.Logger,
                   model: Optional[str] = None) -> str:
    agent = "cursor-agent"
    if not shutil.which(agent):
        raise RuntimeError(
            f"'{agent}' CLI not found in PATH. "
            "Ensure cursor-agent is installed and available."
        )

    opts = "--force --print --output-format stream-json --stream-partial-output"
    if model:
        opts += f" --model {shlex.quote(model)}"
    cmd = f"{agent} {opts} {shlex.quote(prompt)}"
    log.info(f"Launching cursor (model={model or 'default'}): {cmd[:200]}...")

    proc = subprocess.Popen(
        cmd, shell=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=workspace, bufsize=1,
    )
    if proc.stdin:
        proc.stdin.close()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    t1 = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_lines, "[GEN]", log.info), daemon=True)
    t2 = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_lines, "[GEN-ERR]", log.warning), daemon=True)
    t1.start(); t2.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning(f"Agent timed out after {timeout}s; terminating")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    t1.join(timeout=5); t2.join(timeout=5)
    log.info(f"Agent exited with code {proc.returncode}")
    return "\n".join(stdout_lines)


BACKENDS = {
    "claude_code": _launch_claude_code,
    "codex": _launch_codex,
    "cursor": _launch_cursor,
}


# ---------------------------------------------------------------------------
# Per-task generation
# ---------------------------------------------------------------------------

def generate_for_task(
    task_id: str,
    task_dir: Path,
    output_dir: Path,
    backend: str,
    timeout: int,
    model: Optional[str] = None,
) -> bool:
    """Generate held_out_shapes.yaml for a single task. Returns True on success."""
    log = logging.getLogger(f"gen.{task_id}")

    out_path = output_dir / task_id / "held_out_shapes.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(task_id, str(out_path))

    launcher = BACKENDS.get(backend)
    if not launcher:
        log.error(f"Unknown backend: {backend}")
        return False

    try:
        agent_output = launcher(prompt, str(task_dir), timeout, log, model=model)
    except Exception as e:
        log.error(f"Agent launch failed: {e}")
        return False

    # Save agent log alongside the YAML output
    log_path = out_path.parent / "agent_log.txt"
    log_path.write_text(
        f"# Task: {task_id}\n"
        f"# Backend: {backend}\n"
        f"# Model: {model or 'default'}\n"
        f"# Timeout: {timeout}s\n"
        f"{'=' * 80}\n"
        f"{agent_output}\n"
    )
    log.info(f"Agent log saved -> {log_path}")

    if out_path.exists():
        try:
            config = yaml.safe_load(out_path.read_text())
            if not config or "injections" not in config:
                log.error("Generated YAML missing 'injections' key")
                return False
            log.info(f"Generated held-out config -> {out_path}")
            return True
        except yaml.YAMLError as e:
            log.error(f"Generated file is not valid YAML: {e}")
            return False
    else:
        log.error(f"Agent did not create {out_path}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate held-out test shapes via coding agent")
    parser.add_argument("--tasks-dir", default="tasks", help="Root tasks directory")
    parser.add_argument("--output-dir", default="held_out_tests", help="Output directory for held-out configs")
    parser.add_argument("--backend", default="claude_code", choices=list(BACKENDS.keys()),
                        help="Agent backend to use (default: claude_code)")
    parser.add_argument("--model", default=None,
                        help="Model to use for the agent (e.g. claude-4.6-opus-high, o3). "
                             "If not set, uses the agent's default model.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-task agent timeout in seconds")
    parser.add_argument("--tasks", nargs="*", default=None, help="Specific task IDs to generate for")
    parser.add_argument("--dry-run", action="store_true", help="Discover tasks but don't launch agents")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    tasks_dir = Path(args.tasks_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    all_tasks = discover_tasks(tasks_dir)
    logger.info(f"Discovered {len(all_tasks)} in-scope tasks")

    if args.tasks:
        all_tasks = [(tid, td) for tid, td in all_tasks if tid in args.tasks]
        logger.info(f"Filtered to {len(all_tasks)} requested tasks")

    if args.dry_run:
        for task_id, _ in all_tasks:
            print(f"  {task_id}")
        return

    success = 0
    failed = 0

    for task_id, task_dir in all_tasks:
        try:
            if generate_for_task(task_id, task_dir, output_dir, args.backend, args.timeout, model=args.model):
                success += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"[{task_id}] Unexpected error: {e}")
            failed += 1

    logger.info(f"Done. {success} succeeded, {failed} failed out of {len(all_tasks)} tasks.")


if __name__ == "__main__":
    main()
