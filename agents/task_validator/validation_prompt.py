# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import yaml
from pathlib import Path


VALIDATION_REPORT_SCHEMA = """
# Validation Report Schema
# Write this YAML file as `validation_report.yaml` in the workspace directory.

task_name: ""                          # Full task path (e.g., "hip2hip/rmsnorm")
validation_timestamp: ""               # ISO 8601 timestamp of when validation ran
overall_status: ""                     # PASS | FAIL | WARN

checks:
  config_schema:
    status: ""                         # PASS | FAIL
    details: ""                        # Describe what was checked and what was found

  source_files_exist:
    status: ""                         # PASS | FAIL
    details: ""                        # List which files exist or are missing

  target_symbols_found:
    status: ""                         # PASS | FAIL
    details: ""                        # For each target kernel function, report if found and where

  compilation:
    status: ""                         # PASS | FAIL | TIMEOUT | SKIP
    exit_code: null                    # Integer exit code, or null if not run
    duration_seconds: null             # How long the command took
    stdout_snippet: ""                 # First ~500 chars of stdout
    stderr_snippet: ""                 # First ~500 chars of stderr
    report_file_valid: null            # true/false - whether build/compile_report.json exists and has "status": "ok"

  correctness:
    status: ""                         # PASS | FAIL | TIMEOUT | SKIP
    exit_code: null
    duration_seconds: null
    stdout_snippet: ""
    stderr_snippet: ""
    report_file_valid: null            # true/false - whether build/correctness_report.json exists and looks valid
    analysis: ""                       # Brief analysis of what correctness check actually does

  performance:
    status: ""                         # PASS | WARN | FAIL | TIMEOUT | SKIP
    exit_code: null
    duration_seconds: null
    stdout_snippet: ""
    stderr_snippet: ""
    report_file_valid: null
    analysis: ""                       # Include performance methodology review (warmup / measured iters / averaging)

  correctness_implementation_review:
    status: ""                         # PASS | WARN | FAIL
    details: ""                        # Describe what the correctness check does, whether it's a real check
    is_trivially_passing: null         # true if correctness always passes regardless of output

  self_contained:
    status: ""                         # PASS | FAIL
    details: ""                        # Describe any external dependencies found
    missing_files: []                  # List of missing headers, imports, or external paths referenced

  gpu_hang_check:
    status: ""                         # PASS | FAIL | WARN
    details: ""                        # Report if any command timed out or appeared to hang

  result_template_compatibility:
    status: ""                         # PASS | FAIL
    details: ""                        # Whether the task produces output compatible with task_result_template.yaml
    template_name: ""                  # Which template it uses

summary: |
  One-paragraph summary of validation results.
  Include: total checks passed/failed/warned, key issues found.
"""


def build_validation_prompt(task_config_dir: str, workspace: str, eval_config: dict) -> str:
    """
    Build a validation-focused prompt for the task validator agent.

    This prompt instructs the agent to perform a series of checks on the task
    and produce a structured validation_report.yaml.

    Args:
        task_config_dir: Path to the task's config.yaml
        workspace: Path to the duplicated workspace directory
        eval_config: Global evaluation config

    Returns:
        str: Complete validation prompt
    """
    # Load task config
    task_config_path = Path(task_config_dir)
    with open(task_config_path, 'r') as f:
        task_config = yaml.safe_load(f)

    task_config_content = task_config_path.read_text()

    # Extract key fields for context
    task_type = task_config.get('task_type', 'unknown')
    source_files = task_config.get('source_file_path', [])
    target_kernels = task_config.get('target_kernel_functions', [])
    compile_cmds = task_config.get('compile_command', [])
    correctness_cmds = task_config.get('correctness_command', [])
    performance_cmds = task_config.get('performance_command', [])
    python_path = (
        eval_config.get('agent', {}).get('python_path')
        or os.environ.get('AGENT_KERNEL_ARENA_PYTHON')
        or sys.executable
    )
    compile_timeout = eval_config.get('agent', {}).get('compile_timeout', 300)
    correctness_timeout = eval_config.get('agent', {}).get('correctness_timeout', 300)
    performance_timeout = eval_config.get('agent', {}).get('performance_timeout', 300)

    prompt = f"""# Task Validation Agent

You are a **task validator**, not an optimizer. Your job is to validate that a GPU kernel optimization task is correctly configured, self-contained, and functional.

## Workspace
Your working directory is: `{workspace}`

## Task Configuration
The task's files have been COPIED into your working directory, so the config is at
`config.yaml` (i.e. `{workspace}/config.yaml`) — read it there. NOTE: the original
repository path `{task_config_dir}` does NOT exist inside your workspace; do not try
to access it. Likewise, all source/eval files referenced below are workspace-local
(e.g. `hip/...`, `pytorch_code_module/...`, `eval_tools/...`), not under `tasks/`.

Its contents are:
```yaml
{task_config_content}
```

## Your Mission

Perform the following 10 validation checks IN ORDER. For each check, record the result. After all checks are complete, write a `validation_report.yaml` file to the workspace directory.

Use this Python interpreter when needed: `{python_path}`

## CRITICAL: Pre-Validation Cleanup

Before running ANY validation checks, clean up stale JIT compilation caches that may contain leftover lock files from previously killed processes. Stale lock files will cause compilation to hang indefinitely.

Run the following cleanup command FIRST:
```bash
find ~/.cache/torch_extensions/ -name "lock" -delete 2>/dev/null; echo "JIT cache lock files cleaned"
```

## CRITICAL: No-Retry Policy

**Do NOT retry any command that fails or times out.** Run each compile/correctness/performance command EXACTLY ONCE. If it fails (non-zero exit code) or times out (exit code 124), immediately record the result and move on to the next check. Do not attempt alternative approaches, do not re-run with different parameters, and do not try to debug or fix the issue. Your job is to REPORT, not to FIX.

### Check 1: Config Schema Validation
Verify that config.yaml contains all required fields:
- `source_file_path` (list of strings)
- `target_kernel_functions` (list of strings)
- `compile_command` (list of strings)
- `correctness_command` (list of strings)
- `task_type` (string, one of: hip2hip, cuda2hip, triton2triton, triton2flydsl, torch2hip, torch2flydsl, instruction2triton, flydsl2flydsl, repository, image_kernel)
Also check that optional fields (`performance_command`, `prompt`, `platform_support`) are well-formed if present.
`platform_support`, when present, is a mapping with optional `status` (`active` or `skip`),
optional `required_arch` (string such as `gfx942`), and optional `skip_reason` (string).

**IMPORTANT — `task_type: repository` schema differs.** Repository tasks clone a full upstream
project and drive it through `scripts/task_runner.py` instead of shipping an isolated kernel file.
For `task_type: repository`:
- `repo_url` (string) is REQUIRED; `repository_language` (string) is expected.
- `source_file_path` and `target_kernel_functions` are OPTIONAL (they are hints into the cloned
  tree, not always present). Do NOT FAIL this check merely because they are absent for a repository
  task. Optional `post_clone_install` / `post_clone_install_mode` may also be present.
For `task_type: image_kernel`:
- `image_repo_path` (string) is REQUIRED; `repository_language` (string) is expected.
- `source_file_path` and `target_kernel_functions` are required as for other kernel tasks.
- Optional `image_repo_exclude` must be a list of safe relative paths when present. These paths
  name disposable build/cache content to omit while seeding the repository from the task image.
Status: PASS if all required fields for the task_type exist and have correct types, FAIL otherwise.

### Check 2: Source Files Exist
For each file listed in `source_file_path`: {source_files}
Check if the file exists in the workspace directory `{workspace}`.
Look for the file directly and also under common subdirectories (source/, src/, scripts/).
For `task_type: repository` or `task_type: image_kernel`, the source files live inside the upstream repository tree,
whose top-level prefix may differ from the configured path (e.g. a repo `aiter` clones such that
`aiter/ops/triton/x.py` actually resolves to `aiter/aiter/ops/triton/x.py`). Search RECURSIVELY
under the workspace and match by the trailing path / basename; PASS if a matching file is found
anywhere in the tree. If `source_file_path` is absent for a repository task, mark this check SKIP.
Status: PASS if all source files are found, FAIL if any are missing (SKIP if not declared for a repository task).

### Check 3: Target Symbols Found
For each function in `target_kernel_functions`: {target_kernels}
Search the source files for the function name (as a symbol definition, not just a string mention).
For CUDA/HIP: look for `__global__ void <name>` or similar kernel declarations.
For Triton: look for `@triton.jit` decorated functions with the name.
For Python: look for `def <name>`.
Report the file and line number where each symbol is found.
For `task_type: repository`, if `target_kernel_functions` is absent from config.yaml, mark this
check SKIP (it is an optional hint for repository tasks, not a required declaration).
Status: PASS if all target symbols found, FAIL if any are missing (SKIP if not declared for a repository task).

### Check 4: Compilation
Run the compile command(s) from the workspace directory:
```
{chr(10).join(compile_cmds) if compile_cmds else 'No compile command specified'}
```
Use a timeout of {compile_timeout} seconds per command. Run the command EXACTLY ONCE — do NOT retry on failure or timeout.
Capture stdout, stderr, and exit code.
Also check if `build/compile_report.json` is generated and contains a valid status.
If exit code is non-zero but `eval_result.yaml` clearly records `compiled: true`, treat compilation as PASS and document the wrapper/command inconsistency in details.

**IMPORTANT — generation-type tasks with an empty placeholder kernel (do NOT false-FAIL).**
Some task types — notably `task_type: torch2hip` — intentionally ship the target kernel file EMPTY (0 bytes) and provide NO reference kernel (`*_ref.hip`). The empty file is a placeholder that the *optimization* agent is meant to fill in by generating the kernel from the provided PyTorch reference; `source_file_path` and `target_file_path` typically point to the same empty file. Before judging this check, inspect the size of the file(s) the compile command builds. If that file is empty / 0 bytes, then compilation cannot and is NOT expected to succeed on the as-shipped (unfilled) task — this is BY DESIGN, not a task defect (it would otherwise FAIL with "empty source file" or "missing PyInit_* export"). In that case:
- Set `checks.compilation.status` to `SKIP` (NOT FAIL), and explain in `details` that the target is an intentionally-empty generation placeholder (e.g. a torch2hip task awaiting agent-generated kernel code).
- Downstream `correctness` and `performance` are also `SKIP` for the same reason.
Only mark compilation FAIL when a NON-empty kernel genuinely fails to compile (a real defect).

Status: PASS if compilation evidence is successful (exit code 0 OR compile_report status ok OR eval_result compiled=true); SKIP if the target kernel file is an intentionally-empty generation placeholder (see above); FAIL if a non-empty kernel fails to compile; TIMEOUT if exceeded {compile_timeout}s.

### Check 5: Correctness
Run the correctness command(s) from the workspace directory:
```
{chr(10).join(correctness_cmds) if correctness_cmds else 'No correctness command specified'}
```
Use a timeout of {correctness_timeout} seconds per command. Run the command EXACTLY ONCE — do NOT retry on failure or timeout.
Capture stdout, stderr, and exit code.
Check if `build/correctness_report.json` is generated.
If exit code is non-zero but `eval_result.yaml` clearly records `correctness: true`, treat correctness as PASS and explain the inconsistency.

**IMPORTANT — `torch2flydsl` starter-stub contract (task-package validation only).** A shipped
`torch2flydsl` task may intentionally define a declared top-level target whose body contains only an
optional docstring / `pass` and a direct, unconditional `raise NotImplementedError(...)`. This is a
non-empty generation starter, not an optimized implementation. If the correctness harness invokes that
target, catches that specific `NotImplementedError`, clearly reports the target as unimplemented, and
still successfully validates the PyTorch/model reference against an independent oracle such as AITER,
set correctness to `SKIP` and explain that the independent oracle passed. This allowance applies only to
the as-shipped task validator:

- Do not treat a conditional `NotImplementedError` inside an otherwise implemented target as a starter stub.
- Missing target symbols/files, `AttributeError`, `ImportError`, `RuntimeError`, and every exception other
  than the explicit starter `NotImplementedError` are real failures and MUST NOT be converted to `SKIP`,
  even if a harness prints “SKIP” or exits zero.
- If the independent reference/oracle check fails, correctness is `FAIL`, not `SKIP`.
- Performance remains `SKIP` for an accepted starter because there is no implemented target to score.
- The centralized optimization evaluator performs a static guard before correctness and rejects an agent
  submission that leaves any declared target as this starter stub. A validator `SKIP` never makes an
  unimplemented optimization submission eligible for scoring.

Status: PASS if correctness evidence is successful (exit code 0 OR correctness_report status ok OR eval_result correctness=true), FAIL otherwise, TIMEOUT if exceeded {correctness_timeout}s, SKIP if compilation failed/was skipped (e.g. empty generation-placeholder kernel) OR the exact `torch2flydsl` starter-stub contract above is satisfied.

### Check 6: Performance
Run the performance command(s) from the workspace directory (if any):
```
{chr(10).join(performance_cmds) if performance_cmds else 'No performance command specified'}
```
Use a timeout of {performance_timeout} seconds per command. Run the command EXACTLY ONCE — do NOT retry on failure or timeout.
Capture stdout, stderr, and exit code.
If timing fields are present in `eval_result.yaml` (`speedup`, `ori_time`, `opt_time`) and are non-null, treat performance as PASS even if wrapper exit code is inconsistent.
In addition, review the performance measurement implementation (typically `scripts/task_runner.py` or task-specific perf scripts) and determine whether it uses the recommended methodology:
- warmup iterations = 10
- measured iterations = 100
- reported runtime is an average across the measured iterations (and speedup is derived from those average runtimes)

Two timing implementations BOTH satisfy this methodology — treat either as PASS, do NOT WARN:
1. **CUDA-event timing** — 10 warmup, then 100 measured iterations each timed with a CUDA-event
   pair, averaged (e.g. hip2hip `eval_tools/cal_kernel_perf.py::cal_hip_latency`, or a
   `cuda_event_fallback`/`cpu_timer_fallback` path).
2. **CUDA-graph timing** — 10 warmup, then 100 timed graph-replay samples averaged, where each
   replay runs an adaptively-chosen `n_repeat` kernels to amortize launch overhead
   (`_benchmark_cuda_graph_or_events` / `_measure_times`; emits `benchmark_method: cuda_graph`
   and `benchmark_samples: 100`). This is an ACCEPTED EQUIVALENT of "100 measured averaged" —
   do NOT WARN merely because it uses CUDA graphs, an adaptive `n_repeat`, or records
   `benchmark_method`. (Note: the `n_retries` parameter is legacy/unused; the sample count is
   driven by `repetition`/`benchmark_samples`.)

If performance execution succeeds but the methodology is genuinely different (e.g. warmup != 10, a
sample count clearly != 100, or no averaging), or it cannot be verified from code, record a clear
note in `checks.performance.analysis` and set `checks.performance.status` to `WARN` (not FAIL).
Status:
- PASS if performance evidence is successful (exit code 0 OR performance_report status ok OR eval_result timing fields present) AND the 10/100 average methodology (or an accepted CUDA-graph/CUDA-event equivalent above) is verified
- WARN if performance evidence is successful but the methodology differs from 10 warmup / 100 measured average, or cannot be verified
- FAIL if performance evidence is unsuccessful
- TIMEOUT if exceeded {performance_timeout}s
- SKIP if correctness failed or was skipped (e.g. empty generation-placeholder kernel), or no performance command.

### Check 7: Correctness Implementation Review
Read the correctness implementation code (usually in `scripts/task_runner.py` or a test file).
Analyze whether the correctness check is meaningful:
- Does it compare against a known-good reference (numpy, CPU implementation, or known output)?
- Does it use reasonable tolerances (atol, rtol)?
- Could it trivially pass regardless of kernel output (e.g., always returns 0, no actual comparison)?
- Does it test with sufficient input shapes/sizes?
For a `torch2flydsl` starter, catching only the target's explicit `NotImplementedError` while independently
checking the reference/oracle is acceptable. A broad exception handler or fallback that lets missing
symbols, import errors, runtime errors, or incorrect implemented targets pass is trivially passing: mark
this check `FAIL` and set `is_trivially_passing: true`.
Status: PASS if implementation appears sound, WARN if questionable but functional, FAIL if trivially passing.
Set `is_trivially_passing: true` if the check would pass even with garbage output.

### Check 8: Self-Contained Check
Examine all source files for external dependencies:
- Check `#include` directives for headers that don't exist in the workspace
- Check Python `import` statements for modules not available in standard library or common packages
- Check if any file paths reference locations outside the workspace (e.g., `/path/to/vllm/`, `../../external/`)
- Check if scripts reference external repos or data that must be pre-downloaded
Treat standard ROCm/PyTorch toolchain headers (e.g., `torch/extension.h`, `ATen/*`, `hip/hip_runtime.h`, `c10/*`) and common runtime packages (`torch`, `yaml`) as allowed environment dependencies, not self-contained failures.
For `task_type: repository`, the task BY DESIGN clones a full upstream project and builds it with that
project's own build system. Dependencies resolved by the project's build (e.g. CMake `FetchContent`/
`download_project` for GoogleTest, Google Benchmark, rocm-cmake, rocRAND) and packages installed via the
task's declared `post_clone_install` step are EXPECTED and allowed — do NOT FAIL self-contained merely
because the upstream build fetches or installs its standard build/test dependencies. Only FAIL a
repository task here if it references resources outside its own clone + declared install step.
For `task_type: image_kernel`, the task similarly receives a full upstream repository copied from
the declared `image_repo_path`. Dependencies already installed in the task's declared container image
are expected environment dependencies. Do not require an external clone or install step; only FAIL if
runtime files are missing from both the seeded repository and its declared container environment.
List all missing files/dependencies found.
Status: PASS if fully self-contained (or its repository/image-kernel dependencies are covered as described above), FAIL if external dependencies found.

### Check 9: GPU Hang Check
Based on checks 4-6, report whether any command appeared to hang:
- Did any command hit the timeout?
- Were there any signs of GPU hang (e.g., process killed, no output for extended period)?
Status: PASS if all commands completed normally, FAIL if any hung, WARN if timeouts occurred but process was recoverable.

### Check 10: Result Template Compatibility
Check if the task's compile/correctness/performance flow provides the signals needed by the centralized evaluator's `task_result.yaml` schema.
The core schema expects: task_name, pass_compilation, compilation_error_message, pass_correctness, correctness_error_message, base_execution_time, best_optimized_execution_time, speedup_ratio, timing-method metadata, valid-case counts, speedup_calculation_error_message, optimization_summary, and score.
Does the task's runner/script produce timing information? Does it output pass/fail status in a parseable way?
If outputs are in `eval_result.yaml` with parseable keys (`compiled`, `correctness`, `speedup`, `ori_time`, `opt_time`) and/or `build/*.json` reports, consider this compatible via deterministic field mapping; do not require exact file name or exact schema shape.

For `task_type: repository` or `task_type: image_kernel` (driven by `scripts/task_runner.py`), the standard outputs are
`build/compile_report.json` (compile pass/fail), `build/correctness_report.json` (correctness pass/fail),
and `build/performance_report.json` (per-case `execution_time_ms` for the optimized build). These, combined
with `baseline_perf.yaml` (per-case baseline `execution_time_ms` produced by the harness), give a
deterministic mapping: pass_compilation/pass_correctness from the JSON reports, best_optimized_execution_time
from the performance report, base_execution_time from baseline_perf.yaml, and speedup_ratio = base/optimized.
`eval_result.yaml` and the legacy `task_result_template` field are NOT required — do not FAIL solely on their absence. Mark PASS when the command results and reports can be mapped deterministically.

For optimization tasks whose performance test emits per-case timing (e.g. `perf/benchmark_results.json`
with mean/median per configuration) and signals compile/correctness via pytest exit codes — typical of
`triton2triton` / `instruction2triton` — the BASELINE timing does NOT need to live inside the perf test,
and the test does NOT need its own PyTorch/reference baseline. The harness derives `base_execution_time`
by running the SAME `performance_command` against the ORIGINAL (unmodified) kernel before the agent runs
(saved to `baseline_perf.yaml`) and `best_optimized_execution_time` from the optimized run, then computes
`speedup_ratio = base/optimized` itself. Therefore `task_result_template: null` and the absence of an
in-test baseline are EXPECTED and fine — do NOT WARN/FAIL Check 10 for that reason. Mark PASS as long as the
performance command emits parseable per-case timing and compile/correctness pass/fail are observable.
Status: PASS if fields can be mapped deterministically, FAIL only if essential pass/fail/timing signals are missing.

## Output Format

After completing ALL checks, create a file called `validation_report.yaml` in the workspace directory (`{workspace}/validation_report.yaml`) with the following structure:

```yaml
{VALIDATION_REPORT_SCHEMA}
```

### Rules for overall_status:
- **PASS**: All applicable checks passed (no FAIL, no WARN); a contract-allowed `SKIP` such as an
  intentional generation placeholder or confirmed `torch2flydsl` starter does not prevent overall PASS
- **WARN**: No FAIL checks, but at least one WARN
- **FAIL**: At least one check has status FAIL

### Important Notes:
- Run each command from within the workspace directory `{workspace}`
- Capture the FIRST ~500 characters of stdout/stderr for snippets (don't include the full output)
- Use the `timeout` command to enforce time limits. Run each task command verbatim as configured.
  Do NOT wrap commands with `/usr/bin/time` (the GNU time binary may be absent and would make the
  command fail with exit code 127 / "No such file or directory"); if you need wall-clock timing, use
  the bash builtin `time` instead.
- If a command produces no output, note that in the snippet
- Be thorough but objective - report what you find, don't try to fix issues
- The validation_report.yaml MUST be valid YAML - use proper quoting for strings with special characters
"""

    return prompt
