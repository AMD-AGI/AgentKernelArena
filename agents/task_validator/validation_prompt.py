# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
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
    status: ""                         # PASS | FAIL | TIMEOUT | SKIP
    exit_code: null
    duration_seconds: null
    stdout_snippet: ""
    stderr_snippet: ""
    report_file_valid: null
    analysis: ""

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
    python_path = eval_config.get('agent', {}).get('python_path', '/root/AgentKernelArena/.venv/bin/python3')
    compile_timeout = eval_config.get('agent', {}).get('compile_timeout', 300)
    correctness_timeout = eval_config.get('agent', {}).get('correctness_timeout', 300)
    performance_timeout = eval_config.get('agent', {}).get('performance_timeout', 300)

    prompt = f"""# Task Validation Agent

You are a **task validator**, not an optimizer. Your job is to validate that a GPU kernel optimization task is correctly configured, self-contained, and functional.

## Workspace
Your working directory is: `{workspace}`

## Task Configuration
The task config.yaml is located at: `{task_config_dir}`

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
- `task_type` (string, one of: hip2hip, cuda2hip, triton2triton, torch2hip, instruction2triton, rocprim)
Also check that optional fields (`performance_command`, `prompt`) are well-formed if present.
Status: PASS if all required fields exist and have correct types, FAIL otherwise.

### Check 2: Source Files Exist
For each file listed in `source_file_path`: {source_files}
Check if the file exists in the workspace directory `{workspace}`.
Look for the file directly and also under common subdirectories (source/, src/, scripts/).
Status: PASS if all source files are found, FAIL if any are missing.

### Check 3: Target Symbols Found
For each function in `target_kernel_functions`: {target_kernels}
Search the source files for the function name (as a symbol definition, not just a string mention).
For CUDA/HIP: look for `__global__ void <name>` or similar kernel declarations.
For Triton: look for `@triton.jit` decorated functions with the name.
For Python: look for `def <name>`.
Report the file and line number where each symbol is found.
Status: PASS if all target symbols found, FAIL if any are missing.

### Check 4: Compilation
Run the compile command(s) from the workspace directory:
```
{chr(10).join(compile_cmds) if compile_cmds else 'No compile command specified'}
```
Use a timeout of {compile_timeout} seconds per command. Run the command EXACTLY ONCE — do NOT retry on failure or timeout.
Capture stdout, stderr, and exit code.
Also check if `build/compile_report.json` is generated and contains a valid status.
If exit code is non-zero but `eval_result.yaml` clearly records `compiled: true`, treat compilation as PASS and document the wrapper/command inconsistency in details.
Status: PASS if compilation evidence is successful (exit code 0 OR compile_report status ok OR eval_result compiled=true), FAIL otherwise, TIMEOUT if exceeded {compile_timeout}s.

### Check 5: Correctness
Run the correctness command(s) from the workspace directory:
```
{chr(10).join(correctness_cmds) if correctness_cmds else 'No correctness command specified'}
```
Use a timeout of {correctness_timeout} seconds per command. Run the command EXACTLY ONCE — do NOT retry on failure or timeout.
Capture stdout, stderr, and exit code.
Check if `build/correctness_report.json` is generated.
If exit code is non-zero but `eval_result.yaml` clearly records `correctness: true`, treat correctness as PASS and explain the inconsistency.
Status: PASS if correctness evidence is successful (exit code 0 OR correctness_report status ok OR eval_result correctness=true), FAIL otherwise, TIMEOUT if exceeded {correctness_timeout}s, SKIP if compilation failed.

### Check 6: Performance
Run the performance command(s) from the workspace directory (if any):
```
{chr(10).join(performance_cmds) if performance_cmds else 'No performance command specified'}
```
Use a timeout of {performance_timeout} seconds per command. Run the command EXACTLY ONCE — do NOT retry on failure or timeout.
Capture stdout, stderr, and exit code.
If timing fields are present in `eval_result.yaml` (`speedup`, `ori_time`, `opt_time`) and are non-null, treat performance as PASS even if wrapper exit code is inconsistent.
Status: PASS if performance evidence is successful (exit code 0 OR performance_report status ok OR eval_result timing fields present), FAIL otherwise, TIMEOUT if exceeded {performance_timeout}s, SKIP if correctness failed or no performance command.

### Check 7: Correctness Implementation Review
Read the correctness implementation code (usually in `scripts/task_runner.py` or a test file).
Analyze whether the correctness check is meaningful:
- Does it compare against a known-good reference (numpy, CPU implementation, or known output)?
- Does it use reasonable tolerances (atol, rtol)?
- Could it trivially pass regardless of kernel output (e.g., always returns 0, no actual comparison)?
- Does it test with sufficient input shapes/sizes?
Status: PASS if implementation appears sound, WARN if questionable but functional, FAIL if trivially passing.
Set `is_trivially_passing: true` if the check would pass even with garbage output.

### Check 8: Self-Contained Check
Examine all source files for external dependencies:
- Check `#include` directives for headers that don't exist in the workspace
- Check Python `import` statements for modules not available in standard library or common packages
- Check if any file paths reference locations outside the workspace (e.g., `/path/to/vllm/`, `../../external/`)
- Check if scripts reference external repos or data that must be pre-downloaded
Treat standard ROCm/PyTorch toolchain headers (e.g., `torch/extension.h`, `ATen/*`, `hip/hip_runtime.h`, `c10/*`) and common runtime packages (`torch`, `yaml`) as allowed environment dependencies, not self-contained failures.
List all missing files/dependencies found.
Status: PASS if fully self-contained, FAIL if external dependencies found.

### Check 9: GPU Hang Check
Based on checks 4-6, report whether any command appeared to hang:
- Did any command hit the timeout?
- Were there any signs of GPU hang (e.g., process killed, no output for extended period)?
Status: PASS if all commands completed normally, FAIL if any hung, WARN if timeouts occurred but process was recoverable.

### Check 10: Result Template Compatibility
Check if the task's compile/correctness/performance flow would produce output compatible with the standard `task_result_template.yaml` schema.
The schema expects: task_name, best_optimized_source_file_path, best_optimized_kernel_functions, pass_compilation, compilation_error_message, pass_correctness, correctness_error_message, base_execution_time, best_optimized_execution_time, speedup_ratio, optimization_summary.
Does the task's runner/script produce timing information? Does it output pass/fail status in a parseable way?
If outputs are in `eval_result.yaml` with parseable keys (`compiled`, `correctness`, `speedup`, `ori_time`, `opt_time`) and/or `build/*.json` reports, consider this compatible via deterministic field mapping; do not require exact file name or exact schema shape.
Status: PASS if fields can be mapped deterministically, FAIL only if essential pass/fail/timing signals are missing.

## Output Format

After completing ALL checks, create a file called `validation_report.yaml` in the workspace directory (`{workspace}/validation_report.yaml`) with the following structure:

```yaml
{VALIDATION_REPORT_SCHEMA}
```

### Rules for overall_status:
- **PASS**: ALL checks passed (no FAIL, no WARN)
- **WARN**: No FAIL checks, but at least one WARN
- **FAIL**: At least one check has status FAIL

### Important Notes:
- Run each command from within the workspace directory `{workspace}`
- Capture the FIRST ~500 characters of stdout/stderr for snippets (don't include the full output)
- Use `timeout` command or equivalent to enforce time limits
- If a command produces no output, note that in the snippet
- Be thorough but objective - report what you find, don't try to fix issues
- The validation_report.yaml MUST be valid YAML - use proper quoting for strings with special characters
"""

    return prompt
