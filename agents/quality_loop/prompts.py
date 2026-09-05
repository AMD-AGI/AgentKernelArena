# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path

import yaml


def repair_prompt(report: dict, task_id: str) -> str:
    return f"""# quality_loop validation repair

Task: `{task_id}`

The task validator found blocking failures. Fix only the FAIL/TIMEOUT findings in
this task workspace. Do not spend time fixing WARN-only findings. Preserve the
task's intended computation and public contract. Run the relevant compile and
correctness commands after editing. Do not use GitHub, git, network services, or
edit anything outside this workspace.

Validation report:
```yaml
{yaml.safe_dump(report, sort_keys=False)}
```
"""


def optimizer_prompt(base_prompt: str, task_id: str) -> str:
    return base_prompt.rstrip() + f"""

## quality_loop single-iteration boundary

This is the one and only optimization iteration for `{task_id}`. Produce at most
one candidate implementation. Complete all analysis, implementation, compile,
correctness, and performance checks needed for that candidate in this single
iteration. You may edit only declared kernel/source files. Do not edit config,
tests, scripts, performance helpers, or any other harness file. Do not use git or
GitHub. Preserve the exact computation, outputs, dtypes, shapes, aliasing, and
side effects of the original task.
"""


def reviewer_prompt(task_id: str, result_file: Path, output_name: str) -> str:
    return f"""# quality_loop independent evaluation review

Review task `{task_id}` independently. You are a read-only evaluator, not the
optimizer. Compare the candidate's declared source paths with their pre-optimizer
copies under `.quality_loop_original_sources/`. Inspect the config, test harness,
and centralized evaluator evidence in `{result_file.name}`. Decide whether the
candidate preserves the task's computation and whether the evidence is strong
enough to accept it. Also decide whether task cases have material coverage gaps.

Do not edit any existing file. Write exactly one new YAML file `{output_name}`:

```yaml
accepted: true                    # boolean
logic_equivalent: true            # boolean
evidence_sufficient: true         # boolean
case_enhancement_needed: false    # boolean
case_rationale: "..."
summary: "..."
```

Fail closed: set accepted false when behavior is ambiguous, evidence is missing,
the harness changed, performance methods differ, valid case counts shrink, or
the candidate depends on untested assumptions. Do not use git, GitHub, or network
services.
"""


def case_enhancement_prompt(task_id: str, rationale: str) -> str:
    return f"""# quality_loop task-case hardening

Task: `{task_id}`
Reviewer rationale: {rationale}

Strengthen correctness coverage only where the rationale identifies a real gap.
Add a small, targeted set of valid boundary/shape/dtype cases. Do not modify the
kernel/source implementation, computation contract, tolerances merely to accept
wrong answers, benchmark timing helpers, or performance methodology. Every new
case must be valid for and pass the original pre-audit kernel. Run the appropriate
correctness command. Do not use git, GitHub, or network services.
"""
