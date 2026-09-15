"""Agent-independent instructions rendered from the shared task declaration."""
from pathlib import Path
import shlex

from .task_spec import TaskSpec, resolve_task_path


def build_task_prompt(spec: TaskSpec, workspace: Path, *, target_gpu: str,
                      hardware_context: str = "", target_arch: str | None = None) -> str:
    config = spec.to_mapping()
    candidate = spec.candidate
    lines = [
        f"Task: {spec.task_id}",
        config.get("description", "Implement and optimize the declared operator while preserving its task contract."),
        f"Workspace: {workspace}",
        f"Target GPU: {target_gpu}",
        f"Required final implementation backend: {candidate.language}",
        f"Initial candidate state: {candidate.initial_state}",
    ]
    if candidate.initial_language:
        lines.append(f"Initial implementation backend: {candidate.initial_language}")
    if target_arch:
        lines.extend([
            f"Target architecture token: `{target_arch}`",
            "Check runtime/build architecture compatibility before compiling. Report conflicting protected build settings; do not rewrite protected files to bypass them.",
        ])
    lines.extend([
        "The initial state describes the shipped task. On resume, inspect the current candidate before deciding what remains to implement.",
        "All task paths below are relative to this workspace root. Preserve their directory components.",
        "\nEditable implementation:",
    ])
    for edit in candidate.editable:
        if edit.scope == "symbols":
            detail = "only symbols " + ", ".join(edit.symbols)
            detail += "; new implementation helpers allowed" if edit.allow_new_helpers else "; no new helpers"
        elif edit.scope == "tree":
            detail = "implementation subtree; protected task/harness files remain protected"
        else:
            detail = "implementation file"
        lines.append(f"- {edit.path}: {detail}")
    if candidate.entrypoints:
        lines.append("\nFinal callable interface (read the task contract for signatures):")
        for entry in candidate.entrypoints:
            target = f":{entry.symbol}" if entry.symbol else ""
            lines.append(f"- {entry.file}{target} ({entry.kind})")
    lines.append(f"\nPerformance baseline: {spec.baseline.kind}")
    if spec.baseline.source_files:
        lines.append("Read-only baseline source material: " + ", ".join(spec.baseline.source_files))
    if spec.baseline.correctness_policy == "diagnostic":
        lines.extend([
            "The baseline has a declared numerical diagnostic policy: " + spec.baseline.diagnostic_reason,
            "This does not relax candidate correctness. The candidate must meet the task's full numerical requirements.",
        ])
    files = list(config.get("instructions", []))
    if (workspace / "README.md").is_file() and "README.md" not in files:
        files.insert(0, "README.md")
    for relative in files:
        path = resolve_task_path(workspace, relative, must_exist=True)
        lines.extend([f"\nTask instructions from {relative}:", path.read_text(encoding="utf-8")])
    lines.extend([
        "\nCandidate checks:",
        "Run these commands from the candidate workspace root. They use candidate_evaluation semantics, including during optimization.",
        "For each action run its commands sequentially. A failure stops the action; commands are not alternative fallbacks.",
    ])
    for action in ("compile", "correctness", "performance"):
        check = spec.action("candidate", action)
        lines.append(f"{action} (total action timeout: {check.timeout_s}s):")
        lines.extend("  " + shlex.join(command) for command in check.commands)
    lines.extend([
        "A passing command exits zero and emits an ARENA_EVAL_RESULT JSON report. Logs containing the word PASS are not evidence of success.",
        "Correctness/performance must cover the complete task cases. Re-run compilation and correctness after changing candidate code.",
        "Baseline actions execute in the separate frozen baseline workspace provided through ARENA_TASK_CONTEXT, not in the modified candidate tree.",
        "The context identifies the original baseline and task manifest. It is framework-owned; do not modify it or baseline sources.",
        "\nKeep task config, input generators, case data, reference/comparison code, harnesses, and timing policy unchanged.",
        "For a file shared with the harness, edit only the declared implementation symbols and permitted helpers.",
        "Follow the task's allowed-dependency rules. Do not substitute reference/baseline execution for a submitted implementation.",
        "Preserve equivalent baseline/candidate work and timing boundaries. The actual timed/replayed path must produce correct outputs.",
        "If the environment or a protected build script is incompatible, report that failure rather than editing the acceptance rules.",
        "\nDeliver the final implementation files at their declared workspace paths. If you use scratch, install all selected candidate files back before finishing.",
        "Arena independently compiles, validates, runs enabled evaluation tools, measures ordinary-build performance, and generates final results/exports.",
        "Do not write task_result.yaml or validation_report.yaml. Agent logs are supplementary, not scores or correctness evidence.",
    ])
    if hardware_context:
        lines.extend(["\nHardware and implementation reference:", hardware_context])
    return "\n".join(lines) + "\n"
