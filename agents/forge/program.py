"""The task contract an engine session is handed, written from the declaration.

This is Arena's prose, not the engine's: it restates the schema-v3 task's
candidate tree, entry point, baseline and evaluation so a session reads the
contract rather than inferring one from a backend guide. Nothing here imports
the engine.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.task_contract import BASELINE_DIR, EVAL_DIR, SOLUTION_DIR
from agents.forge.task_context import TaskContext


def program_text(plan: dict, *, prefix: str = "", port: bool = False, initialize: bool = False) -> str:
    context = TaskContext.load(plan["context"])
    spec = context.spec
    config = spec.to_mapping()
    entry = spec.candidate.entrypoints[0].file
    instructions = [config.get("description", "")]
    if plan.get("initial_candidate_failure"):
        instructions += [
            "Historical input assessment: an earlier candidate compiled but failed the complete numerical check. "
            "The protected driver determines whether the current candidate now passes. Optimization requires "
            "a currently passing implementation; these retained diagnostics do not override a later successful check:",
            json.dumps(plan["initial_candidate_failure"], ensure_ascii=False)[:4000],
        ]
    for name in dict.fromkeys(["README.md", *config.get("instructions", [])]):
        path = Path(plan["template"]) / name
        if path.is_file():
            instructions.append(f"\n## {name}\n" + path.read_text())
    if initialize:
        phase = ("INITIALIZE: replace the unimplemented template with a complete target-language "
                 "implementation. Full task correctness and all candidate rules are required; no speedup "
                 "is required. Keep partial work between attempts and use compiler and correctness "
                 "feedback. The same Forge loop optimizes the first valid implementation afterward.")
    elif port:
        phase = "PORT first produces a correct implementation; the nested loop then optimizes it."
    else:
        phase = "Optimize the existing candidate. Keep its entry point and every helper it imports."
    return "\n".join([
        "# Arena task contract", "Task: " + spec.task_id,
        f"Operator: {context.operator}. Implement it in {context.target_language}.",
        "Task implementation and dependency constraints take precedence over backend guides, examples, "
        "and knowledge-base suggestions. Passing the driver does not waive these constraints. "
        "Do not treat an available library or backend example as permission to delegate an operator "
        "when the task forbids that delegation. A prohibited library operator remains prohibited "
        "even if it uses the target language internally. Wrapping that operator or tuning its launch "
        "parameters does not satisfy a requirement to implement the operator in candidate-owned kernels.",
        "The following paths are relative to " + (prefix or "the workspace root") + ".",
        f"Edit only {SOLUTION_DIR}/. The entry point is {entry}::run; other files there form one flat "
        "package imported relatively. Every other file is read-only.",
        f"{EVAL_DIR}/CANDIDATE.md states the exact run() signature, interface and candidate rules.",
        f"The baseline to beat is {BASELINE_DIR}/ ({context.baseline_language}); read it, task.yaml, "
        "workloads.json and the reference for the semantics, but never import them into the candidate.",
        "The protected driver calls the task's compile and complete correctness commands.",
        "Correctness uses the task's own reference and comparison, not a generic SNR threshold.",
        "Run python3 arena_forge_driver.py for correctness; --bench-mode for candidate timing;",
        "--ref-bench-mode for the independent baseline. Task-owned cases and warmups are fixed.",
        "The driver reports allclose from the task verdict. Do not modify task configuration, the",
        f"evaluator ({EVAL_DIR}/), inputs, references, or the baseline.",
        phase,
        *instructions,
    ])
