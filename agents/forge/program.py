"""The task contract an engine session is handed, written from the declaration.

This is Arena's prose, not the engine's: it restates the task's own editable
scope, entrypoints, baseline and evaluation so a session reads the contract
rather than inferring one from a backend guide. Nothing here imports the
engine.
"""
from __future__ import annotations

import json
from pathlib import Path

from agents.forge.task_context import TaskContext


def program_text(plan: dict, *, prefix: str = "", port: bool = False, initialize: bool = False) -> str:
    context = TaskContext.load(plan["context"])
    spec = context.spec

    config = spec.to_mapping()
    declarations = config["candidate"].copy()
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
    return "\n".join([
        "# Arena task contract", "Task: " + spec.task_id,
        "Implement the task using " + spec.candidate.language + ".",
        "Task implementation and dependency constraints take precedence over backend guides, examples, "
        "and knowledge-base suggestions. Passing the driver does not waive these constraints. "
        "Do not treat an available library or backend example as permission to delegate an operator "
        "when the task forbids that delegation. A prohibited library operator remains prohibited "
        "even if it uses the target language internally. Wrapping that operator or tuning its launch "
        "parameters does not satisfy a requirement to implement the operator in candidate-owned kernels.",
        "The following paths are relative to " + (prefix or "the workspace root") + ".",
        "Editable declarations and real entrypoints (do not invent a factory convention):",
        json.dumps(declarations, indent=2),
        "Read the task's baseline/reference and evaluator for the exact interface and semantics:",
        json.dumps({"baseline": config["baseline"], "evaluation": config["evaluation"]}, indent=2),
        "The protected driver calls the task's compile and complete correctness commands.",
        "Correctness uses the task's own reference and comparison, not a generic SNR threshold.",
        "Run python3 arena_forge_driver.py for correctness; --bench-mode for candidate timing;",
        "--ref-bench-mode for the independent baseline. Task-owned cases and warmups are fixed.",
        "The driver reports allclose from the task verdict. Do not modify task configuration,",
        "harnesses, inputs, references, or import protected implementations into your candidate.",
        "INITIALIZE: implement the missing target-language code. Full task correctness and all declared "
        "implementation constraints are required; no speedup is required. Keep partial work between attempts and use compiler "
        "and correctness feedback. The same Forge loop optimizes the first valid implementation afterward."
        if initialize else "PORT first produces a correct implementation; the nested loop then optimizes it." if port else
        "Optimize the existing candidate. Keep all declared entrypoints and dependent source files.",
        *instructions,
    ])
