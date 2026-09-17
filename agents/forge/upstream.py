"""Arena's view of the two KernelForge CLIs.

The adapter drives the engine through its published command line and the
measurement driver it hands over. It installs nothing into the engine and
reads none of its private interfaces, so the two release cadences stay
independent: an engine that still accepts the documented options works.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.forge.bridge import bound_candidate_root, load_plan
from agents.forge.task_context import TaskContext
from agents.forge.bundles import candidate_files, protected_paths

ADAPTER_API = 1


def _modules():
    from kernelforge import cli
    from kernelforge.kernel_backends.constants import KERNEL_BACKENDS
    return SimpleNamespace(**locals())


def probe() -> dict:
    """Check the published command line the adapter builds, nothing deeper."""
    modules = _modules()
    try:
        version = importlib.metadata.version("hyperloom-inference_optimizer")
    except importlib.metadata.PackageNotFoundError:
        version = "source-checkout"
    required = {
        "forge-loop": {"kernel", "driver", "workspace_dir", "deadline_unix",
                       "source_files", "target_functions", "baseline_json"},
        "forge-rewrite-by-flydsl": {"source_kernel", "driver", "workspace_dir", "deadline_unix",
                                    "prepare_driver", "flydsl_kernel_name"},
    }
    for name, parameters in required.items():
        command = modules.cli.main.commands.get(name)
        if command is None:
            raise RuntimeError(f"Installed KernelForge lacks {name}")
        missing = parameters - {parameter.name for parameter in command.params}
        if missing:
            raise RuntimeError(f"KernelForge {name} lacks adapter-required parameters: {sorted(missing)}")
    return {"adapter_api": ADAPTER_API, "version": version,
            "initialization_targets": ["flydsl", "hip", "triton"],
            "backends": sorted(modules.KERNEL_BACKENDS), "rewrite_target": "flydsl"}


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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--arena-probe"]:
        print(json.dumps(probe()))
        return 0
    plan = load_plan(Path(os.environ["ARENA_FORGE_PLAN"]))
    probe()  # Refuse an engine whose published command line moved.
    from kernelforge.cli import main as cli
    from agents.forge.process_tree import managed_children
    with managed_children():
        initialization = None
        if argv and argv[0] == "--arena-initialize":
            from agents.forge.initialization import initialize, prepare_loop
            initialization = asyncio.run(initialize(plan))
            argv = prepare_loop(plan, argv[1:])
        exit_code = cli(args=argv, standalone_mode=False)
        # Click returns Exit.exit_code instead of raising SystemExit in this
        # mode. Preserve a failed engine exit, even if it wrote partial JSON.
        if isinstance(exit_code, int) and exit_code != 0:
            raise SystemExit(exit_code)
        if initialization is not None:
            from kernelforge.tracker.usage import combine_usage_totals
            result_path = Path(plan["result"])
            result = json.loads(result_path.read_text())
            result["initialization"] = initialization
            result["optimization_llm_usage"] = result.get("llm_usage")
            result["llm_usage"] = combine_usage_totals(initialization["llm_usage"], result.get("llm_usage"),
                incomplete=not isinstance(result.get("llm_usage"), dict) or result.get("llm_usage_complete") is False)
            result_path.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
