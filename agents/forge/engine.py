"""The two things Arena cannot ask of the engine over its command line.

Optimization and rewrite campaigns are plain subprocesses: the adapter builds
argv and runs ``kernelforge.cli``. Two jobs still need an interpreter that has
the engine importable, so they live here and are invoked as scripts.

``--arena-probe`` answers what the installed engine accepts, in the exact
interpreter the campaign will use.

``--arena-initialize`` runs the correctness-only implementer the engine exposes
only as a library, then hands the same process to ``forge-loop``. Remove this
once the engine offers that phase as a command of its own.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.forge.bridge import load_plan

ADAPTER_API = 1


def probe() -> dict:
    """Check the published command line the adapter builds, nothing deeper."""
    from kernelforge import cli
    from kernelforge.kernel_backends.constants import KERNEL_BACKENDS

    try:
        version = importlib.metadata.version("hyperloom-inference_optimizer")
    except importlib.metadata.PackageNotFoundError:
        version = "source-checkout"
    required = {
        "forge-loop": {"kernel", "driver", "workspace_dir", "max_hours",
                       "source_files", "target_functions", "baseline_json"},
        "forge-rewrite-by-flydsl": {"source_kernel", "driver", "workspace_dir", "max_hours",
                                    "prepare_driver", "flydsl_kernel_name"},
    }
    for name, parameters in required.items():
        command = cli.main.commands.get(name)
        if command is None:
            raise RuntimeError(f"Installed KernelForge lacks {name}")
        missing = parameters - {parameter.name for parameter in command.params}
        if missing:
            raise RuntimeError(f"KernelForge {name} lacks adapter-required parameters: {sorted(missing)}")
    return {"adapter_api": ADAPTER_API, "version": version,
            "initialization_targets": ["flydsl", "hip", "triton"],
            "backends": sorted(KERNEL_BACKENDS), "rewrite_target": "flydsl"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--arena-probe"]:
        print(json.dumps(probe()))
        return 0
    if not argv or argv[0] != "--arena-initialize":
        raise SystemExit("Only --arena-probe and --arena-initialize run through this entry point")
    plan = load_plan(Path(os.environ["ARENA_FORGE_PLAN"]))
    probe()  # Refuse an engine whose published command line moved.
    from kernelforge.cli import main as cli
    from agents.forge.initialization import initialize, prepare_loop
    from agents.forge.process_tree import managed_children
    with managed_children():
        initialization = asyncio.run(initialize(plan))
        exit_code = cli(args=prepare_loop(plan, argv[1:]), standalone_mode=False)
        # Click returns Exit.exit_code instead of raising SystemExit in this
        # mode. Preserve a failed engine exit, even if it wrote partial JSON.
        if isinstance(exit_code, int) and exit_code != 0:
            raise SystemExit(exit_code)
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
