#!/usr/bin/env python3
"""Task-local entrypoint for generated DeepSeek inputs and frozen references."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != Path(path).resolve():
            raise RuntimeError(f"trusted helper alias names a different file: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("_deepseek_arena_runner", HERE / "task_runner.py")
controller = load("_deepseek_generated_controller", HERE / "generated_correctness.py")


def main():
    runner.verify_fixtures = lambda: controller.verify_required(runner)
    runner.run_correctness = lambda cfg, timeout: controller.run_correctness(
        runner, cfg, timeout
    )
    # cases._resolve loads frozen or editable source in its original package
    # context. No startup overlay can run before helper attestation.
    runner.overlays = lambda: (None, None)
    return runner.main()


if __name__ == "__main__":
    sys.exit(main())
