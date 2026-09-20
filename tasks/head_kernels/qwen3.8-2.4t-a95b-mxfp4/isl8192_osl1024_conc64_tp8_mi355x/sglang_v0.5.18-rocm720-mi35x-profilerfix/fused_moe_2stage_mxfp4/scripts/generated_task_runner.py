#!/usr/bin/env python3
"""Qwen generated-input entrypoint with the unchanged common performance runner.

Performance delegates to scripts/_bench.py and its canonical _aka_benchmark
helper, materialized beside this entrypoint by Arena workspace setup.
"""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != Path(path).resolve():
            raise RuntimeError(f"trusted alias names another file: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    runner = load("_qwen_arena_runner", HERE / "task_runner.py")
    controller = load("_qwen_generated_controller", HERE / "generated_correctness.py")
    runner.verify_fixtures = lambda: controller.verify_required(runner)
    runner.run_correctness = lambda cfg, timeout: controller.run_correctness(runner, cfg, timeout)
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
