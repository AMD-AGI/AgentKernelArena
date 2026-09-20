#!/usr/bin/env python3
"""Kimi generated inputs, using materialized _aka_benchmark timing and parent comparison.

Original-archive replay remains an explicit optional mode.
"""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("_kimi_original_runner", HERE / "task_runner.py")
controller = load("_kimi_generated_controller", HERE / "generated_controller.py")


def verify_archive():
    controller.verify_required(runner)
    meta = json.loads((runner.UT_DIR / "meta.json").read_text())
    path = runner.UT_DIR / "reference_io.pt"
    if not path.is_file() or runner.file_digest(path) != meta["archival_capture"]["reference_io_sha256"]:
        raise RuntimeError("optional archival replay requires the unchanged original archive")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "archival":
        raise RuntimeError("Archive bytes and original UT are retained; archive-only execution needs a separately reviewed preloader configuration")
    else:
        runner.verify_fixtures = lambda: controller.verify_required(runner)
        runner.run_correctness = lambda cfg, timeout: controller.run_correctness(runner, cfg, timeout)
        runner.run_performance = lambda cfg, timeout: controller.run_performance(runner, cfg, timeout)
    return runner.main()


if __name__ == "__main__":
    sys.exit(main())
