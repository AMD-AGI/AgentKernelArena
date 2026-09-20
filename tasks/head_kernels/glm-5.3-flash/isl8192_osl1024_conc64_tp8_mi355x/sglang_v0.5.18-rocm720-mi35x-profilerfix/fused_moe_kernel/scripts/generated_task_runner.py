#!/usr/bin/env python3
"""Portable GLM task entrypoint with parent-only reference comparisons."""
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


runner = load("_glm_arena_runner", HERE / "task_runner.py")
controller = load("_glm_generated_controller", HERE / "generated_correctness.py")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "performance":
        metadata = json.loads((runner.UT_DIR / "meta.json").read_text())
        scoring = metadata.get("generated_inputs", {}).get("workload_scoring", {})
        if scoring.get("enabled") is False:
            runner.write_report("performance_report.json", {"status": "fail", "test_cases": [],
                "workload_scoring_status": scoring["status"], "error": scoring["reason"]})
            print("Workload scoring blocked: " + scoring["reason"])
            return 1
    runner.verify_fixtures = lambda: controller.verify_required(runner)
    runner.run_correctness = lambda cfg, timeout: controller.run_correctness(runner, cfg, timeout)
    runner.run_performance = lambda cfg, timeout: controller.run_performance(runner, cfg, timeout)
    return runner.main()


if __name__ == "__main__":
    sys.exit(main())
