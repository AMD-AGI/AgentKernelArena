#!/usr/bin/env python3
"""MiniMax generated-input entrypoint; original capture replay is explicit and optional."""
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


runner = load("_minimax_arena_runner", HERE / "task_runner.py")
controller = load("_minimax_generated_controller", HERE / "generated_correctness.py")


def verify_archive():
    controller.verify_required(runner)
    meta = json.loads((runner.UT_DIR / "meta.json").read_text())
    path = runner.UT_DIR / "reference_io.pt"
    if not path.is_file() or runner.file_digest(path) != meta["archival_capture"]["reference_io_sha256"]:
        raise RuntimeError("optional archival replay needs the unchanged original reference_io.pt")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "performance":
        meta = json.loads((runner.UT_DIR / "meta.json").read_text())
        gate = meta.get("performance_contract") or {}
        if gate.get("status") != "ready":
            reason = gate.get("reason", "exact observed performance contract is missing")
            runner.write_report("performance_report.json", {
                "status": "fail", "reason": "missing_observed_workload_controls",
                "error": reason, "scenario_contract": gate, "test_cases": []})
            print("Performance: BLOCKED - " + reason)
            return 1
    if len(sys.argv) > 1 and sys.argv[1] == "archival":
        sys.argv[1] = "correctness"
        runner.verify_fixtures = verify_archive
        original_load = runner.load_config
        def archival_config():
            cfg = original_load()
            cfg["headkernel"]["oracle"] = "optional unchanged captured-value reference_io.pt"
            return cfg
        runner.load_config = archival_config
    else:
        runner.verify_fixtures = lambda: controller.verify_required(runner)
        runner.run_correctness = lambda cfg, timeout: controller.run_correctness(runner, cfg, timeout)
    return runner.main()


if __name__ == "__main__":
    sys.exit(main())
