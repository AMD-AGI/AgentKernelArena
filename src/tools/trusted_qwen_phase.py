"""Bind an existing protected Qwen phase to a host request; stdlib only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time


def run(task, request_path, timeout):
    task = Path(task)
    request = json.loads(Path(request_path).read_text())
    phase = request["phase"]
    if phase not in ("compile", "correctness", "performance"):
        raise ValueError("unknown trusted phase")
    def sources():
        return {name: hashlib.sha256((task / name).read_bytes()).hexdigest()
                for name in request["source_sha256"]}
    if sources() != request["source_sha256"]:
        raise ValueError("trusted phase source differs before execution")
    receipt = {"schema": "trusted-qwen-phase-v1", "host_request": request,
               "started_unix": time.time(), "returncode": 1}
    runner = task / "scripts/task_runner.py"
    prior_argv, prior_path = sys.argv, sys.path[:]
    try:
        sys.path.insert(0, str(runner.parent))
        sys.argv = [str(runner), phase, "--timeout", str(timeout)]
        try:
            runpy.run_path(str(runner), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise RuntimeError("protected Qwen phase failed") from error
        if sources() != request["source_sha256"]:
            raise ValueError("trusted phase source changed during execution")
        report = task / "build" / (phase + "_report.json")
        receipt.update(returncode=0, producer_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    finally:
        sys.argv, sys.path[:] = prior_argv, prior_path
        receipt["ended_unix"] = time.time()
        with (task / "build/host_phase_receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    run(args.task, args.request, args.timeout)
