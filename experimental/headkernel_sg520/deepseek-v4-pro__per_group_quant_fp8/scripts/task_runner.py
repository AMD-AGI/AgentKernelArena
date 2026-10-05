"""Minimal experimental AKA task entrypoint; framework validation is pending."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    build = ROOT / "build"
    build.mkdir(parents=True, exist_ok=True)
    report = build / (args.mode + "_report.json")
    report.unlink(missing_ok=True)
    if args.mode == "compile":
        sys.path.insert(0, str(ROOT / "ut"))
        from native import baseline, candidate
        baseline()
        candidate()
        report.write_text(json.dumps({"status": "ok", "kind": "native extension build", "task_validator_status": "pending"}) + "\n")
        print("Compilation: PASS")
        return 0
    script = "check.py" if args.mode == "correctness" else "benchmark.py"
    completed = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT)
    if args.mode == "correctness":
        report.write_text(json.dumps({"status": "ok" if completed.returncode == 0 else "fail", "kind": "independent native correctness", "task_validator_status": "pending"}) + "\n")
        print("Correctness: PASS" if completed.returncode == 0 else "Correctness: FAIL")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
