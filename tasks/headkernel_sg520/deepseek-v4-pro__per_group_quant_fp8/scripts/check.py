"""Qualify both isolated native legs against the independent oracle."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from contract import LEGS, run_workers, write_report


def main():
    output = ROOT / "build/correctness_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    manifest, cases, results, identity = run_workers("correctness")
    report = {"status": "ok", **identity, "source_run": manifest["source_run"],
              "case_count": len(cases), "isolated_native_processes": True,
              "task_validator_status": "pending", "workers": results}
    write_report(output, report)
    print(json.dumps({"status": "ok", "case_count": len(cases), "legs": list(LEGS),
                      "report": str(output.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
