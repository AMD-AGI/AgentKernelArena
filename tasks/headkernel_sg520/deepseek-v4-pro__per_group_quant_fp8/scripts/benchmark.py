"""Fresh-input graph timings from separately loaded native processes."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from contract import performance_report, run_workers, write_report


def main():
    output = ROOT / "build/performance_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    try:
        manifest, cases, results, identity = run_workers("performance")
        report = performance_report(manifest, cases, results, identity)
        write_report(output, report)
    except BaseException:
        output.unlink(missing_ok=True)
        output.with_suffix(".tmp").unlink(missing_ok=True)
        raise
    print(json.dumps({"status": "ok", "case_count": len(cases),
                      "weight_sum_per_rank": report["weight_sum_per_rank"],
                      "report": str(output.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
