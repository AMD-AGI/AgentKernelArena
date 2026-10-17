"""Arena sikl-schema export: the validated, faster candidate's repo path and text, or nothing."""
import json
import os
import sys
from pathlib import Path

# `python3 -I` leaves the script's directory off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_runner as runner  # noqa: E402


def validated_candidate(workspace: Path) -> tuple[str, str]:
    reports = runner.report_dir(workspace, "candidate")
    if not (reports / "exit_code").is_file():
        raise runner.Failed(f"no report for source/ {reports.name}")
    exit_code = int((reports / "exit_code").read_text())
    if exit_code:
        raise runner.Failed(f"validate exit code {exit_code}")
    bench = runner.stage(json.loads((reports / "report.json").read_text(encoding="utf-8")), "bench")
    status = bench["status"] if bench else "missing"
    if status != "passed":
        raise runner.Failed(f"bench stage is {status}")
    files = sorted(reports.glob("sikl/ops/*/impls/*/*/*.json"))
    if len(files) != 1:
        raise runner.Failed(f"{len(files)} candidate files in {reports}")
    path = files[0].relative_to(reports).as_posix()
    commit = runner.read_json(workspace, "scripts/provenance.json")["sikl_commit"]
    if runner.git(runner.SIKL_ROOT, "ls-tree", commit, "--", path).strip():
        raise runner.Failed(f"{path} exists at base")
    # SIKL holds a new file to "retain" (>= 0.99) against the kernel it would displace; only gains are exported.
    slow = [runner.axes_text(r["axes"]) for r in bench["workload_records"]
            if any(t["threshold"] == "retain" and t["ratio_lower_bound"] <= 1.0 for t in r["timings"])]
    if slow:
        raise runner.Failed("not faster than the served kernel at " + ", ".join(slow))
    return path, files[0].read_text(encoding="utf-8")


def main() -> int:
    try:
        path, text = validated_candidate(Path.cwd())
    except Exception as exc:  # Arena records the export as FAIL with this message, never a traceback
        print(f"no export: {exc or type(exc).__name__}", file=sys.stderr)
        return 1
    Path(os.environ["ARENA_EXPORT_PATH"]).write_text(
        json.dumps({"path": path, "text": text}, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
