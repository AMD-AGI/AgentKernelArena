"""Roofline floors of the v3 kernel tasks; every output directory must be outside the repository.

    python3 -m src.roofline floors --arch mi355x_datasheet --out <OUT> [--precision-probe <PROBE.json>]
    python3 -m src.roofline join --floors <OUT>/floors.jsonl --evidence <EVIDENCE_DIR> --out <OUT> \
        [--practical-arch <MEASURED_PEAKS.json>]
    python3 -m src.roofline report --joined <OUT>/joined.jsonl --out <OUT>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from ..task_contract import TASK_FILE, load_task_contract
from .arch import load_peaks
from .floor import task_floors
from .join import join, measurements, read_jsonl, report_rows
from .precision_policy import load_probe

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = REPO_ROOT / "tasks"
EXCLUDED_ROOTS = ("e2e",)


class OutputInsideRepository(SystemExit):
    pass


def output_directory(path: str) -> Path:
    """Resolve an output directory, refusing one inside the repository: results are not committed."""
    resolved = Path(path).expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise OutputInsideRepository(f"--out must be outside the repository: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def discover_tasks(tasks_dir: Path = TASKS_DIR, selected: list[str] | None = None) -> list[str]:
    if selected:
        return list(selected)
    found = []
    for task_file in sorted(tasks_dir.glob(f"*/*/*/*/{TASK_FILE}")):
        task_id = task_file.parent.relative_to(tasks_dir).as_posix()
        if task_id.split("/", 1)[0] not in EXCLUDED_ROOTS:
            found.append(task_id)
    return found


def floors(args: argparse.Namespace) -> int:
    out = output_directory(args.out)
    peaks = load_peaks(args.arch)
    probe = load_probe(args.precision_probe) if args.precision_probe else None
    tasks_dir = Path(args.tasks_dir).resolve()
    records, failures = [], []
    device = args.device
    if args.generate and device == "auto":
        try:
            from .generated import default_device
            device = default_device()
        except ImportError:
            device = "cpu"  # generation will fail per case and record why
    for task_id in discover_tasks(tasks_dir, args.tasks):
        try:
            contract = load_task_contract(tasks_dir / task_id / TASK_FILE, task_id=task_id)
            records.extend(task_floors(contract, peaks, tasks_dir / task_id, generate=args.generate,
                                       device=device, probe=probe))
        except Exception as exc:  # one broken model must not hide every other floor
            failures.append({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
    _write_jsonl(out / "floors.jsonl", records)
    _write_jsonl(out / "floor_failures.jsonl", failures)
    tiers: dict[str, int] = {}
    for record in records:
        tiers[record["tier"]] = tiers.get(record["tier"], 0) + 1
    print(f"{len(records)} cases from {len({r['task_id'] for r in records})} tasks; tiers {tiers}; "
          f"{len(failures)} task failures -> {out}")
    return 1 if failures else 0


def join_command(args: argparse.Namespace) -> int:
    out = output_directory(args.out)
    measured, skipped = measurements(Path(args.evidence))
    practical = load_peaks(str(Path(args.practical_arch).expanduser())) if args.practical_arch else None
    joined, violations = join(read_jsonl(Path(args.floors)), measured, practical)
    _write_jsonl(out / "joined.jsonl", joined)
    _write_jsonl(out / "violations.jsonl", violations)
    _write_jsonl(out / "skipped_measurements.jsonl", skipped)
    matched = sum(1 for r in joined if "baseline_us" in r or "candidate_us" in r)
    print(f"{len(measured)} measurements, {matched} floors matched, {len(violations)} violations, "
          f"{len(skipped)} skipped -> {out}")
    return 0


REPORT_FIELDS = ("task_id", "operator", "tier", "bound", "cases", "measured_cases", "median_fraction",
                 "max_fraction", "median_practical_fraction", "max_practical_fraction", "violations")


def report_command(args: argparse.Namespace) -> int:
    out = output_directory(args.out)
    joined = read_jsonl(Path(args.joined))
    rows = report_rows(joined)
    practical_sources = sorted({r["practical_peaks"] for r in joined if r.get("practical_peaks")})
    with (out / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "violations": " ".join(row["violations"])})

    def fraction(value):
        return "" if value is None else f"{value:.0%}"
    datasheet_sources = sorted({r["peaks"]["source"] for r in joined if isinstance(r.get("peaks"), dict)})
    lines = ["# Roofline report", "",
             "Datasheet fraction = floor / measured baseline time: 100% means no headroom is left under the "
             "datasheet peaks, and only this column is a lower bound (violations use it).",
             "Practical fraction = the same counts at measured peaks / measured baseline time: how close the "
             "baseline is to what existing libraries reach; a good kernel can exceed 100%.", "",
             f"Datasheet peaks: {', '.join(datasheet_sources) or 'n/a'}. "
             f"Practical peaks: {', '.join(practical_sources) or 'none given (join --practical-arch)'}.", "",
             "| task | tier | bound | measured/cases | datasheet median | datasheet max | practical median "
             "| practical max | violations |", "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['task_id']} | {row['tier']} | {row['bound']} | {row['measured_cases']}/{row['cases']} "
                     f"| {fraction(row['median_fraction'])} | {fraction(row['max_fraction'])} "
                     f"| {fraction(row['median_practical_fraction'])} | {fraction(row['max_practical_fraction'])} "
                     f"| {' '.join(row['violations'])} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} tasks -> {out}/report.md, report.csv")
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m src.roofline", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    floor_parser = commands.add_parser("floors", help="compute the floor of every performance case")
    floor_parser.add_argument("--arch", default="mi355x_datasheet", help="bundled peak name or a peak file path")
    floor_parser.add_argument("--out", required=True, help="output directory outside the repository")
    floor_parser.add_argument("--tasks", nargs="*", help="task IDs relative to tasks/ (default: all v3 kernel tasks)")
    floor_parser.add_argument("--generate", action="store_true",
                              help="count data-dependent operators from generated inputs (needs torch and a GPU; "
                                   "on the CPU these cases keep their shape-only floor or tier none)")
    floor_parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                              help="device for --generate; only the GPU reproduces the evaluator's inputs")
    floor_parser.add_argument("--precision-probe", help="precision-probe results (kept outside the repository): "
                              "float32 products are charged at the cheapest arithmetic that passed for the case; "
                              "without it, at the float32 matrix peak")
    floor_parser.add_argument("--tasks-dir", default=str(TASKS_DIR), help=argparse.SUPPRESS)
    floor_parser.set_defaults(handler=floors)
    join_parser = commands.add_parser("join", help="join floors with measured performance evidence")
    join_parser.add_argument("--floors", required=True, help="floors.jsonl from the floors command")
    join_parser.add_argument("--evidence", required=True, help="directory searched for action-*-performance.json")
    join_parser.add_argument("--out", required=True, help="output directory outside the repository")
    join_parser.add_argument("--practical-arch", help="measured-peak file (TraceLens arch JSON, kept outside the "
                             "repository); adds t_practical_us and practical_fraction_baseline, never the floor")
    join_parser.set_defaults(handler=join_command)
    report_parser = commands.add_parser("report", help="summarize joined floors per task")
    report_parser.add_argument("--joined", required=True, help="joined.jsonl from the join command")
    report_parser.add_argument("--out", required=True, help="output directory outside the repository")
    report_parser.set_defaults(handler=report_command)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
