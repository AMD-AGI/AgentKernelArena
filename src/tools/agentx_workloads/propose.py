# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Propose a weighted ``workloads.json`` for one task from a kernel census.

    python3 -m src.tools.agentx_workloads propose tasks/<...>/<scenario> --census <capture>/kernel_census.json

1. The operator's matcher (``matchers.py``) collects the census rows whose
   shapes match the scenario's const axes; their GPU time is the operator's
   recorded time.
2. Decode shapes are padded to the graph leg's captured CUDA-graph batch sizes
   (production runs the padded shape), unless ``--no-graph-padding``.
3. Shapes are grouped: neighbours whose var-axis sizes lie within
   ``--merge-ratio`` of each other merge into one case at the member with the
   most GPU time, until at most ``--max-cases`` remain. A shape whose per-call
   time is below ``--min-case-us`` (the task's baseline time from
   ``--baseline-us`` when given, else the served kernel's) cannot be a
   performance case; it merges into a neighbour above the floor or stays
   correctness-only. Groups below ``--min-share`` of the time are dropped
   while more than the minimum number of cases remain.
4. Each case's ``weight`` is how often its group runs: the group's calls per
   second of the production leg's steady state (TP rank 0). AKA scores a
   weighted task as ``sum(w * t_baseline) / sum(w * t_candidate)``, the time the
   workload spends in the operator before and after, so the weights are
   re-applied to AKA's own baseline instead of the served kernel's times. The
   group's share of the operator's recorded GPU time, its phases and its merged
   shapes go to the case's ``observed`` record; ``weight_provenance.coverage``
   is the sum of the shares, and ``weight_provenance`` records the deployment
   (effective server settings, replay load) and the corpus coverage.
5. The task's existing correctness-only cases are kept, its former performance
   cases stay as correctness-only, and the boundary cases the contract requires
   are added when missing. The result is checked with
   ``src.task_contract.validate_task_contract``.

Writes ``<out>/<task id>.workloads.json`` (only when the proposal is valid)
and ``<out>/<task id>.report.json``; ``--write`` replaces the task's
``workloads.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import census as census_mod
from . import matchers

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.task_contract import (MAX_PERFORMANCE_CASES, MIN_PERFORMANCE_CASES, MIN_WEIGHT_COVERAGE,  # noqa: E402
                               UNALIGNED_BLOCK, validate_task_contract)
from src.task_spec import TaskConfigError  # noqa: E402

DEFAULT_MIN_CASE_US = 10.0


@dataclass
class Point:
    """One distinct var-axis shape with its recorded time."""
    axes: dict
    us: float = 0.0
    calls: float = 0.0
    phases: set = field(default_factory=set)
    sources: list = field(default_factory=list)
    extrapolated: bool = False

    @property
    def size(self) -> int:
        return math.prod(self.axes.values())

    @property
    def per_call_us(self) -> float:
        return self.us / self.calls if self.calls else 0.0


@dataclass
class Group:
    points: list

    @property
    def us(self) -> float:
        return sum(p.us for p in self.points)

    @property
    def lo(self) -> int:
        return min(p.size for p in self.points)

    @property
    def hi(self) -> int:
        return max(p.size for p in self.points)


def snap_up(value: int, sizes: list[int]) -> int:
    for s in sorted(sizes):
        if s >= value:
            return s
    return value


def collect(observations: list, graph_bs: list[int], axes: dict) -> tuple[list[Point], int]:
    """Observations -> distinct points (decode batch padded to graph sizes); count of padded observations."""
    points: dict[str, Point] = {}
    padded = 0
    for o in observations:
        values = dict(o.axes)
        if graph_bs and o.phase == "decode" and o.pad_axis in values:
            snapped = snap_up(values[o.pad_axis], graph_bs)
            padded += snapped != values[o.pad_axis]
            values[o.pad_axis] = snapped
        values = {k: min(max(v, axes[k]["min"]), axes[k]["max"]) for k, v in values.items()}
        step = {k: axes[k].get("multiple_of", 1) for k in values}
        values = {k: -(-v // step[k]) * step[k] for k, v in values.items()}
        key = json.dumps(values, sort_keys=True)
        p = points.setdefault(key, Point(values))
        p.us += o.us
        p.calls += o.calls
        p.phases.add(o.phase)
        p.sources.append(o.source)
        p.extrapolated |= o.extrapolated
    return sorted(points.values(), key=lambda p: (p.size, json.dumps(p.axes, sort_keys=True))), padded


def baseline_estimate(path: str | None):
    """Per-call baseline time by var-axis size: {size: us} from a JSON file, interpolated in log-log."""
    if not path:
        return None
    raw = json.loads(Path(path).read_text())
    pts = sorted((int(k), float(v)) for k, v in raw.items())

    def estimate(size: int) -> float:
        if size <= pts[0][0]:
            return pts[0][1]
        if size >= pts[-1][0]:
            return pts[-1][1] * size / pts[-1][0]
        for (s0, u0), (s1, u1) in zip(pts, pts[1:]):
            if s0 <= size <= s1:
                t = (math.log(size) - math.log(s0)) / (math.log(s1) - math.log(s0)) if s1 > s0 else 0.0
                return math.exp(math.log(u0) + t * (math.log(u1) - math.log(u0)))
        return pts[-1][1]
    return estimate


def representative(group: Group, per_call, floor: float) -> Point | None:
    """The member with the most GPU time among those at or above the floor."""
    eligible = [p for p in group.points if per_call(p) >= floor]
    return max(eligible, key=lambda p: (p.us, p.size)) if eligible else None


def select(points: list[Point], per_call, *, floor: float, merge_ratio: float, max_cases: int,
           min_cases: int, min_share: float) -> tuple[list[Group], list[Point], list[str]]:
    """Group points into at most ``max_cases`` performance groups; return (groups, uncovered points, log)."""
    total = sum(p.us for p in points)
    groups = [Group([p]) for p in points]
    log = []

    def can_merge(a: Group, b: Group) -> bool:
        return max(a.hi, b.hi) <= merge_ratio * min(a.lo, b.lo)

    def merge_at(i: int) -> None:
        groups[i:i + 2] = [Group(groups[i].points + groups[i + 1].points)]

    # 1. Groups without an eligible representative merge with a neighbour that has one.
    changed = True
    while changed:
        changed = False
        for i, g in enumerate(groups):
            if representative(g, per_call, floor):
                continue
            options = [j for j in (i - 1, i + 1) if 0 <= j < len(groups) and can_merge(g, groups[j])
                       and representative(groups[j], per_call, floor)]
            if options:
                j = max(options, key=lambda j: groups[j].us)
                merge_at(min(i, j))
                changed = True
                break
    uncovered = [p for g in groups if not representative(g, per_call, floor) for p in g.points]
    groups = [g for g in groups if representative(g, per_call, floor)]
    if uncovered:
        log.append("below the %.0f us floor with no eligible neighbour within %.1fx: %s" % (
            floor, merge_ratio, [p.axes for p in uncovered]))
    # 2. Merge the closest neighbours (in log size) until the case cap is met.
    while len(groups) > max_cases:
        pairs = [(math.log(groups[i + 1].lo) - math.log(groups[i].hi), i) for i in range(len(groups) - 1)
                 if can_merge(groups[i], groups[i + 1])]
        if not pairs:
            break
        merge_at(min(pairs)[1])
    # 3. Drop the smallest groups beyond the cap, then those under min_share while above the minimum count.
    groups.sort(key=lambda g: -g.us)
    while len(groups) > max_cases or (len(groups) > min_cases and total and groups[-1].us / total < min_share):
        dropped = groups.pop()
        uncovered.extend(dropped.points)
        log.append("dropped %s (%.2f%% of the time)" % ([p.axes for p in dropped.points], 100 * dropped.us / total))
    groups.sort(key=lambda g: g.lo)
    return groups, uncovered, log


def case_id(axes: dict, var: list[str]) -> str:
    return "_".join(f"{n}{axes[n]}" for n in var) if len(var) > 1 else f"{var[0][0]}{axes[var[0]]}"


def boundary_kinds(axes_values: dict, axes: dict) -> set[str]:
    kinds = set()
    for name, value in axes_values.items():
        step = axes[name].get("multiple_of", 1)
        units = value // step
        if value == axes[name]["min"]:
            kinds.add("minimum")
        if units & (units - 1):
            kinds.add("non_power_of_two")
        if units > UNALIGNED_BLOCK and units % UNALIGNED_BLOCK:
            kinds.add("unaligned")
    return kinds


def _first_units(lo: int, hi: int, ok) -> int | None:
    """Smallest unit count in [lo, hi] that satisfies ``ok``."""
    for units in range(lo, min(hi, lo + 4096) + 1):
        if ok(units):
            return units
    return None


def regenerate_boundaries(cases: list[dict], axes: dict, var: list[str]) -> list[dict]:
    """Correctness-only boundary cases the contract requires, when no correctness case provides them.

    The other var axes stay at their minimum; the first var axis carries the boundary value.
    """
    have = set()
    for c in cases:
        if c["checks"] == ["correctness"]:
            have |= boundary_kinds(c["axes"], axes)
    identities = {json.dumps(c["axes"], sort_keys=True) for c in cases}
    first = var[0]
    step = axes[first].get("multiple_of", 1)
    lo, hi = -(-axes[first]["min"] // step), axes[first]["max"] // step
    rules = {"minimum": lambda u: u * step == axes[first]["min"],
             "non_power_of_two": lambda u: u > 2 and bool(u & (u - 1)),
             "unaligned": lambda u: u > UNALIGNED_BLOCK and u % UNALIGNED_BLOCK != 0}
    added = []
    for kind, rule in rules.items():
        if kind in have:
            continue
        units = _first_units(lo, hi, rule)
        if units is None:
            continue
        values = {n: axes[n]["min"] for n in var}
        values[first] = units * step
        key = json.dumps(values, sort_keys=True)
        if key in identities:
            continue
        identities.add(key)
        added.append({"id": case_id(values, var), "axes": values, "checks": ["correctness"],
                      "source": f"boundary case ({kind.replace('_', ' ')}), added by agentx_workloads propose"})
    return added


def propose(task_dir: Path, census: dict, args: argparse.Namespace) -> dict:
    import yaml

    raw = yaml.safe_load((task_dir / "task.yaml").read_text())
    workloads = json.loads((task_dir / "workloads.json").read_text())
    task_id = task_dir.resolve().relative_to((REPO_ROOT / "tasks").resolve()).as_posix()
    axes = raw["interface"]["axes"]
    var = [n for n, d in axes.items() if d["kind"] == "var"]
    operator = raw["operator"]["name"]
    found = matchers.find(operator)
    report = {"task": task_id, "operator": operator, "census": census.get("capture_id"), "model": census.get("model"),
              "valid": False}
    if not found:
        report["error"] = f"no matcher for operator {operator!r}; register one in matchers.py"
        return report
    name, fn = found
    match = fn(axes, raw["interface"]["inputs"], census)
    graph_bs = [] if args.no_graph_padding else census.get("cuda_graph_bs") or []
    points, padded = collect(match.observations, graph_bs, axes)
    total = sum(p.us for p in points)
    clamped_us = sum(o.us for o in match.observations
                     if any(not axes[k]["min"] <= v <= axes[k]["max"] for k, v in o.axes.items()))
    steady_s = census.get("steady_state_s")
    report.update(matcher=name, unchecked_const_axes=match.unchecked_const_axes, notes=match.notes,
                  operator_gpu_ms=round(total / 1e3, 3),
                  operator_share_of_census=(round(total / 1e3 / census["total_gpu_ms"], 6)
                                            if census.get("total_gpu_ms") else None),
                  decode_observations_padded=padded, graph_bs_used=bool(graph_bs),
                  clamped_to_axis_bounds_share=round(clamped_us / total, 6) if total else 0,
                  steady_state_s=steady_s,
                  observed=[{"axes": p.axes, "share": round(p.us / total, 6) if total else 0,
                             "calls": round(p.calls, 1), "per_call_us": round(p.per_call_us, 2),
                             "phases": sorted(p.phases), "extrapolated_cost": p.extrapolated} for p in points])
    if census.get("measured_leg") == "graph":
        report["warnings"] = ["the census was measured on the CUDA-graph leg: decode kernels of graph replays "
                              "carry no shapes, so decode time is missing and coverage is overstated"]
    if census.get("speculative"):
        report.setdefault("warnings", []).append(
            "speculative decoding: decode rounds run draft-model steps and a target VERIFY step over r tokens "
            "per sequence; GEMM and token-row shapes include them (phases verify/draft)"
            + ("; decode attention of the target runs inside VERIFY (several query tokens per sequence) and is "
               "not matched as single-token decode" if name == "decode_attention" else ""))
    if not points:
        report["error"] = "no census row matches the scenario's const axes"
        return report
    base = baseline_estimate(args.baseline_us)
    per_call = (lambda p: base(p.size)) if base else (lambda p: p.per_call_us)
    report["floor"] = {"min_case_us": args.min_case_us,
                       "per_call_time": "task baseline (--baseline-us)" if base else
                       "served kernel per call (census); confirm with the QA auditor's baseline timing"}
    groups, uncovered, log = select(points, per_call, floor=args.min_case_us, merge_ratio=args.merge_ratio,
                                    max_cases=args.max_cases, min_cases=MIN_PERFORMANCE_CASES,
                                    min_share=args.min_share)
    report["selection_log"] = log
    coverage = sum(g.us for g in groups) / total
    report["coverage"] = round(coverage, 4)
    report["extrapolated_cost_share"] = round(sum(p.us for p in points if p.extrapolated) / total, 4)
    perf_cases = []
    for g in groups:
        rep = representative(g, per_call, args.min_case_us)
        members = sorted(g.points, key=lambda p: p.size)
        phases = sorted({ph for p in members for ph in p.phases})
        span = (f"{members[0].axes} .. {members[-1].axes}" if len(members) > 1 else f"{rep.axes}")
        calls = sum(p.calls for p in members)
        rate = calls / steady_s if steady_s else calls
        perf_cases.append({
            "id": case_id(rep.axes, var), "axes": rep.axes, "checks": ["correctness", "performance"],
            "weight": max(float(f"{rate:.4g}"), 1e-6),
            "observed": {"phases": phases, "time_share": max(round(g.us / total, 4), 0.0001),
                         "members": [p.axes for p in members]},
            "source": f"AgentX {'/'.join(phases)}: {len(members)} recorded shape(s) {span}, "
                      f"{100 * g.us / total:.1f}% of the operator's GPU time"
                      + (" (per-call cost extrapolated beyond the measured shapes)" if rep.extrapolated else "")})
    report["performance_cases"] = [{"axes": c["axes"], "weight": c["weight"],
                                    "time_share": c["observed"]["time_share"],
                                    "members": [p.axes for p in g.points]}
                                   for c, g in zip(perf_cases, groups)]
    chosen = {json.dumps(c["axes"], sort_keys=True) for c in perf_cases}
    ids = {c["id"] for c in perf_cases}
    correctness = []
    for c in workloads["cases"]:  # keep every existing shape as a correctness case
        key = json.dumps(c["axes"], sort_keys=True)
        if key in chosen:
            continue
        kept = {k: v for k, v in c.items() if k not in ("weight", "observed")}
        kept["checks"] = ["correctness"]
        while kept["id"] in ids:
            kept["id"] += "_c"
        ids.add(kept["id"])
        chosen.add(key)
        correctness.append(kept)
    for p in uncovered:  # real shapes that cannot be performance cases stay correctness-only
        key = json.dumps(p.axes, sort_keys=True)
        if key in chosen:
            continue
        cid = case_id(p.axes, var)
        while cid in ids:
            cid += "_c"
        ids.add(cid)
        chosen.add(key)
        correctness.append({"id": cid, "axes": p.axes, "checks": ["correctness"],
                            "source": f"AgentX {'/'.join(sorted(p.phases))}, not a performance case "
                                      f"({100 * p.us / total:.2f}% of the operator's GPU time)"})
    cases = sorted(perf_cases + correctness, key=lambda c: (math.prod(c["axes"].values()), c["id"]))
    for extra in regenerate_boundaries(cases, axes, var):
        while extra["id"] in ids:
            extra["id"] += "_b"
        ids.add(extra["id"])
        cases.append(extra)
    cases.sort(key=lambda c: (math.prod(c["axes"].values()), c["id"]))
    corpus = census.get("corpus", {})
    proposal = {
        "description": (f"{raw['operator']['name']} shapes of the AgentX agentic serving corpus replayed on "
                        f"{census.get('model')} (TP {census.get('tp')}); each performance case is weighted by how "
                        f"often its shapes run (calls per second of the steady state) and the task scores the "
                        f"ratio of total times. The task's earlier shapes and the boundary shapes are "
                        f"correctness-only."),
        "weight_provenance": {
            "kind": "agentx",
            "source": f"{corpus.get('id')}@{corpus.get('revision')}",
            "capture": census.get("capture_id", "unknown"),
            "model": census.get("model", "unknown"),
            "coverage": math.floor(coverage * 1000) / 1000,
            "weight_unit": ("calls per second of the steady state, TP rank 0" if steady_s
                            else "calls over the recorded steady state, TP rank 0"),
            "description": (f"{census.get('scope')}; "
                            + (f"GEMM time = {census['gemm_log_weighting']['method']}; "
                               if census.get("gemm_log_weighted") and name == "gemm" else "")
                            + f"image {census.get('image')}; matcher {name}; "
                            "decode shapes " + ("padded to captured CUDA-graph batch sizes" if graph_bs
                                                else "unpadded") + "."),
        },
        "cases": cases,
    }
    for key in ("deployment", "corpus_coverage"):
        record = {k: census_mod.model_ref(v) if k in census_mod.MODEL_PATH_KEYS else v
                  for k, v in (census.get(key) or {}).items()
                  if v is None or type(v) in (str, int, float, bool)}
        if record:
            proposal["weight_provenance"]["corpus" if key == "corpus_coverage" else key] = record
    report["proposal"] = proposal
    if not MIN_PERFORMANCE_CASES <= len(perf_cases) <= MAX_PERFORMANCE_CASES:
        report["error"] = (f"{len(perf_cases)} performance cases; the contract needs "
                           f"{MIN_PERFORMANCE_CASES}-{MAX_PERFORMANCE_CASES}")
    elif coverage < MIN_WEIGHT_COVERAGE:
        report["error"] = f"coverage {coverage:.3f} is below {MIN_WEIGHT_COVERAGE}"
    else:
        try:
            validate_task_contract(raw, proposal, task_id=task_id)
            report["valid"] = True
        except TaskConfigError as exc:
            report["error"] = f"task contract: {exc}"
    return report


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("task_dir", help="tasks/<baseline_language>/<category>/<operator>/<scenario>")
    ap.add_argument("--census", required=True, help="kernel_census.json from the census step")
    ap.add_argument("--out", help="output directory (default: <census dir>/proposals)")
    ap.add_argument("--max-cases", type=int, default=MAX_PERFORMANCE_CASES)
    ap.add_argument("--merge-ratio", type=float, default=1.5,
                    help="largest var-axis size ratio inside one merged case")
    ap.add_argument("--min-case-us", type=float, default=DEFAULT_MIN_CASE_US,
                    help="per-call time floor of a performance case")
    ap.add_argument("--baseline-us", help="JSON {var-axis size: task baseline us per call} for the floor")
    ap.add_argument("--min-share", type=float, default=0.005, help="drop groups below this share of the time")
    ap.add_argument("--no-graph-padding", action="store_true", help="keep decode shapes unpadded")
    ap.add_argument("--write", action="store_true", help="replace the task's workloads.json with a valid proposal")


def main(a: argparse.Namespace) -> int:
    task_dir = Path(a.task_dir)
    census_path = Path(a.census)
    census = json.loads(census_path.read_text())
    report = propose(task_dir, census, a)
    out = Path(a.out) if a.out else census_path.parent / "proposals"
    out.mkdir(parents=True, exist_ok=True)
    stem = report["task"].replace("/", "__")
    (out / f"{stem}.report.json").write_text(json.dumps(report, indent=1) + "\n")
    if report["valid"]:
        text = json.dumps(report["proposal"], indent=1) + "\n"
        (out / f"{stem}.workloads.json").write_text(text)
        if a.write:
            (task_dir / "workloads.json").write_text(text)
    summary = ", ".join(f"{c['axes']}:{c['weight']}/s ({100 * c['time_share']:.1f}%)"
                        for c in report.get("performance_cases", []))
    print(f"propose {report['task']}: valid={report['valid']} coverage={report.get('coverage')} [{summary}]"
          + (f" error: {report['error']}" if report.get("error") else ""))
    return 0 if report["valid"] else 1
