"""Join floors with measured performance evidence and summarize per task.

Measurements come from task-session action records (``action-NNNN-<role>-performance.json``)
whose session directory holds ``session.json`` with the task ID. Only cases timed with a cold-cache
method (``cuda_graph_cold_cache_amortized``, the evaluator's current one, or the earlier single-call
``cuda_graph_cold_cache``) are joined: both evict the caches before every call, so every byte comes
from HBM and the HBM bandwidth applies. The single-call method also charges the fixed cost of one
graph replay to every call, so it overstates short operators; each joined record keeps its method.
Other methods are skipped with a reason.

A measurement more than ``VIOLATION_MARGIN`` faster than its floor is a violation, and its role
decides what it means. A baseline is a reviewed reference implementation, so a baseline violation
means the operator model is wrong: every floor of that model is downgraded to tier ``none`` and the
model goes on the to-investigate list. A candidate violation means the candidate is suspect (a
broken timing, skipped work, or a model to recheck): it is listed with ``role: candidate`` and the
floor that flagged it stays.

A floor counted from generated inputs (``floors --generate``) records the runtime it ran in. When a
measurement's session records a different torch build or GPU (``runtime_identity.json``), the
generated inputs need not be the timed ones; the joined record and any violation say so.

Measured (practical) peaks, for example the TraceLens microbench result kept outside the
repository, only add a second column: ``t_practical_us`` is the same FLOP and byte counts at the
measured peaks, the time existing libraries would need. A good kernel can beat it, so it is never
used for the floor, the tier or violations.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from .arch import Peaks
from .generated import GENERATION_KEYS

COLD_CACHE = "cuda_graph_cold_cache"
AMORTIZED_COLD_CACHE = "cuda_graph_cold_cache_amortized"
COLD_CACHE_METHODS = (AMORTIZED_COLD_CACHE, COLD_CACHE)
VIOLATION_MARGIN = 0.95
ACTION_FILE = re.compile(r"action-\d+-(?P<role>baseline|candidate)-performance\.json$")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def measurements(evidence_root: Path) -> tuple[list[dict], list[dict]]:
    """(rows, skipped): one row per timed case found under ``evidence_root``."""
    rows, skipped = [], []
    for path in sorted(Path(evidence_root).rglob("action-*-performance.json")):
        match = ACTION_FILE.search(path.name)
        session = path.parent / "session.json"
        if not match or not session.is_file():
            skipped.append({"file": str(path), "reason": "no session.json beside the action record"})
            continue
        task_id = json.loads(session.read_text(encoding="utf-8"))["task_id"]
        runtime_file = path.parent / "runtime_identity.json"
        runtime = json.loads(runtime_file.read_text(encoding="utf-8")) if runtime_file.is_file() else None
        record = json.loads(path.read_text(encoding="utf-8"))
        result = record.get("result") or {}
        if result.get("status") != "PASS":
            skipped.append({"file": str(path), "task_id": task_id, "reason": f"status {result.get('status')}"})
            continue
        for case in result.get("cases", []):
            method = case.get("benchmark_method")
            if method not in COLD_CACHE_METHODS:
                skipped.append({"file": str(path), "task_id": task_id, "case_id": case.get("test_case_id"),
                                "reason": f"benchmark_method {method!r} is not a cold-cache method"})
                continue
            rows.append({"task_id": task_id, "case_id": case["test_case_id"], "role": match["role"],
                         "us": float(case["execution_time_ms"]) * 1e3, "method": method, "file": str(path),
                         "runtime": runtime})
    return rows, skipped


def practical_time_us(floor: dict, peaks: Peaks) -> float | None:
    """The floor's FLOPs and bytes at the given (measured) peaks, in microseconds.

    None when a peak the case needs is missing or zero: the microbench records a metric it could
    not measure as 0.0.
    """
    def seconds(amount: float, rate: float) -> float | None:
        if not amount:
            return 0.0
        return amount / rate if rate > 0 else None

    def matrix_rate(key: str) -> float:
        try:
            return peaks.matrix_flops_per_s(key)  # also resolves emulated keys such as matrix_bf16x3
        except KeyError:
            return 0.0

    by_precision = floor.get("flops_matrix_by_precision") or {}
    matrix = [seconds(flops, matrix_rate(key)) for key, flops in by_precision.items()]
    vector_rates = [value for key, value in peaks.tflops.items() if key.startswith("vector_") and value > 0]
    parts = [*matrix,
             seconds(floor.get("flops_vector", 0), max(vector_rates, default=0.0) * 1e12),
             seconds(floor.get("bytes_min", 0), peaks.mem_bw_gbps * 1e9)]
    if any(part is None for part in parts):
        return None
    return max(sum(matrix), parts[-2], parts[-1]) * 1e6


def generation_mismatch(floor: dict, runtime: dict | None) -> dict[str, list] | None:
    """The runtime fields in which a generated floor's inputs and a measurement differ, or None."""
    generation = floor.get("generation")
    if not generation or not runtime:
        return None
    differ = {key: [generation.get(key), runtime.get(key)] for key in GENERATION_KEYS
              if generation.get(key) != runtime.get(key)}
    return differ or None


def join(floors: Iterable[dict], measured: Iterable[dict],
         practical: Peaks | None = None) -> tuple[list[dict], list[dict]]:
    """(joined, violations). A case's fastest measurement per role is used; only baseline violations
    downgrade a model. With ``practical`` peaks each record also gets ``t_practical_us`` and, when
    measured, ``practical_fraction_baseline``."""
    best: dict[tuple[str, str, str], dict] = {}
    for row in measured:
        key = (row["task_id"], row["case_id"], row["role"])
        if key not in best or row["us"] < best[key]["us"]:
            best[key] = row
    joined, violations = [], []
    for floor in floors:
        record = dict(floor)
        if practical is not None:
            record["practical_peaks"] = practical.source
            record["t_practical_us"] = practical_time_us(floor, practical)
            hit = best.get((floor["task_id"], floor["case_id"], "baseline"))
            if hit is not None and record["t_practical_us"] is not None:
                record["practical_fraction_baseline"] = record["t_practical_us"] / hit["us"]
        for role in ("baseline", "candidate"):
            hit = best.get((floor["task_id"], floor["case_id"], role))
            if hit is None:
                continue
            record[f"{role}_us"], record[f"{role}_method"] = hit["us"], hit["method"]
            record[f"floor_fraction_{role}"] = floor["t_floor_us"] / hit["us"]
            mismatch = generation_mismatch(floor, hit.get("runtime"))
            if mismatch:
                record[f"generation_mismatch_{role}"] = mismatch
                record["notes"] = [*record["notes"], f"generated_inputs_runtime_differs_from_{role}: "
                                   + ", ".join(sorted(mismatch))]
            if hit["us"] < VIOLATION_MARGIN * floor["t_floor_us"]:
                violation = {"task_id": floor["task_id"], "case_id": floor["case_id"], "role": role,
                             "model": floor["model"], "measured_us": hit["us"],
                             "t_floor_us": floor["t_floor_us"], "file": hit["file"]}
                if mismatch:
                    violation["generation_mismatch"] = mismatch
                violations.append(violation)
        joined.append(record)
    broken = {v["model"] for v in violations if v["role"] == "baseline"}
    for record in joined:
        if record["model"] in broken and record["tier"] != "none":
            record["tier"] = "none"
            record["notes"] = [*record["notes"], "downgraded: a baseline measurement beat a floor of this model"]
    return joined, violations


def report_rows(joined: Iterable[dict]) -> list[dict[str, Any]]:
    tasks: dict[str, list[dict]] = defaultdict(list)
    for record in joined:
        tasks[record["task_id"]].append(record)
    rows = []
    for task_id, records in sorted(tasks.items()):
        fractions = [r["floor_fraction_baseline"] for r in records if "floor_fraction_baseline" in r]
        practical = [r["practical_fraction_baseline"] for r in records if "practical_fraction_baseline" in r]
        rows.append({
            "task_id": task_id, "operator": records[0]["operator"],
            "tier": min((r["tier"] for r in records), key=("none", "estimated", "verified").index),
            "bound": statistics.mode(r["bound"] for r in records),
            "cases": len(records), "measured_cases": len(fractions),
            "median_fraction": statistics.median(fractions) if fractions else None,
            "max_fraction": max(fractions) if fractions else None,
            "median_practical_fraction": statistics.median(practical) if practical else None,
            "max_practical_fraction": max(practical) if practical else None,
            "violations": [r["case_id"] for r in records if r.get("floor_fraction_baseline", 0) > 1 / VIOLATION_MARGIN],
            "candidate_violations": [r["case_id"] for r in records
                                     if r.get("floor_fraction_candidate", 0) > 1 / VIOLATION_MARGIN],
        })
    return rows
