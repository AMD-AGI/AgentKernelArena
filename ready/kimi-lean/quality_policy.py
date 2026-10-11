"""Conservative admission of raw timing comparisons; never trim or retime data.

Call only after the task's trusted validator has established complete cases,
source/request identities, and (for variable work) the realized work classes.
This is a catastrophic-instability gate, not a statistical proof of speedup.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
import statistics

SCHEMA = "benchmark-quality-v1"
POLICY = {"expected_samples": 100, "max_over_median": 10.0,
          "single_sample_share": 0.10, "p95_over_p05": 10.0}
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(value, message):
    if not value:
        raise ValueError("Benchmark quality: " + message)


def fingerprint(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def source_identity(value):
    """Require executable-source identity, never package/harness identity."""
    if isinstance(value, str):
        require(SHA256.fullmatch(value), "invalid executable-source digest")
        return value
    require(isinstance(value, dict) and bool(value), "executable-source identity is missing")
    require(all(isinstance(k, str) and k and isinstance(v, str) and SHA256.fullmatch(v)
                for k, v in value.items()), "invalid executable-source map")
    return dict(sorted(value.items()))


def _statistics(values):
    ordered = sorted(values)
    n = len(values)
    total = math.fsum(values)
    median = statistics.median(ordered)
    p05 = ordered[math.ceil(0.05 * n) - 1]
    p95 = ordered[math.ceil(0.95 * n) - 1]
    maximum = max(values)
    reasons = []
    if maximum / median > POLICY["max_over_median"] and maximum / total > POLICY["single_sample_share"]:
        reasons.append("dominant_extreme_replay")
    if p95 / p05 > POLICY["p95_over_p05"]:
        reasons.append("extreme_replay_spread")
    return {"sample_count": n, "mean_ms": total / n, "median_ms": median,
            "min_ms": min(values), "max_ms": maximum, "p05_ms": p05, "p95_ms": p95,
            "max_over_median": maximum / median, "max_share_of_sum": maximum / total,
            "p95_over_p05": p95 / p05, "reasons": reasons}


def assess_series(samples, *, work_ids=None):
    """Inspect every sample; grouping affects admission only, never the mean.

    work_ids must come from validated work controls, e.g. histogram variant IDs
    or KV lengths. Seeds, addresses, and report-provided arbitrary labels are
    not workload classes. A singleton class cannot reveal repeat instability;
    its count is explicit in the receipt rather than mixed with unrelated work.
    """
    require(isinstance(samples, list) and len(samples) == POLICY["expected_samples"],
            "all 100 raw samples are required")
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples),
            "raw samples must be finite and positive")
    if work_ids is None:
        groups = {"fixed_work": list(range(len(samples)))}
    else:
        require(isinstance(work_ids, list) and len(work_ids) == len(samples),
                "one validated work class is required per sample")
        require(all(isinstance(v, str) and v for v in work_ids), "invalid work class")
        groups = {}
        for index, work in enumerate(work_ids):
            groups.setdefault(work, []).append(index)
    rows = []
    for work, indices in groups.items():
        stats = _statistics([samples[i] for i in indices])
        rows.append({"work_id": work, "sample_indices": indices, **stats})
    return {"status": "pass" if all(not r["reasons"] for r in rows) else "reject",
            "sample_count": len(samples), "raw_mean_ms": math.fsum(samples) / len(samples),
            "raw_samples_sha256": fingerprint(samples), "work_classes": rows,
            "singleton_work_classes": sum(r["sample_count"] == 1 for r in rows)}


def assess_comparison(cases, *, reference_source, candidate_source):
    """Gate a complete validated comparison, retaining all case/sample means.

    Each case contains case_id, reference_samples_ms and candidate_samples_ms.
    Variable cases additionally require identical reference_work_ids and
    candidate_work_ids, reconstructed by the existing protected receipt check.
    Fixed cases explicitly declare work_kind='fixed'. Unknown/missing kinds
    reject rather than silently applying a fixed-work assumption.
    """
    before, after = source_identity(reference_source), source_identity(candidate_source)
    require(isinstance(cases, list) and bool(cases), "complete cases are required")
    names = [case.get("case_id") for case in cases]
    require(all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names),
            "case identities are missing or duplicated")
    rows = []
    for case in cases:
        kind = case.get("work_kind")
        require(kind in ("fixed", "paired_variable"), "work kind is missing or unsupported")
        work_ids = None
        if kind == "paired_variable":
            left, right = case.get("reference_work_ids"), case.get("candidate_work_ids")
            require(isinstance(left, list) and left == right,
                    "variable-work schedules are not paired")
            work_ids = left
        left = assess_series(case.get("reference_samples_ms"), work_ids=work_ids)
        right = assess_series(case.get("candidate_samples_ms"), work_ids=work_ids)
        rows.append({"case_id": case["case_id"], "work_kind": kind,
                     "reference": left, "candidate": right,
                     "raw_speedup": left["raw_mean_ms"] / right["raw_mean_ms"],
                     "status": "pass" if left["status"] == right["status"] == "pass" else "reject"})
    quality_pass = all(row["status"] == "pass" for row in rows)
    changed = before != after
    raw_ratio = math.fsum(row["raw_speedup"] for row in rows) / len(rows)
    accepted_ratio = raw_ratio if quality_pass and changed else None
    return {"schema": SCHEMA, "policy": dict(POLICY),
            "status": "pass" if quality_pass else "reject",
            "comparison_status": ("rejected_timing_quality" if not quality_pass else
                                  "source_changed" if changed else "unchanged_source_control"),
            "source_changed": changed, "reference_source": before, "candidate_source": after,
            "gain_eligible": quality_pass and changed,
            "accepted_gain": accepted_ratio is not None and accepted_ratio > 1,
            "accepted_arithmetic_mean_speedup": accepted_ratio,
            "raw_arithmetic_mean_speedup": raw_ratio, "cases": rows,
            "repeat_policy": "No automatic retries. Any authorized repeat is one fresh complete comparison; retain every prior result."}


def gate_native_measurement(measurement, reference_report, candidate_report):
    """Native quant adapter, after existing full report/identity validation."""
    result = deepcopy(measurement)
    require(result.get('status') == 'measured', 'an unsuccessful measurement cannot become scoreable')
    report_rows = [report.get("paired_cases", []) for report in (reference_report, candidate_report)]
    names = [row["test_case_id"] for row in result["cases"]]
    require(all([row.get("case_id") for row in rows] == names for rows in report_rows),
            "quality gate requires both complete case sets")
    comparison = [{"case_id": name, "work_kind": "fixed",
        "reference_samples_ms": left["timings"]["candidate_native"]["samples_ms"],
        "candidate_samples_ms": right["timings"]["candidate_native"]["samples_ms"]}
        for name, left, right in zip(names, *report_rows)]
    quality = assess_comparison(comparison, reference_source=result["reference_source_sha256"],
                               candidate_source=result["candidate_source_sha256"])
    # The production diagnostics are also measured fixed work; a broken native
    # diagnostic series invalidates the comparison, not just its own case.
    diagnostics = []
    for outer, rows in zip(("reference", "candidate"), report_rows):
        for row in rows:
            diagnostics.append({"outer_leg": outer, "case_id": row["case_id"],
                **assess_series(row["timings"]["production_native"]["samples_ms"])})
    quality["production_diagnostics"] = diagnostics
    for checked in diagnostics:
        row = next(row for row in result['cases'] if row['test_case_id'] == checked['case_id'])
        if 'production_diagnostic_ms' in row:
            require(math.isclose(row['production_diagnostic_ms'][checked['outer_leg'] + '_run'],
                                 checked['raw_mean_ms'], rel_tol=1e-13),
                    'production diagnostic mean differs from retained samples')
    if any(row["status"] == "reject" for row in diagnostics):
        quality.update(status="reject", comparison_status="rejected_timing_quality", gain_eligible=False,
                       accepted_gain=False, accepted_arithmetic_mean_speedup=None)
    require(math.isclose(result["arithmetic_mean_speedup"], quality["raw_arithmetic_mean_speedup"],
                         rel_tol=1e-13), "raw score differs from retained samples")
    for row, checked in zip(result["cases"], quality["cases"]):
        require(math.isclose(row['reference_ms'], checked['reference']['raw_mean_ms'], rel_tol=1e-13)
                and math.isclose(row['candidate_ms'], checked['candidate']['raw_mean_ms'], rel_tol=1e-13),
                'raw case means differ from retained samples')
        require(math.isclose(row["speedup"], checked["raw_speedup"], rel_tol=1e-13),
                "raw case score differs from retained samples")
        row["raw_speedup"] = row["speedup"]
        if not quality["gain_eligible"]:
            row["speedup"] = None
    result["raw_arithmetic_mean_speedup"] = result["arithmetic_mean_speedup"]
    result["arithmetic_mean_speedup"] = quality["accepted_arithmetic_mean_speedup"]
    result["benchmark_quality"] = quality
    result["gain_eligible"] = quality["gain_eligible"]
    result["accepted_gain"] = quality["accepted_gain"]
    if quality["status"] != "pass":
        result["status"] = "rejected_timing_quality"
    return result
