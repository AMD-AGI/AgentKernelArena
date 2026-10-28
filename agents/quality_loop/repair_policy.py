"""Controller-owned invariants for task maintenance, separate from optimization."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from src.task_protocol import IDENTITY_FIELDS
from src.task_spec import TaskSpec


class RepairRejected(RuntimeError):
    """A proposed repair changes the original evaluation contract."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def blocking_findings(report: dict, warning_checks: tuple[str, ...] = ()) -> dict[str, dict]:
    """Stable IDs retain the original finding across successive repair reviews."""
    result = {}
    for name, check in report.get("checks", {}).items():
        status = check.get("status")
        if status not in ("FAIL", "TIMEOUT") and not (status == "WARN" and name in warning_checks):
            continue
        finding = {"check": name, "status": status, "details": check.get("details", ""),
                   "evidence": check.get("evidence", [])}
        digest = hashlib.sha256(_json(finding).encode()).hexdigest()[:16]
        result[f"{name}:{digest}"] = finding
    return result


@dataclass(frozen=True)
class RepairPolicy:
    specification: str
    cases: str
    timing: str

    @classmethod
    def capture(cls, spec: TaskSpec, cases, report: dict | None = None) -> "RepairPolicy":
        if not cases:
            raise RuntimeError("Task repair requires an independent original case manifest")
        identities = {}
        for row in cases:
            name = row["test_case_id"]
            if name in identities:
                raise RuntimeError("Duplicate case in original repair manifest")
            identities[name] = {k: row[k] for k in IDENTITY_FIELDS if k in row}
            identities[name]["checks"] = sorted(row["checks"])
        policy = {}
        rows = (report or {}).get("checks", {}).get("performance", {}).get("result", {}).get("cases", [])
        fixed = {"benchmark_method", "reference_benchmark_method", "benchmark_samples",
                 "benchmark_warmup", "benchmark_target_ms", "benchmark_max_repeats",
                 "benchmark_cache_policy", "benchmark_cache_flush_bytes",
                 "benchmark_cache_flush_in_timing", "benchmark_metric"}
        for row in rows:
            values = {}
            meta = row.get("metadata", {})
            for container in (row, meta, meta.get("device_timing", {})):
                for key in fixed & container.keys():
                    if key in values and _json(values[key]) != _json(container[key]):
                        raise RuntimeError(f"Conflicting captured timing policy: {key}")
                    values[key] = container[key]
            policy[row["test_case_id"]] = values
        return cls(_json(spec.to_mapping()), _json(identities), _json(policy))

    def verify(self, spec: TaskSpec, cases, report: dict | None = None) -> None:
        proposed = self.capture(spec, cases, report)
        self.verify_specification(spec)
        original = json.loads(self.cases)
        updated = json.loads(proposed.cases)
        for name, identity in original.items():
            if _json(updated.get(name)) != _json(identity):
                raise RepairRejected(f"Task repair removed or changed an original case: {name}")
        for name in updated.keys() - original.keys():
            if "performance" in updated[name]["checks"]:
                raise RepairRejected("Task repair added scored workloads; use a separately reviewed migration")
        timing = json.loads(proposed.timing)
        for name, policy in json.loads(self.timing).items():
            if name not in timing and (report or {}).get("initial_validation_gate") == "FAIL":
                continue  # A failed earlier action legitimately prevents timing.
            if _json(timing.get(name)) != _json(policy):
                raise RepairRejected(f"Task repair changed captured benchmark policy: {name}")

    def verify_specification(self, spec: TaskSpec) -> None:
        if _json(spec.to_mapping()) != self.specification:
            raise RepairRejected("Task repair changed the original task specification")
