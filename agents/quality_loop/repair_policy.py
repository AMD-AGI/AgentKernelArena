"""Controller-owned invariants for task maintenance, separate from optimization.

A repair may make a schema-v3 task's checks stronger; it may never change what
is scored. The interface, reference declaration, baseline (language, entry,
provenance and vendored sources), operator, platform, timeout and every
original workload case stay byte-for-byte equivalent. Repairs may add
correctness-only cases, edit inputs.py, reference.py, README.md or the
description, add a baseline LICENSE, and tighten an error bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from agents.qa_auditor.rubric import BLOCKING
from src.task_contract import BASELINE_DIR, TASK_FILE, load_task_contract
from src.task_spec import TaskConfigError


class RepairRejected(RuntimeError):
    """A proposed repair changes the original evaluation contract."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def finding_key(finding: dict) -> str:
    """Stable across reports: the same check failing for the same reason."""
    digest = hashlib.sha256(_json([finding.get("id"), finding.get("detail", "")]).encode()).hexdigest()
    return f"{finding.get('id')}:{digest[:16]}"


def blocking_findings(report: dict, warning_checks: tuple[str, ...] = ()) -> dict[str, dict]:
    """Every blocking QA finding plus warnings whose check id was opted in."""
    result = {}
    for finding in report.get("findings", []):
        if not isinstance(finding, dict):
            continue
        if finding.get("severity") != BLOCKING and finding.get("id") not in warning_checks:
            continue
        result[finding_key(finding)] = {
            key: finding.get(key) for key in ("id", "severity", "title", "detail", "evidence", "fix")}
    return result


# Fields of task.yaml a repair may not change. ``description`` is prose and
# ``compare`` may only tighten; everything else decides what is scored.
_LOCKED_FIELDS = ("schema_version", "operator", "interface", "reference", "baseline", "element_order",
                  "workloads", "platform_support", "timeout_s")
_DEFAULTS = {"k": 2.0, "max_violation_fraction": 0.0, "baseline_min_sqnr_db": 25.0}
# Parameters where a smaller value is a stricter bound; ``min_db`` and the
# baseline sanity floor are stricter when larger.
_SMALLER_IS_TIGHTER = {"k", "max_violation_fraction", "atol", "rtol"}
_LARGER_IS_TIGHTER = {"baseline_min_sqnr_db", "min_db"}


def compare_tightened(original: dict, proposed: dict) -> bool:
    """True when ``proposed`` accepts no output that ``original`` rejects.

    Only decidable cases count: ``exact`` is the strictest bound, otherwise the
    bound kind must stay the same and every parameter must move toward
    strictness. Anything else (for example another bound kind or a changed
    dot_product length) is not a provable tightening.
    """
    if _json(original) == _json(proposed):
        return True
    if proposed.get("bound") == "exact":
        return True
    if proposed.get("bound") != original.get("bound"):
        return False
    for key in set(original) | set(proposed):
        if key == "bound":
            continue
        before = original.get(key, _DEFAULTS.get(key))
        after = proposed.get(key, _DEFAULTS.get(key))
        if before == after:
            continue
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (before, after)):
            return False
        if key in _SMALLER_IS_TIGHTER and after <= before:
            continue
        if key in _LARGER_IS_TIGHTER and after >= before:
            continue
        return False
    return True


def _baseline_sources(task_dir: Path) -> dict[str, str]:
    root = Path(task_dir) / BASELINE_DIR
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(task_dir).as_posix()
        if "__pycache__" in path.parts or path.name.startswith("LICENSE"):
            continue
        if path.is_symlink() or path.is_file():
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


@dataclass(frozen=True)
class RepairPolicy:
    task_id: str
    locked: str
    compare: str
    cases: str
    baseline: str

    @staticmethod
    def _load(task_dir: Path, task_id: str):
        try:
            return load_task_contract(Path(task_dir) / TASK_FILE, task_id=task_id)
        except (TaskConfigError, OSError, ValueError) as exc:
            raise RepairRejected(f"Repair proposed an invalid task contract: {exc}") from exc

    @classmethod
    def capture(cls, task_dir: Path, task_id: str) -> "RepairPolicy":
        contract = cls._load(task_dir, task_id)
        raw = contract.raw
        cases = {}
        for case in contract.workloads["cases"]:
            cases[case["id"]] = {"axes": case["axes"], "scalars": case.get("scalars", {}),
                                 "checks": case["checks"]}
        return cls(task_id, _json({key: raw.get(key) for key in _LOCKED_FIELDS}), _json(raw["compare"]),
                   _json(cases), _json(_baseline_sources(task_dir)))

    def verify(self, task_dir: Path) -> None:
        proposed = self.capture(task_dir, self.task_id)
        if proposed.locked != self.locked:
            raise RepairRejected("Task repair changed the scored task contract (interface, reference, "
                                 "baseline, operator, platform or timeout)")
        if proposed.baseline != self.baseline:
            raise RepairRejected("Task repair changed the vendored baseline sources")
        original_compare, updated_compare = json.loads(self.compare), json.loads(proposed.compare)
        for name, declaration in original_compare.items():
            if not compare_tightened(declaration, updated_compare.get(name, {})):
                raise RepairRejected(f"Task repair loosened or replaced the error bound of {name}")
        original, updated = json.loads(self.cases), json.loads(proposed.cases)
        for name, identity in original.items():
            if _json(updated.get(name)) != _json(identity):
                raise RepairRejected(f"Task repair removed or changed an original case: {name}")
        for name in updated.keys() - original.keys():
            if "performance" in updated[name]["checks"]:
                raise RepairRejected("Task repair added scored workloads; use a separately reviewed migration")
