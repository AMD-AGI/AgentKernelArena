"""Deterministic validation of task-owned arena-eval-v1 command evidence.

The task chooses its numerical comparison. The framework verifies execution,
case coverage, identity, timing fields, and lifecycle policy independently.
No task family, agent name, or success-looking log message changes acceptance.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Any, Iterable

from .task_spec import ACTIONS, BaselineSpec


RESULT_PREFIX = "ARENA_EVAL_RESULT="
PROTOCOL = "arena-eval-v1"
IDENTITY_FIELDS = ("shape", "dtype", "params")
TIMING_METHODS = {"cuda_graph", "cuda_event_fallback"}


class TaskProtocolError(ValueError):
    """An action's output is missing, inconsistent, or invalid."""


def _fail(message: str) -> None:
    raise TaskProtocolError(message)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            _fail(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _text(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{where} must be a nonempty string")


def _status(value: Any, where: str) -> None:
    if not isinstance(value, str) or value not in ("PASS", "FAIL"):
        _fail(f"{where} must be PASS or FAIL")


def _validate_json_numbers(value: Any) -> None:
    # parse_constant rejects literal NaN/Infinity; 1e999 also overflows to inf.
    if isinstance(value, float) and not math.isfinite(value):
        _fail("Report contains a nonfinite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _validate_json_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json_numbers(item)


def _identity(row: dict) -> str:
    return json.dumps({key: row[key] for key in IDENTITY_FIELDS if key in row},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ActionResult:
    role: str
    action: str
    status: str
    cases: tuple[dict, ...]
    reason: str | None = None
    failure_kind: str | None = None
    metadata: dict | None = None

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_mapping(self) -> dict:
        obj = {"protocol": PROTOCOL, "role": self.role, "action": self.action,
               "status": self.status, "cases": deepcopy(list(self.cases))}
        for key in ("reason", "failure_kind", "metadata"):
            if (value := getattr(self, key)) is not None:
                obj[key] = deepcopy(value)
        return obj


def parse_command_result(stdout: str, *, role: str, action: str, returncode: int) -> ActionResult:
    """Parse ONLY this completed process's stdout, never an old workspace file."""
    if (role, action) not in ACTIONS:
        _fail(f"Unsupported action: {role}.{action}")
    lines = [line[len(RESULT_PREFIX):] for line in stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if len(lines) != 1:
        _fail(f"Expected exactly one {RESULT_PREFIX} line; found {len(lines)}")
    try:
        obj = json.loads(lines[0], object_pairs_hook=_object_pairs,
                         parse_constant=lambda value: _fail(f"Invalid JSON constant: {value}"))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise TaskProtocolError(f"Malformed command report: {exc}") from exc
    if not isinstance(obj, dict):
        _fail("Command report must be an object")
    _validate_json_numbers(obj)
    unknown = set(obj) - {"protocol", "role", "action", "status", "cases", "reason", "failure_kind", "metadata"}
    if unknown:
        _fail(f"Unknown command report fields: {sorted(unknown)}")
    if obj.get("protocol") != PROTOCOL or obj.get("role") != role or obj.get("action") != action:
        _fail("Command report protocol/role/action does not match invocation")
    _status(obj.get("status"), "status")
    passed = obj["status"] == "PASS"
    if (returncode == 0) != passed:
        _fail("Command exit code contradicts report status")
    if returncode < 0:
        _fail("Command was terminated by a signal")
    if not passed:
        _text(obj.get("reason"), "failure reason")
    if "failure_kind" in obj:
        _text(obj["failure_kind"], "failure_kind")
        if passed:
            _fail("PASS report cannot carry failure_kind")
    if "metadata" in obj and not isinstance(obj["metadata"], dict):
        _fail("metadata must be an object")
    cases = obj.get("cases")
    if not isinstance(cases, list):
        _fail("cases must be an array")
    if passed and action != "compile" and not cases:
        _fail(f"A passing {action} requires nonempty cases")
    ids = set()
    for row in cases:
        if not isinstance(row, dict):
            _fail("Each case must be an object")
        case_id = row.get("test_case_id")
        _text(case_id, "test_case_id")
        if case_id in ids:
            _fail(f"Duplicate test_case_id: {case_id}")
        ids.add(case_id)
        _status(row.get("status"), f"case {case_id}.status")
        if passed and row["status"] != "PASS":
            _fail(f"PASS report contains failed case {case_id}")
        if "params" in row and not isinstance(row["params"], dict):
            _fail(f"case {case_id}.params must be an object")
        if "dtype" in row:
            _text(row["dtype"], f"case {case_id}.dtype")
        for key in ("metrics", "metadata"):
            if key in row and not isinstance(row[key], dict):
                _fail(f"case {case_id}.{key} must be an object")
        if action == "validate-task":
            checks = row.get("checks")
            if (not isinstance(checks, list) or not checks
                    or any(not isinstance(c, str) or c not in ("correctness", "performance") for c in checks)
                    or len(checks) != len(set(checks))):
                _fail(f"case {case_id} requires a checks list")
            if "performance" in checks and "correctness" not in checks:
                _fail(f"Performance case {case_id} lacks correctness coverage")
        if action == "performance" and row["status"] == "PASS":
            latency = row.get("execution_time_ms")
            if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
                _fail(f"case {case_id} requires finite positive execution_time_ms")
            method = row.get("benchmark_method")
            if not isinstance(method, str) or method not in TIMING_METHODS:
                _fail(f"case {case_id} has unsupported device benchmark_method")
    return ActionResult(role, action, obj["status"], tuple(deepcopy(cases)),
                        obj.get("reason"), obj.get("failure_kind"), deepcopy(obj.get("metadata")))


@dataclass(frozen=True)
class CaseManifest:
    """Captured by the framework before optimization from protected task input."""
    cases: tuple[dict, ...]

    @classmethod
    def from_result(cls, result: ActionResult) -> "CaseManifest":
        if (result.role, result.action, result.status) != ("task", "validate-task", "PASS"):
            _fail("Manifest requires a successful validate-task action")
        if not any("performance" in row["checks"] for row in result.cases):
            _fail("Task manifest has no performance cases")
        return cls(tuple(deepcopy(result.cases)))

    def validate(self, result: ActionResult) -> None:
        """Validate against the independently captured manifest, not intersection."""
        if result.action == "validate-task":
            _fail("A validate-task result creates the manifest; it is not candidate evidence")
        rows = {r["test_case_id"]: r for r in self.cases}
        expected = {key for key, row in rows.items() if result.action in row["checks"]}
        actual = {row["test_case_id"] for row in result.cases}
        if len(actual) != len(result.cases):
            _fail("Duplicate cases in action result")
        if result.action == "compile":
            # Whole-build checks may omit cases; specialization checks may name
            # any subset, but cannot invent or change a declared input identity.
            if not actual <= rows.keys():
                _fail("Compile result contains cases outside the manifest")
        elif actual != expected:
            _fail(f"Case manifest mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
        for row in result.cases:
            if _identity(row) != _identity(rows[row["test_case_id"]]):
                _fail(f"Case input identity changed: {row['test_case_id']}")


def merge_command_results(results: Iterable[ActionResult]) -> ActionResult:
    results = tuple(results)
    if not results:
        _fail("Action produced no command results")
    first = results[0]
    ids = set()
    cases = []
    for result in results:
        if (result.role, result.action) != (first.role, first.action):
            _fail("Cannot merge different actions")
        for row in result.cases:
            if row["test_case_id"] in ids:
                _fail(f"Duplicate case across commands: {row['test_case_id']}")
            ids.add(row["test_case_id"])
            cases.append(deepcopy(row))
    failed = [result for result in results if not result.passed]
    reasons = "; ".join(result.reason or "Command failed" for result in failed) or None
    kinds = {result.failure_kind for result in failed}
    kind = kinds.pop() if len(kinds) == 1 else None
    return ActionResult(first.role, first.action, "FAIL" if failed else "PASS", tuple(cases),
                        reasons, kind, {"commands": [deepcopy(r.metadata) for r in results]})


def baseline_correctness_accepted(result: ActionResult, *, baseline: BaselineSpec,
                                  phase: str, manifest: CaseManifest) -> bool:
    """Apply only the declared baseline exception; never relabel its FAIL as PASS."""
    if (result.role, result.action) != ("baseline", "correctness"):
        _fail("Baseline policy cannot be applied to another role/action")
    manifest.validate(result)
    if result.passed:
        return True
    if phase != "task_validation" or baseline.correctness_policy != "diagnostic":
        return False
    if result.failure_kind != "numerical_mismatch":
        return False
    failed = [row for row in result.cases if row["status"] == "FAIL"]
    return bool(failed) and all(row.get("failure_kind") == "numerical_mismatch" for row in failed)


def performance_cases(result: ActionResult) -> list:
    """Convert validated v2 timing evidence into the existing scoring input."""
    from .testcases import TestCaseResult

    if result.action != "performance" or not result.passed:
        _fail("Scoring requires a passing performance action")
    cases = []
    for row in result.cases:
        metadata = deepcopy(row.get("metadata", {}))
        for key in ("dtype", "params", "metrics", "benchmark_method"):
            if key in row:
                metadata[key] = deepcopy(row[key])
        cases.append(TestCaseResult(row["test_case_id"], deepcopy(row.get("shape")),
                                    row["execution_time_ms"], metadata))
    return cases
