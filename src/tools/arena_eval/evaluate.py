# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""AgentKernelArena's evaluator for schema-v3 tasks: the seven task actions.

Run from the task workspace:

    python3 -m _aka_eval.evaluate validate-task
    python3 -m _aka_eval.evaluate {baseline|candidate} {compile|correctness|performance}

Every action prints exactly one ``ARENA_EVAL_RESULT=`` arena-eval-v1 line and
exits nonzero on failure. The framework, not this script, decides acceptance,
pairs actions with the captured manifest and computes scores.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import traceback
from pathlib import Path

from .contract import WORKSPACE, Case, Contract, ContractError, load_contract
from .integrity import TamperError, snapshot

PREFIX = "ARENA_EVAL_RESULT="
PROTOCOL = "arena-eval-v1"
TEMPLATE_MARKER = "AKA-UNIMPLEMENTED-TEMPLATE"


class ActionFailure(RuntimeError):
    def __init__(self, message: str, kind: str = "execution_error", cases: list | None = None):
        super().__init__(message)
        self.kind = kind
        self.cases = cases or []


def _report(role: str, action: str, status: str, cases: list, reason: str | None = None,
            kind: str | None = None, metadata: dict | None = None) -> dict:
    report = {"protocol": PROTOCOL, "role": role, "action": action, "status": status, "cases": cases}
    if reason:
        report["reason"] = reason[:4000]
    if kind and status == "FAIL":
        report["failure_kind"] = kind
    if metadata:
        report["metadata"] = metadata
    return report


def _device():
    import torch

    if not torch.cuda.is_available():
        raise ActionFailure("A GPU is required: torch.cuda.is_available() is false", "environment")
    return torch.device("cuda", torch.cuda.current_device())


def _synchronize() -> None:
    import torch

    torch.cuda.synchronize()


def candidate_state(contract: Contract) -> str:
    entry = WORKSPACE / contract.candidate.entry_file
    if not entry.is_file() or not entry.read_text(encoding="utf-8", errors="replace").strip():
        return "unimplemented"
    return "unimplemented" if TEMPLATE_MARKER in entry.read_text(encoding="utf-8", errors="replace") else "implemented"


def validate_task(contract: Contract) -> dict:
    from . import values

    for entry in (contract.reference, contract.generator, contract.layout):
        if entry is not None:
            values.workspace_path(entry["file"])
    values.workspace_path(contract.baseline.entry_file)
    rows = []
    for case in contract.cases:
        row = case.identity()
        row.update(status="PASS", checks=list(case.checks))
        rows.append(row)
    state = candidate_state(contract)
    smoke = []
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
            for case in (contract.cases[0], contract.cases[-1]):
                inputs = values.make_inputs(contract, case, case.seeds[0], device)
                values.reference_outputs(contract, inputs.logical)
                smoke.append(case.case_id)
    except (ContractError, RuntimeError, TypeError, ValueError) as exc:
        raise ActionFailure(f"Task inputs or reference failed: {type(exc).__name__}: {exc}", "task_error")
    metadata = {"candidate_state": state, "target_language": contract.candidate.language,
                "baseline_language": contract.baseline.language, "smoke_cases": smoke}
    return _report("task", "validate-task", "PASS", rows, metadata=metadata)


def _load(contract: Contract, role: str, *, audit: bool, integrity=None):
    from . import audit as policy
    from .loader import implementation_sources, load_implementation

    implementation = contract.implementation(role)
    sources = implementation_sources(implementation)
    evidence = {"language": implementation.language, "files": sorted(sources)}
    if implementation.language == "asm":
        from . import asm

        built = len(asm.build_evidence()["files"])
    if audit:
        entry = Path(implementation.entry_file).relative_to(implementation.root).as_posix()
        if implementation.language == "hip":
            evidence.update(policy.audit_hip_sources(sources, entry))
        else:
            evidence.update(policy.audit_python_sources(sources, implementation.language))
        with policy.preparation_only(f"{role} import"):
            loaded = load_implementation(contract, role)
    else:
        loaded = load_implementation(contract, role)
    if integrity is not None:
        integrity(f"{role} import")
    call = policy.audited_call(loaded.call, language=implementation.language, package=loaded.package,
                               role=role, enforce=audit, integrity=integrity)
    evidence["package"] = loaded.package
    if implementation.language == "asm":
        build = asm.build_evidence()
        evidence["asm_build"] = {**build, "files": build["files"][built:]}
    if loaded.hip_op:
        evidence["hip_op"] = loaded.hip_op
    return call, evidence


def _policy_enforced(role: str) -> bool:
    # The target-language policy applies to every candidate invocation, in both
    # phases; the task-provided baseline is the trusted comparison point.
    return role == "candidate"


def _integrity(role: str):
    """A check of the evaluator's bindings, snapshotted before candidate code imports."""
    return snapshot(__package__) if _policy_enforced(role) else None


def compile_action(contract: Contract, role: str) -> dict:
    from .audit import PolicyError
    from .loader import BuildError

    if role == "candidate" and candidate_state(contract) == "unimplemented":
        raise ActionFailure("The candidate is still the unimplemented template", "unimplemented")
    try:
        if contract.implementation(role).language == "hip" or role == "candidate":
            _device()
        _, evidence = _load(contract, role, audit=_policy_enforced(role), integrity=_integrity(role))
    except PolicyError as exc:
        raise ActionFailure(str(exc), "policy_violation")
    except BuildError as exc:
        raise ActionFailure(str(exc), "compile_error")
    return _report(role, "compile", "PASS", [], metadata={"implementation": evidence})


def _bound(contract: Contract, name: str) -> dict:
    return contract.compare[name]


def _dtype_name(tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _check(contract: Contract, role: str, written: dict, expected: dict, baseline: dict | None) -> tuple[bool, dict]:
    """Compare every written tensor; the baseline role applies its sanity rule."""
    from .compare import compare_tensor, sqnr_db

    metrics, passed = {}, True
    for name in contract.checked_tensors:
        declaration = _bound(contract, name)
        actual = written[name]
        reference = expected[name]
        if role == "baseline" and declaration.get("bound", "baseline_relative") == "baseline_relative":
            result = compare_tensor({"bound": "output_rounding", "max_violation_fraction": 1.0},
                                    actual, reference, dtype_name=_dtype_name(actual))
            sqnr = sqnr_db(actual, reference.to(actual.dtype)) if actual.dtype.is_floating_point else math.inf
            minimum = float(declaration.get("baseline_min_sqnr_db", 25.0))
            ok = result.nonfinite_mismatches == 0 and sqnr >= minimum
            metrics[name] = {"bound": "baseline_sanity", "passed": ok, "sqnr_db": sqnr if math.isfinite(sqnr) else 1e300,
                             "minimum_sqnr_db": minimum, "nonfinite_mismatches": result.nonfinite_mismatches}
        else:
            result = compare_tensor(declaration, actual, reference, dtype_name=_dtype_name(actual),
                                    baseline=None if baseline is None else baseline.get(name),
                                    element_order=contract.element_order)
            ok = result.passed
            metrics[name] = result.to_mapping()
        passed &= ok
    return passed, metrics


class _CaseData:
    """Inputs and expected values of one case for its two seeds."""

    def __init__(self, contract: Contract, case: Case, device, baseline_call, need_baseline: bool):
        from . import values

        self.inputs = [values.make_inputs(contract, case, seed, device) for seed in case.seeds[:2]]
        self.references = [values.reference_outputs(contract, item.logical) for item in self.inputs]
        self.baselines = None
        if need_baseline:
            self.baselines = []
            for item in self.inputs:
                kernel = values.clone_inputs(item.kernel)
                outputs = values.allocate_outputs(contract, case, device)
                values.poison(outputs, 7)
                baseline_call(**kernel, **outputs)
                _synchronize()
                self.baselines.append({name: tensor.clone() for name, tensor in
                                       values.checked_values(contract, kernel, outputs).items()})


def _invoke_and_check(contract: Contract, role: str, call, case: Case, data: _CaseData, device) -> dict:
    from . import values

    working = values.clone_inputs(data.inputs[0].kernel)
    outputs = values.allocate_outputs(contract, case, device)
    read_only = [name for name, d in contract.inputs.items() if d["kind"] == "tensor" and name not in contract.mutable]
    calls = []
    passed = True
    for index in range(2):
        if index:
            values.copy_into(working, data.inputs[index].kernel)
        values.poison(outputs, 11 + index)
        before = values.snapshot(working, read_only)
        _synchronize()
        call(**working, **outputs)
        _synchronize()
        changed = values.changed_inputs(before, working)
        ok, metrics = _check(contract, role, values.checked_values(contract, working, outputs),
                             data.references[index],
                             None if data.baselines is None else data.baselines[index])
        record = {"seed": case.seeds[index], "outputs": metrics}
        if changed:
            record["modified_inputs"] = changed
            ok = False
        passed &= ok
        calls.append(record)
    return {"passed": passed, "calls": calls}


def _role_calls(contract: Contract, role: str):
    """The role's audited call and, when its bound needs it, the baseline's."""
    from .audit import PolicyError
    from .loader import BuildError

    if role == "candidate" and candidate_state(contract) == "unimplemented":
        raise ActionFailure("The candidate is still the unimplemented template", "unimplemented")
    need_baseline = role == "candidate" and any(
        _bound(contract, n).get("bound", "baseline_relative") == "baseline_relative" for n in contract.checked_tensors)
    try:
        # Reference and baseline modules load before candidate code imports.
        baseline_call = _load(contract, "baseline", audit=False)[0] if (role == "baseline" or need_baseline) else None
        check = _integrity(role)
        call = baseline_call if role == "baseline" else _load(contract, "candidate", audit=True,
                                                              integrity=check)[0]
    except PolicyError as exc:
        raise ActionFailure(str(exc), "policy_violation")
    except BuildError as exc:
        raise ActionFailure(str(exc), "compile_error")
    return call, baseline_call, need_baseline, check


def correctness_action(contract: Contract, role: str) -> dict:
    device = _device()
    call, baseline_call, need_baseline, check = _role_calls(contract, role)
    rows, failures, kinds = [], [], set()
    for case in contract.cases:
        row = case.identity()
        try:
            data = _CaseData(contract, case, device, baseline_call, need_baseline)
            result = _invoke_and_check(contract, role, call, case, data, device)
            row["metrics"] = {"calls": result["calls"]}
            if result["passed"]:
                row["status"] = "PASS"
            else:
                row["status"] = "FAIL"
                modified = any(c.get("modified_inputs") for c in result["calls"])
                row["failure_kind"] = "input_mutation" if modified else "numerical_mismatch"
        except Exception as exc:  # A crash in one case is that case's failure; a TamperError is not an Exception.
            row.update(status="FAIL", failure_kind="policy_violation" if "PyTorch operation" in str(exc)
                       or "kernel launch" in str(exc) or "launched kernels" in str(exc) else "execution_error",
                       metadata={"error": f"{type(exc).__name__}: {exc}"[:2000]})
        if row["status"] == "FAIL":
            failures.append(row["test_case_id"])
            kinds.add(row["failure_kind"])
        rows.append(row)
    if failures:
        kind = kinds.pop() if len(kinds) == 1 else "mixed"
        if kind == "mixed":
            kind = "numerical_mismatch" if kinds <= {"numerical_mismatch"} else "execution_error"
        return _report(role, "correctness", "FAIL", rows, f"{len(failures)} case(s) failed: {failures[:8]}", kind)
    if check is not None:
        check("correctness")
    return _report(role, "correctness", "PASS", rows)


_FLUSH = None


def _cache_flush(nbytes: int):
    global _FLUSH
    from ._aka_benchmark import ReadOnlyCacheFlush

    if _FLUSH is None or _FLUSH.nbytes != nbytes:
        _FLUSH = ReadOnlyCacheFlush(nbytes)
    return _FLUSH


def _single_call_cold(contract: Contract, role: str, call, data: _CaseData, working: dict, outputs: dict,
                      flush, check) -> tuple[list[float], dict, list]:
    """The single-call cold-cache samples: one call per replay, its outputs checked after every sample."""
    from . import values
    from ._aka_benchmark import TimedRun, benchmark_cuda_graph_cold_cache_samples

    timing = contract.timing
    state = {"prepared": 0, "current": 0, "checked": 0, "failed": []}

    def prepare() -> None:
        index = state["prepared"] % 2
        values.copy_into(working, data.inputs[index].kernel)
        values.poison(outputs, 100 + state["prepared"])
        state["current"] = index
        state["prepared"] += 1

    def after_sample(_outputs) -> None:
        # First, before the sample is used: candidate code ran while it was measured.
        if check is not None:
            check("a timing sample")
        index = state["current"]
        ok, metrics = _check(contract, role, values.checked_values(contract, working, outputs),
                             data.references[index],
                             None if data.baselines is None else data.baselines[index])
        state["checked"] += 1
        if not ok and len(state["failed"]) < 3:
            state["failed"].append({"sample": state["checked"], "outputs": metrics})

    timed = TimedRun()
    timed.after_sample = after_sample
    samples, metadata = benchmark_cuda_graph_cold_cache_samples(
        lambda: call(**working, **outputs), cache_flush=flush,
        warmup=int(timing.get("warmup", 10)), repetition=int(timing.get("repetition", 100)),
        prepare_fn=prepare, timed_run=timed)
    if check is not None:
        check("timing")
    return samples, dict(metadata, checked_samples=state["checked"]), state["failed"]


def _amortized_cold(contract: Contract, role: str, call, data: _CaseData, working: dict, outputs: dict,
                    flush, check, estimate_ms: float) -> tuple[list[float], dict, list]:
    """Amortized cold-cache samples with every call's results checked.

    The candidate's Python runs once per call while the graph is captured, so
    with one shared output buffer it could count the calls and do the work on
    the last one only. Each call of a replay therefore writes its own outputs
    and mutable inputs (read-only inputs are shared), and every call is checked
    after every sample. Copying the results out after each call instead would
    not cancel: the copy reads values the operator has just left in the caches,
    while the reference graph's copy reads them cold.
    """
    import torch

    from . import values
    from ._aka_benchmark import (TimedRun, amortized_calls_per_replay,
                                 benchmark_cuda_graph_cold_cache_amortized_samples)

    timing = contract.timing
    mutable = {name: working[name] for name in contract.mutable}
    # Each call owns its outputs, its mutable inputs and a poison pattern of every
    # output that is not NaN-filled.
    per_call = values.nbytes(outputs) + values.nbytes(mutable) + values.nbytes(
        {name: tensor for name, tensor in outputs.items() if not values.nan_poisoned(tensor)})
    count = amortized_calls_per_replay(estimate_ms, per_call_bytes=per_call)
    # Distinct poison per call, precomputed: generators cannot be created under capture.
    patterns = [values.poison_pattern(outputs, 1000 + index) for index in range(count)]
    inputs = [{**working, **{name: torch.empty_like(tensor) for name, tensor in mutable.items()}}
              for _ in range(count)]
    written = [{name: torch.empty_like(tensor) for name, tensor in outputs.items()} for _ in range(count)]
    state = {"checked": 0, "failed": []}

    def prepare(index: int) -> None:
        values.copy_into(inputs[index], data.inputs[index % 2].kernel)
        values.apply_poison(written[index], patterns[index])

    def check_calls() -> list[dict]:
        failed = []
        for index in range(count):
            ok, metrics = _check(contract, role, values.checked_values(contract, inputs[index], written[index]),
                                 data.references[index % 2],
                                 None if data.baselines is None else data.baselines[index % 2])
            if not ok:
                failed.append({"call": index, "outputs": metrics})
        return failed

    def after_sample(_outputs) -> None:
        if check is not None:
            check("a timing sample")
        state["checked"] += 1
        for failure in check_calls():
            if len(state["failed"]) < 3:
                state["failed"].append({"sample": state["checked"], **failure})

    timed = TimedRun()
    timed.after_sample = after_sample
    samples, metadata = benchmark_cuda_graph_cold_cache_amortized_samples(
        [lambda index=index: call(**inputs[index], **written[index]) for index in range(count)],
        cache_flush=flush, prepare_fn=prepare,
        warmup=int(timing.get("warmup", 10)), repetition=int(timing.get("repetition", 100)),
        timed_run=timed)
    if check is not None:
        check("timing")
    # Control: the helper ended with a reference-graph replay, which never runs
    # the operator. Checks that accept those buffers cannot show that each call
    # did its work, so a candidate is not scored on them.
    undetected = count - len(check_calls())
    metadata = dict(metadata, checked_samples=state["checked"], checked_calls=state["checked"] * count,
                    benchmark_call_buffer_bytes=per_call, reference_control_undetected_calls=undetected)
    failed = state["failed"]
    if undetected and role == "candidate":
        failed = failed + [{"control": f"{undetected} of {count} calls without the operator passed the "
                                       "output checks; per-call work cannot be verified"}]
    return samples, metadata, failed


def performance_action(contract: Contract, role: str) -> dict:
    from . import values

    device = _device()
    call, baseline_call, need_baseline, check = _role_calls(contract, role)
    flush = _cache_flush(int(contract.timing.get("cache_flush_bytes", 1 << 30)))
    rows, failures = [], []
    for case in contract.cases:
        if "performance" not in case.checks:
            continue
        row = case.identity()
        try:
            data = _CaseData(contract, case, device, baseline_call, need_baseline)
            working = values.clone_inputs(data.inputs[0].kernel)
            outputs = values.allocate_outputs(contract, case, device)
            single, single_metadata, failed = _single_call_cold(contract, role, call, data, working, outputs,
                                                                flush, check)
            if not failed:
                samples, metadata, failed = _amortized_cold(contract, role, call, data, working, outputs,
                                                            flush, check, statistics.median(single))
            if failed:
                control = any("control" in failure for failure in failed)
                row.update(status="FAIL",
                           failure_kind="unverifiable_measurement" if control else "numerical_mismatch",
                           metadata={"measured_output_failures": failed})
            else:
                metadata.update(samples_min_ms=min(samples), samples_max_ms=max(samples),
                                samples_count=len(samples),
                                single_call_cold_ms=statistics.median(single),
                                single_call_cold_method=single_metadata["benchmark_method"],
                                single_call_cold_checked_samples=single_metadata["checked_samples"])
                row.update(status="PASS", execution_time_ms=statistics.median(samples),
                           benchmark_method=metadata["benchmark_method"], metadata=metadata)
        except Exception as exc:
            row.update(status="FAIL", failure_kind="execution_error",
                       metadata={"error": f"{type(exc).__name__}: {exc}"[:2000]})
        if row["status"] == "FAIL":
            failures.append(row["test_case_id"])
        rows.append(row)
    if failures:
        return _report(role, "performance", "FAIL", rows, f"{len(failures)} case(s) failed: {failures[:8]}",
                       "performance_error")
    return _report(role, "performance", "PASS", rows)


def run(argv: list[str]) -> tuple[dict, int]:
    if argv == ["validate-task"]:
        role, action = "task", "validate-task"
    elif len(argv) == 2 and argv[0] in ("baseline", "candidate") and argv[1] in ("compile", "correctness", "performance"):
        role, action = argv
    else:
        return _report("task", "validate-task", "FAIL", [], f"Unknown action {argv}", "usage"), 2
    try:
        contract = load_contract()
        if action == "validate-task":
            report = validate_task(contract)
        elif action == "compile":
            report = compile_action(contract, role)
        elif action == "correctness":
            report = correctness_action(contract, role)
        else:
            report = performance_action(contract, role)
    except ActionFailure as exc:
        report = _report(role, action, "FAIL", exc.cases, str(exc), exc.kind)
    except TamperError as exc:
        report = _report(role, action, "FAIL", [], str(exc), "harness_tampering")
    except ContractError as exc:
        report = _report(role, action, "FAIL", [], f"Contract error: {exc}", "task_error")
    except Exception as exc:
        detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        report = _report(role, action, "FAIL", [], detail, "execution_error")
    return report, 0 if report["status"] == "PASS" else 1


def main() -> int:
    report, code = run(sys.argv[1:])
    sys.stdout.flush()
    print(PREFIX + json.dumps(report, allow_nan=False, default=str), flush=True)
    return code


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Containers often run as a uid absent from /etc/passwd; getpass-based
    # cache paths in PyTorch/Triton must not turn that into a task failure.
    os.environ.setdefault("USER", "aka")
    sys.exit(main())
