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


# The single-call cold-cache median is kept as metadata for continuity with
# earlier results; it is not scored, so a few samples suffice.
SINGLE_CALL_SAMPLES = 10


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
        warmup=int(timing.get("warmup", 10)), repetition=SINGLE_CALL_SAMPLES,
        prepare_fn=prepare, timed_run=timed)
    if check is not None:
        check("timing")
    return samples, dict(metadata, checked_samples=state["checked"]), state["failed"]


def draw_seed(case: Case, index: int) -> int:
    """The seed of the amortized measurement's draw ``index``, derived from the case's framework seeds.

    A SplitMix64 step in integer arithmetic only, so candidate code that runs in
    this process has no library function to replace. The draws differ from the
    two correctness seeds, which warm the operator up and time the single-call
    samples, and from each other.
    """
    mask = (1 << 64) - 1
    value = (case.seeds[0] << 32 | case.seeds[1]) + (index + 1) * 0x9E3779B97F4A7C15 & mask
    value = (value ^ value >> 30) * 0xBF58476D1CE4E5B9 & mask
    value = (value ^ value >> 27) * 0x94D049BB133111EB & mask
    return (value ^ value >> 31) & 0x7FFFFFFF


def _draw(contract: Contract, case: Case, index: int, device) -> tuple[dict, dict]:
    """Kernel inputs and reference outputs of amortized draw ``index``."""
    from . import values

    drawn = values.make_inputs(contract, case, draw_seed(case, index), device)
    return drawn.kernel, values.reference_outputs(contract, drawn.logical)


def _baseline_outputs(contract: Contract, case: Case, baseline_call, inputs: dict, device) -> dict:
    from . import values

    kernel = values.clone_inputs(inputs)
    outputs = values.allocate_outputs(contract, case, device)
    values.poison(outputs, 7)
    baseline_call(**kernel, **outputs)
    _synchronize()
    return {name: tensor.clone() for name, tensor in values.checked_values(contract, kernel, outputs).items()}


def _same_bytes(first: dict, second: dict) -> bool:
    import torch

    from . import values

    return all(torch.equal(values.tensor_bytes(first[name]), values.tensor_bytes(second[name])) for name in first)


def _secret_permutation(count: int) -> list[int]:
    """A fresh uniformly random order of ``range(count)`` from the operating system's entropy.

    It is drawn after the candidate's last Python call of the measurement and is
    held only in local variables, so candidate code can neither predict nor read it.
    """
    order = list(range(count))
    for index in range(count - 1, 0, -1):
        # Rejection sampling over 32-bit draws: no modulo bias.
        limit = (1 << 32) - (1 << 32) % (index + 1)
        while True:
            value = int.from_bytes(os.urandom(4), "little")
            if value < limit:
                break
        other = value % (index + 1)
        order[index], order[other] = order[other], order[index]
    return order


def _amortized_cold(contract: Contract, role: str, call, case: Case, data: _CaseData, working: dict,
                    outputs: dict, flush, check, device, baseline_call) -> tuple[list[float], dict, list]:
    """Amortized cold-cache samples with every call's results checked.

    The candidate's Python runs once per captured call, so it could count the
    calls: with one shared output buffer it could work on the last call only;
    with repeated inputs a call could copy an earlier call's result; and with a
    fixed draw per graph a call could keep device state, compute in the first
    sample and restore a stored result in every later one. So the case has K
    distinct draws (each with its own reference and, where the bound needs it,
    baseline outputs), every call writes its own outputs and mutable inputs,
    and before every sample a secret random permutation assigns the draws to the
    calls: the draws are copied into the calls' input buffers outside both Event
    spans. Every call is checked after every sample against its draw of that
    sample: fully the first time a draw is checked, then by bitwise equality
    with that draw's first checked result (a difference is checked fully again).
    """
    import torch

    from . import values
    from ._aka_benchmark import (TimedRun, amortized_calls_per_replay,
                                 benchmark_cuda_graph_cold_cache_amortized_samples)

    timing = contract.timing
    mutable = {name: working[name] for name in contract.mutable}
    first_inputs, first_reference = _draw(contract, case, 0, device)
    checked_bytes = values.nbytes(outputs) + values.nbytes(mutable)
    # One call holds its draw, the copy of a draw staged into its input buffers, its
    # reference, its output and mutable-input buffers, the first checked copy of its draw's
    # results, the baseline's outputs and a poison pattern of each output that is not
    # NaN-filled. Baseline outputs count in both roles, so K never depends on the role.
    per_call = (2 * values.nbytes(first_inputs) + values.nbytes(first_reference) + 3 * checked_bytes
                + values.nbytes({name: tensor for name, tensor in outputs.items() if not values.nan_poisoned(tensor)}))
    count = amortized_calls_per_replay(per_call)
    draws, references = [first_inputs], [first_reference]
    for index in range(1, count):
        inputs, reference = _draw(contract, case, index, device)
        draws.append(inputs)
        references.append(reference)
    baselines = None
    if data.baselines is not None:
        baselines = [_baseline_outputs(contract, case, baseline_call, inputs, device) for inputs in draws]
    # Distinct poison per call, precomputed: generators cannot be created under capture.
    patterns = [values.poison_pattern(outputs, 1000 + index) for index in range(count)]
    # Call i reads its read-only inputs from staged[i] and works on its own copy of the
    # mutable ones, which its preparation restores from staged[i].
    staged = [values.clone_inputs(first_inputs) for _ in range(count)]
    inputs = [{**staged[index], **{name: torch.empty_like(tensor) for name, tensor in mutable.items()}}
              for index in range(count)]
    written = [{name: torch.empty_like(tensor) for name, tensor in outputs.items()} for _ in range(count)]
    assigned: list[int] = []        # the draw of each call in the current sample; empty until timing starts
    verified: list[dict | None] = [None] * count
    state = {"checked": 0, "full_checks": 0, "failed": []}

    def stage() -> None:
        assigned[:] = _secret_permutation(count)
        for index, draw in enumerate(assigned):
            values.copy_into(staged[index], draws[draw])

    def prepare(index: int) -> None:
        for name in mutable:
            inputs[index][name].copy_(staged[index][name])
        values.apply_poison(written[index], patterns[index])

    def warm(index: int) -> None:
        # Warm-up uses the correctness draws and the single-call buffers, never a timed draw.
        values.copy_into(working, data.inputs[index % 2].kernel)
        values.poison(outputs, 50 + index)
        call(**working, **outputs)

    def check_call(index: int, *, full: bool) -> tuple[bool, dict]:
        draw = assigned[index]
        result = values.checked_values(contract, inputs[index], written[index])
        if not full and verified[draw] is not None and _same_bytes(verified[draw], result):
            return True, {}
        state["full_checks"] += 1
        ok, metrics = _check(contract, role, result, references[draw],
                             None if baselines is None else baselines[draw])
        if ok and verified[draw] is None and not full:
            verified[draw] = {name: tensor.clone() for name, tensor in result.items()}
        return ok, metrics

    def after_sample(_outputs) -> None:
        if check is not None:
            check("a timing sample")
        state["checked"] += 1
        for index in range(count):
            ok, metrics = check_call(index, full=False)
            if not ok and len(state["failed"]) < 3:
                state["failed"].append({"sample": state["checked"], "call": index, "draw": assigned[index],
                                        "outputs": metrics})

    timed = TimedRun()
    timed.after_sample = after_sample
    samples, metadata = benchmark_cuda_graph_cold_cache_amortized_samples(
        [lambda index=index: call(**inputs[index], **written[index]) for index in range(count)],
        cache_flush=flush, prepare_fn=prepare, warmup_fn=warm, stage_fn=stage,
        warmup=int(timing.get("warmup", 10)), repetition=int(timing.get("repetition", 100)),
        timed_run=timed)
    if check is not None:
        check("timing")
    # Control: the helper ended with a reference replay, which never runs the
    # operator. Checks that accept those buffers cannot show that each call did
    # its work, so a candidate is not scored on them.
    undetected = sum(check_call(index, full=True)[0] for index in range(count))
    metadata = dict(metadata, checked_samples=state["checked"], checked_calls=state["checked"] * count,
                    full_checks=state["full_checks"], benchmark_call_buffer_bytes=per_call,
                    benchmark_draw_assignment="secret_permutation_per_sample",
                    reference_control_undetected_calls=undetected)
    failed = state["failed"]
    if undetected and role == "candidate":
        failed = failed + [{"control": f"{undetected} of {count} calls without the operator passed the "
                                       "output checks; per-call work cannot be verified"}]
    return samples, metadata, failed


# A per-call time far below the same operator's single-call time cannot come
# from removing the replay's fixed cost (about 7 us on MI355X) and the noise
# of either measurement: it means the timed calls did less than the
# single-call samples. Honest cases sit within a few us (short) or a few
# percent (long) of the single-call median.
CONSISTENCY_RATIO = 0.8
CONSISTENCY_SLACK_MS = 0.010


def _consistent(amortized_ms: float, single_ms: float) -> bool:
    return amortized_ms >= CONSISTENCY_RATIO * single_ms - CONSISTENCY_SLACK_MS


def performance_action(contract: Contract, role: str) -> dict:
    from . import values

    from ._aka_benchmark import failed_captures

    device = _device()
    call, baseline_call, need_baseline, check = _role_calls(contract, role)
    flush = _cache_flush(int(contract.timing.get("cache_flush_bytes", 1 << 30)))
    rows, failures = [], []
    for case in contract.cases:
        if "performance" not in case.checks:
            continue
        row = case.identity()
        if failed_captures():
            # A failed capture leaves the process's capture state unusable, so later
            # cases cannot be measured here; say so instead of reporting their errors.
            row.update(status="FAIL", failure_kind="execution_error",
                       metadata={"error": "not measured: a CUDA-graph capture of an earlier case failed in "
                                          "this process"})
            failures.append(row["test_case_id"])
            rows.append(row)
            continue
        try:
            data = _CaseData(contract, case, device, baseline_call, need_baseline)
            working = values.clone_inputs(data.inputs[0].kernel)
            outputs = values.allocate_outputs(contract, case, device)
            single, single_metadata, failed = _single_call_cold(contract, role, call, data, working, outputs,
                                                                flush, check)
            if not failed:
                samples, metadata, failed = _amortized_cold(contract, role, call, case, data, working, outputs,
                                                            flush, check, device, baseline_call)
            if not failed and not _consistent(statistics.median(samples), statistics.median(single)):
                failed = [{"consistency": f"amortized median {statistics.median(samples) * 1e3:.2f} us is below "
                                          f"{CONSISTENCY_RATIO} x the single-call median "
                                          f"{statistics.median(single) * 1e3:.2f} us - "
                                          f"{CONSISTENCY_SLACK_MS * 1e3:.0f} us"}]
            if failed:
                control = any("control" in failure or "consistency" in failure for failure in failed)
                row.update(status="FAIL",
                           failure_kind="unverifiable_measurement" if control else "numerical_mismatch",
                           metadata={"measured_output_failures": failed})
            else:
                metadata.update(samples_min_ms=min(samples), samples_max_ms=max(samples),
                                samples_count=len(samples),
                                single_call_cold_ms=statistics.median(single),
                                single_call_cold_method=single_metadata["benchmark_method"],
                                single_call_cold_samples=len(single))
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


def _failed_captures() -> int:
    helper = sys.modules.get(f"{__package__}._aka_benchmark")
    return helper.failed_captures() if helper is not None else 0


def _exit(code: int) -> None:
    if _failed_captures():
        # PyTorch aborts while destroying a graph whose capture failed, which
        # would replace the reported result's exit code with a signal.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    sys.exit(code)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Containers often run as a uid absent from /etc/passwd; getpass-based
    # cache paths in PyTorch/Triton must not turn that into a task failure.
    os.environ.setdefault("USER", "aka")
    _exit(main())
