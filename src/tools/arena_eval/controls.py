# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic task-quality controls run by the QA auditor on a GPU.

    python3 -m _aka_eval.controls [--json PATH]

Each control wraps the trusted baseline in a known-wrong behavior and runs the
same two-draw, repeated-call correctness check every candidate faces. A task
whose check accepts any of them cannot distinguish that wrong implementation,
which is a blocking audit finding. The module also reports degenerate input
draws. It never edits task files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import sys
from typing import Any, Callable

import torch

from . import values
from .contract import load_contract
from .evaluate import _CaseData, _invoke_and_check, _load

RELATIVE_SHIFT_ULPS = 16


def _tensors(contract, outputs, working):
    return values.checked_values(contract, working, outputs)


def _mutants(contract, baseline: Callable) -> dict[str, Callable]:
    read_only = [n for n, d in contract.inputs.items() if d["kind"] == "tensor" and n not in contract.mutable]
    cache: dict[str, Any] = {}
    effective = set()  # controls whose perturbation changed at least one checked tensor

    def noop(**call):
        return None

    def zeros(**call):
        baseline(**call)
        for name in contract.checked_tensors:
            if bool(call[name].ne(0).any()):
                effective.add("zeros")
            call[name].zero_()

    def half(**call):
        baseline(**call)
        for name in contract.checked_tensors:
            flat = call[name].view(-1)
            if flat.numel() >= 2:
                tail = flat[flat.numel() // 2:]
                if bool(tail.ne(0).any()):
                    effective.add("half_written")
                tail.zero_()

    def stale(**call):
        if not cache:
            baseline(**call)
            cache.update({name: call[name].clone() for name in contract.checked_tensors})
        else:
            for name, value in cache.items():
                call[name].copy_(value)

    def input_write(**call):
        baseline(**call)
        if read_only:
            target = call[read_only[0]].view(-1)
            if target.numel():
                raw = target.view(torch.uint8)
                raw[0] ^= 0x5A

    def shifted(**call):
        baseline(**call)
        for name in contract.checked_tensors:
            tensor = call[name]
            dtype = str(tensor.dtype).removeprefix("torch.")
            if dtype in values.UNIT_ROUNDOFF:
                factor = 1.0 + RELATIVE_SHIFT_ULPS * values.UNIT_ROUNDOFF[dtype]
                shifted_value = (tensor.double() * factor).to(tensor.dtype)
                if not torch.equal(values.tensor_bytes(shifted_value), values.tensor_bytes(tensor)):
                    effective.add("shifted_16_ulp")
                tensor.copy_(shifted_value)
            elif tensor.numel():
                effective.add("shifted_16_ulp")
                raw = tensor.view(-1).view(torch.uint8)
                raw[0] ^= 1

    def low_precision(**call):
        baseline(**call)
        for name in contract.checked_tensors:
            tensor = call[name]
            if tensor.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16) and hasattr(
                    torch, "float8_e4m3fn"):
                rounded = tensor.to(torch.float8_e4m3fn).to(tensor.dtype)
                if not torch.equal(values.tensor_bytes(rounded), values.tensor_bytes(tensor)):
                    effective.add("fp8_rounded")
                tensor.copy_(rounded)

    controls = {"noop": noop, "zeros": zeros, "half_written": half, "stale_outputs": stale,
                "input_write": input_write, "shifted_16_ulp": shifted, "fp8_rounded": low_precision}
    if not read_only:
        del controls["input_write"]

    def reset():
        cache.clear()

    controls["_reset"] = reset
    controls["_effective"] = effective
    return controls


@contextmanager
def _float32_torch():
    """Run PyTorch code with float64 requests served in float32.

    The reference evaluated this way is a second correct implementation of the
    operator with float32 accumulation and library reduction orders, which is
    what a candidate kernel looks like to the error bound.
    """
    saved = torch.float64, torch.double, torch.Tensor.double
    torch.float64 = torch.double = torch.float32
    torch.Tensor.double = lambda self, memory_format=torch.preserve_format: self.float()
    try:
        yield
    finally:
        torch.float64, torch.double, torch.Tensor.double = saved


def _calibration_probe(contract, case, data: _CaseData) -> list[dict]:
    """Whether a float32 evaluation of the reference passes each baseline-relative bound.

    It runs only when the evaluation actually differs from the float64 reference.
    A failure is a calibration warning, not proof of a wrong bound: a float32
    evaluation can be less accurate than a baseline that accumulates more carefully.
    """
    from .compare import compare_tensor
    from .evaluate import _bound, _dtype_name

    names = [n for n in contract.checked_tensors if _bound(contract, n).get("bound", "baseline_relative") ==
             "baseline_relative" and data.baselines[0][n].dtype.is_floating_point]
    if not names:
        return []
    values.reference_function(contract)  # import the reference (and its constants) outside the patch
    try:
        with _float32_torch():
            probe = values.reference_outputs(contract, data.inputs[0].logical)
    except Exception as exc:  # noqa: BLE001 - a reference that cannot run in float32 is not probed
        return [{"case": case.case_id, "skipped": f"{type(exc).__name__}: {str(exc)[:200]}"}]
    results = []
    for name in names:
        baseline = data.baselines[0][name]
        reference = data.references[0][name]
        candidate = probe[name]
        if candidate.dtype == reference.dtype and torch.equal(candidate, reference):
            continue  # the reference never ran in reduced precision
        candidate = candidate.to(baseline.dtype)
        result = compare_tensor(_bound(contract, name), candidate, reference, dtype_name=_dtype_name(baseline),
                                baseline=baseline, element_order=contract.element_order)
        results.append({"case": case.case_id, "output": name, **result.to_mapping()})
    return results


def _input_findings(contract, data: _CaseData) -> list[dict]:
    findings = []
    for name, declaration in contract.inputs.items():
        if declaration["kind"] != "tensor" or declaration.get("role") == "state":
            continue
        tensor = data.inputs[0].logical[name]
        if not isinstance(tensor, torch.Tensor) or tensor.numel() < 2:
            continue
        if tensor.dtype.is_floating_point:
            sample = tensor.reshape(-1)[:1 << 20].double()
            if sample.isfinite().all() and float(sample.std()) == 0.0:
                findings.append({"input": name, "issue": "constant floating input draw"})
            if tensor.dtype in (torch.float32, torch.float64):
                exact = (sample == sample.to(torch.bfloat16).double()).double().mean().item()
                if exact > 0.5:
                    findings.append({"input": name, "issue": f"{exact:.0%} of values are exact in bfloat16"})
        else:
            sample = tensor.reshape(-1)[:1 << 20]
            if declaration.get("role") not in ("index", "metadata") and bool((sample == sample[0]).all()):
                findings.append({"input": name, "issue": "constant integer input draw"})
    return findings


def run_controls() -> dict:
    contract = load_contract()
    device = torch.device("cuda", torch.cuda.current_device())
    baseline, _ = _load(contract, "baseline", audit=False)
    mutants = _mutants(contract, baseline)
    reset = mutants.pop("_reset")
    effective = mutants.pop("_effective")
    rejected = {name: [] for name in mutants}
    input_findings, determinism, reference_dtypes, calibration = [], [], {}, []
    for case in contract.cases:
        data = _CaseData(contract, case, device, baseline, need_baseline=True)
        if case is contract.cases[0] or case is contract.cases[-1]:
            input_findings += [{"case": case.case_id, **item} for item in _input_findings(contract, data)]
            reference_dtypes.update({name: str(tensor.dtype).removeprefix("torch.")
                                     for name, tensor in data.references[0].items()})
        try:
            calibration += _calibration_probe(contract, case, data)
        except Exception as exc:  # noqa: BLE001 - the probe is informational and must not stop the controls
            calibration.append({"case": case.case_id, "skipped": f"{type(exc).__name__}: {str(exc)[:200]}"})
        # Baseline run-to-run agreement on identical inputs (informational).
        runs = []
        for _ in range(2):
            working = values.clone_inputs(data.inputs[0].kernel)
            outputs = values.allocate_outputs(contract, case, device)
            values.poison(outputs, 3)
            baseline(**working, **outputs)
            torch.cuda.synchronize()
            runs.append({n: t.clone() for n, t in _tensors(contract, outputs, working).items()})
        if any(not torch.equal(values.tensor_bytes(runs[0][n]), values.tensor_bytes(runs[1][n])) for n in runs[0]):
            determinism.append(case.case_id)
        for name, mutant in mutants.items():
            if rejected[name]:
                continue
            reset()
            try:
                result = _invoke_and_check(contract, "candidate", mutant, case, data, device)
                if not result["passed"]:
                    rejected[name].append(case.case_id)
            except Exception as exc:  # A crash also rejects the control.
                rejected[name].append(f"{case.case_id}:{type(exc).__name__}")
    # A perturbing control that never changed a checked tensor returned the baseline's own outputs, so it
    # proves nothing about the task and is not applicable: fp8 rounding of integer outputs or of values fp8
    # represents exactly, zeroing a half that is already zero, a single-element output, an all-zero output.
    not_applicable = sorted(name for name in ("zeros", "half_written", "shifted_16_ulp", "fp8_rounded")
                            if name in rejected and not rejected[name] and name not in effective)
    for name in not_applicable:
        del rejected[name]
    survived = sorted(name for name, cases in rejected.items() if not cases)
    return {"controls": {name: {"rejected": bool(cases), "first_rejections": cases[:3]}
                         for name, cases in rejected.items()},
            "survived": survived, "not_applicable": not_applicable, "input_findings": input_findings,
            "calibration_probe": calibration,
            "baseline_nondeterministic_cases": determinism, "reference_dtypes": reference_dtypes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json")
    args = parser.parse_args()
    try:
        report = {"status": "COMPLETE", **run_controls()}
    except Exception as exc:
        report = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    text = json.dumps(report, indent=1, default=str, allow_nan=False)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print("ARENA_CONTROLS_RESULT=" + json.dumps(report, default=str, allow_nan=False))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    import os

    os.environ.setdefault("USER", "aka")
    sys.exit(main())
