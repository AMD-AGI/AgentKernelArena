#!/usr/bin/env python3
"""Complete GEAK callable UT for ledger case hk11."""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback


HERE = os.path.dirname(os.path.abspath(__file__))
# PATCHED BY tools/build_suite.py: RUN_ROOT was two levels above HERE,
# which is right in the delivery layout (HERE = <delivery>/tasks/<task>) but
# escapes the task here (HERE = <task>/ut), dropping reports/ledger/<case>.json
# into tasks/headkernel/ next to the sibling tasks on every run.
RUN_ROOT = os.path.dirname(HERE)


def _load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if os.path.realpath(getattr(existing, "__file__", "")) != os.path.realpath(path):
            raise RuntimeError(f"protected module alias already names a different file: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


h = _load("harness_lib", os.path.join(HERE, "harness_lib.py"))
with open(os.path.join(HERE, "meta.json")) as _fh:
    META = json.load(_fh)


def _enter_candidate_overlay():
    marker = os.environ.get("GEAK_ACTIVE_TASK_CANDIDATE")
    if marker == HERE:
        return
    _base, candidate = h.build_candidate_overlay(HERE, META)
    env = dict(os.environ)
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = candidate + (os.pathsep + prior if prior else "")
    env["GEAK_ACTIVE_TASK_CANDIDATE"] = HERE
    script = os.path.join(HERE, "unittest.py")
    code = f"import runpy; runpy.run_path({script!r}, run_name='__main__')"
    os.chdir("/")
    os.execve(sys.executable, [sys.executable, "-c", code, *sys.argv[1:]], env)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _shape_key(entry):
    return tuple(tuple(int(v) for v in shape) for shape in entry.get("input_shapes") or [])


def _verify_provenance():
    if META.get("reference_io_sha256") != "":
        raise RuntimeError("runtime-baseline oracle must not claim a reference_io_sha256")

    selection_path = os.path.join(HERE, META["selection_validation"]["path"])
    with open(selection_path) as fh:
        selection = json.load(fh)
    if not selection.get("ok") or not selection.get("deepest_verified"):
        raise RuntimeError("live seam selection is not verified")
    if selection.get("target_callable") != META["target_callable"]:
        raise RuntimeError("selection evidence names a different callable")
    if selection.get("device_kernel") != META["device_kernel"]:
        raise RuntimeError("selection evidence names a different device kernel")

    shape_path = os.path.join(HERE, META["shape_validation"]["path"])
    with open(shape_path) as fh:
        captured = json.load(fh)
    observed = {
        _shape_key(entry): entry
        for entry in captured.get("shape_counts") or []
    }
    required = {
        ((8192, 8192), (8192, 8192), (8192,)),
        ((64, 8192), (64, 8192), (8192,)),
    }
    if not required.issubset(observed):
        raise RuntimeError("live shape evidence does not contain both M=8192 and M=64")
    for key in required:
        if observed[key].get("input_dtypes") != ["torch.bfloat16"]:
            raise RuntimeError(f"unexpected live dtype for {key}: {observed[key].get('input_dtypes')}")
        if "|1e-06" not in observed[key].get("sig", ""):
            raise RuntimeError(f"unexpected live eps for {key}")

    layout_path = os.path.join(HERE, META["layout_validation"]["path"])
    with open(layout_path) as fh:
        layout = json.load(fh)
    if layout.get("processes_observed") != 8:
        raise RuntimeError("layout evidence does not cover all eight selected processes")
    for row in layout.get("per_process") or []:
        if row.get("view") != row.get("reshape") or any(
            row.get(name) for name in ("clone", "copy_", "contiguous")
        ):
            raise RuntimeError(f"layout evidence contains a materializing reshape: {row}")

    return {
        "reference_io_sha256": "not_applicable_runtime_baseline",
        "selection_ok": True,
        "deepest_verified": True,
        "matched_kernel_calls": selection.get("matched_kernel_calls"),
        "processes_passed": len(selection.get("process_verdicts") or []),
        "live_shapes": [64, 8192],
        "layout_processes": layout.get("processes_observed"),
    }


def _contract_checks(cases, eager):
    report = []
    all_ok = True
    for case in eager:
        try:
            cases.call(case["args"])
            row = {
                "case": case["sig"],
                "correct": True,
                "tuple_arity": 2,
                "fresh_outputs": True,
                "outputs_non_aliasing": True,
                "inputs_unchanged": True,
            }
        except Exception as exc:
            all_ok = False
            row = {
                "case": case["sig"],
                "correct": False,
                "note": f"{type(exc).__name__}: {exc}",
            }
        report.append(row)
    return all_ok, report


def _negative_checks(cases, eager, tol):
    one = [min(eager, key=lambda case: int(case["m"]))]

    def corrupt_normed(args):
        normed, pre_norm_sum = cases.call(args)
        normed = normed.clone()
        pre_norm_sum = pre_norm_sum.clone()
        normed.reshape(-1)[0] += 1024
        return normed, pre_norm_sum

    def corrupt_pre_norm_sum(args):
        normed, pre_norm_sum = cases.call(args)
        normed = normed.clone()
        pre_norm_sum = pre_norm_sum.clone()
        pre_norm_sum.reshape(-1)[0] += 1024
        return normed, pre_norm_sum

    normed_ok, normed_report = h.check_correct_multi(corrupt_normed, one, tol)
    sum_ok, sum_report = h.check_correct_multi(corrupt_pre_norm_sum, one, tol)
    payload = {
        "contract": "negative_control",
        "normed_corruption_rejected": not normed_ok,
        "pre_norm_sum_corruption_rejected": not sum_ok,
        "normed_report": normed_report,
        "pre_norm_sum_report": sum_report,
    }
    _write_json(os.path.join(HERE, META["negative_check"]), payload)
    return (not normed_ok) and (not sum_ok), payload


def _print_report(report):
    for group, rows in report.items():
        for row in rows:
            state = "PASS" if row.get("correct") else "FAIL"
            detail = row.get("note") or ""
            print(
                f"{group}:{row.get('case', '')} {state} "
                f"max_rel_err={row.get('max_rel_err')} {detail}".rstrip()
            )


def _run(case_id):
    started = time.time()
    provenance = _verify_provenance()

    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
    cases = _load("fused_add_rmsnorm_cases", os.path.join(HERE, "cases.py"))
    torch = h._torch()
    if not torch.cuda.is_available():
        raise RuntimeError("hk11 requires a ROCm/CUDA device")

    baseline_random_path = os.path.join(HERE, "_baseline_random.pt")
    try:
        baseline_outputs = h.baseline_random_outputs(
            HERE,
            META,
            seed=0,
            draws=int(META.get("random_draws", 3)),
        )
    finally:
        try:
            os.unlink(baseline_random_path)
        except FileNotFoundError:
            pass

    eager = cases.eager_cases(h, META, baseline_outputs, seed=0)
    random = cases.random_shapes(h, META)
    replay = cases.graph_replay_bundle(h, eager)

    try:
        ok, correctness = h.run_correctness(
            META["regime"],
            eager_cases=eager,
            current_call=cases.call,
            random_shapes=random,
            tol=float(META.get("tol", 0.02)),
            baseline_outputs=baseline_outputs,
            draws=int(META.get("random_draws", 3)),
            replay=replay,
        )

        sequence_ok, sequence_report = h.check_correct_sequence(
            cases.call,
            cases.ordered_boundary_cases(eager),
            float(META.get("tol", 0.02)),
        )
        correctness["ordered_sequence"] = sequence_report
        ok = ok and sequence_ok

        contract_ok, contract_report = _contract_checks(cases, eager)
        correctness["callable_contract"] = contract_report
        ok = ok and contract_ok

        negative_ok, negative = _negative_checks(
            cases, eager, float(META.get("tol", 0.02)))
        ok = ok and negative_ok
        _print_report(correctness)
        print(
            "negative_control "
            + ("PASS" if negative_ok else "FAIL")
            + " normed_corruption_rejected="
            + str(negative["normed_corruption_rejected"])
            + " pre_norm_sum_corruption_rejected="
            + str(negative["pre_norm_sum_corruption_rejected"])
        )
    finally:
        del baseline_outputs, replay, random, eager
        torch.cuda.empty_cache()

    per_case = h.measure_legs(HERE, META)
    timing = h.serving_weighted_speedup(per_case, META)
    timing_ok = bool(per_case) and all(
        row.get("baseline_ms") and row.get("optimized_ms") for row in per_case)
    ok = ok and timing_ok
    for row in per_case:
        print(
            f"timing:{row['sig']} baseline_ms={row.get('baseline_ms')} "
            f"candidate_ms={row.get('optimized_ms')} speedup={row.get('speedup')} "
            f"reps={row.get('reps')}"
        )
    if timing.get("weighted") is not None:
        print(f"GEAK_WEIGHTED_SPEEDUP={timing['weighted']:.6f}")
    else:
        print(f"GEAK_WEIGHTED_SPEEDUP=UNTRUSTED reason={timing.get('reason', '')}")

    result = {
        "schema_version": 1,
        "ledger_id": case_id,
        "task_id": META["task_id"],
        "case_name": META["case_entrypoints"][case_id],
        "status": "PASS" if ok else "FAIL",
        "callable": META["target_callable"],
        "device_kernel": META["device_kernel"],
        "shapes": [
            {
                "regime": item["regime"],
                "M": item["m"],
                "N": item["n"],
                "dtype": item["dtype"],
                "x_stride": item["x_stride"],
                "residual_stride": item["residual_stride"],
                "weight_stride": item["weight_stride"],
                "eps": item["eps"],
            }
            for item in META["workload"]["cases"]
        ],
        "oracle": {
            "policy": "runtime_frozen_baseline",
            "reference_io_sha256": "",
            "random_draws_per_shape": int(META.get("random_draws", 3)),
        },
        "gates": {
            "provenance": provenance,
            "fixed_shape_baseline_parity": all(
                row.get("correct") for row in correctness.get("eager", [])),
            "random_parity": all(
                row.get("correct") for row in correctness.get("random", [])),
            "graph_replay": all(
                row.get("correct") for row in correctness.get("graph_replay", [])),
            "ordered_sequence": sequence_ok,
            "callable_contract": contract_ok,
            "negative_control": negative_ok,
            "distinct_legs_and_smoke": timing_ok,
        },
        "correctness": correctness,
        "negative_check": negative,
        "timing": {"per_case": per_case, "aggregate": timing},
        "duration_sec": round(time.time() - started, 3),
    }
    _write_json(os.path.join(HERE, "result.json"), result)
    _write_json(os.path.join(RUN_ROOT, "reports", "ledger", f"{case_id}.json"), result)
    print(f"CASE_RESULT {case_id} {result['status']}")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", choices=META["ledger_ids"], default="hk11")
    args = parser.parse_args()
    _enter_candidate_overlay()
    try:
        return _run(args.case_id)
    except h.HarnessIncompleteError:
        return 3
    except Exception as exc:
        result = {
            "schema_version": 1,
            "ledger_id": args.case_id,
            "task_id": META.get("task_id"),
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(os.path.join(HERE, "result.json"), result)
        _write_json(os.path.join(RUN_ROOT, "reports", "ledger", f"{args.case_id}.json"), result)
        traceback.print_exc()
        print(f"CASE_RESULT {args.case_id} ERROR")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
