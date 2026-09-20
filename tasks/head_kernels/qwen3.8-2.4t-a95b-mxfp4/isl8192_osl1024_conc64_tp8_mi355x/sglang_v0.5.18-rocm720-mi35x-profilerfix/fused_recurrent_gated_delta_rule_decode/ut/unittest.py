#!/usr/bin/env python3
"""Complete GEAK UT for ledger case hk10."""

import argparse
import hashlib
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


def _reference_sha256():
    digest = hashlib.sha256()
    with open(os.path.join(HERE, META["reference_io"]), "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_provenance():
    actual = _reference_sha256()
    expected = META.get("reference_io_sha256")
    if not expected or actual != expected:
        raise RuntimeError(
            f"reference_io_sha256 mismatch: expected={expected!r}, actual={actual!r}")
    selection_path = os.path.join(HERE, META["selection_validation"]["path"])
    with open(selection_path) as fh:
        selection = json.load(fh)
    if not selection.get("ok") or not selection.get("deepest_verified"):
        raise RuntimeError("live seam selection is not verified")
    if selection.get("device_kernel") != META["device_kernel"]:
        raise RuntimeError("selection evidence names a different device kernel")
    return {
        "reference_io_sha256": "match",
        "selection_ok": True,
        "deepest_verified": True,
        "matched_kernel_calls": selection.get("matched_kernel_calls"),
        "processes_passed": len(selection.get("process_verdicts") or []),
    }


def _negative_checks(cases, eager, tol):
    one = [eager[0]]

    def corrupt_output(args):
        out, state = cases.call(args)
        out = out.clone()
        state = state.clone()
        out.reshape(-1)[0] += 1024
        return out, state

    def corrupt_state(args):
        out, state = cases.call(args)
        out = out.clone()
        state = state.clone()
        state.reshape(-1)[0] += 1024
        return out, state

    output_ok, output_report = h.check_correct_multi(corrupt_output, one, tol)
    state_ok, state_report = h.check_correct_multi(corrupt_state, one, tol)
    payload = {
        "contract": "negative_control",
        "output_corruption_rejected": not output_ok,
        "state_corruption_rejected": not state_ok,
        "output_report": output_report,
        "state_report": state_report,
    }
    _write_json(os.path.join(HERE, META["negative_check"]), payload)
    return (not output_ok) and (not state_ok), payload


def _print_report(report):
    for group, rows in report.items():
        for row in rows:
            state = "PASS" if row.get("correct") else "FAIL"
            detail = row.get("note") or ""
            print(f"{group}:{row.get('case', '')} {state} max_rel_err={row.get('max_rel_err')} {detail}".rstrip())


def _run(case_id):
    started = time.time()
    provenance = _verify_provenance()

    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
    cases = _load("gated_delta_cases", os.path.join(HERE, "cases.py"))
    torch = h._torch()
    if not torch.cuda.is_available():
        raise RuntimeError("hk10 requires a ROCm/CUDA device")

    eager = cases.eager_cases(h, META, device="cuda")
    random = cases.random_shapes(h, META)
    baseline_outputs = h.baseline_random_outputs(
        HERE, META, draws=int(META.get("random_draws", 3)))
    baseline_random_path = os.path.join(HERE, "_baseline_random.pt")
    try:
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
        except h.HarnessIncompleteError:
            raise

        sequence_ok, sequence_report = h.check_correct_sequence(
            cases.call,
            cases.ordered_boundary_cases(eager),
            float(META.get("tol", 0.02)),
        )
        correctness["ordered_sequence"] = sequence_report
        ok = ok and sequence_ok

        negative_ok, negative = _negative_checks(
            cases, eager, float(META.get("tol", 0.02)))
        ok = ok and negative_ok
        _print_report(correctness)
        print(
            "negative_control "
            + ("PASS" if negative_ok else "FAIL")
            + " output_corruption_rejected="
            + str(negative["output_corruption_rejected"])
            + " state_corruption_rejected="
            + str(negative["state_corruption_rejected"])
        )
    finally:
        try:
            os.unlink(baseline_random_path)
        except FileNotFoundError:
            pass

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
            {"regime": item["regime"], "B": item["B"]}
            for item in META["workload"]["cases"]
        ],
        "gates": {
            "provenance": provenance,
            "eager_oracle": all(row.get("correct") for row in correctness.get("eager", [])),
            "random_parity": all(row.get("correct") for row in correctness.get("random", [])),
            "graph_replay": all(row.get("correct") for row in correctness.get("graph_replay", [])),
            "ordered_sequence": sequence_ok,
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
    parser.add_argument("--case-id", choices=META["ledger_ids"], default="hk10")
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
