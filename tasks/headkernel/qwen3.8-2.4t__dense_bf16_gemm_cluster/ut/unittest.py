#!/usr/bin/env python3
"""Complete callable UT for dense BF16 GEMM ledger cases hk04-hk08."""

import argparse
import importlib
import importlib.util
import inspect
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
RUN_ROOT = os.environ.get("HK_RUN_ROOT", HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _update_case_file(path, case_id, payload):
    try:
        with open(path) as fh:
            root = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        root = {"schema_version": 1, "task_id": "dense_bf16_gemm_task", "cases": {}}
    root.setdefault("cases", {})[case_id] = payload
    root["status"] = (
        "PASS"
        if set(root["cases"]) == set(META["ledger_ids"])
        and all(row.get("status") == "PASS" for row in root["cases"].values())
        else "IN_PROGRESS"
    )
    _write_json(path, root)


with open(os.path.join(HERE, "meta.json")) as fh:
    META = json.load(fh)

if META.get("reference_io_sha256") != "":
    raise RuntimeError("runtime baseline GEMM oracle must not claim reference_io_sha256")

os.environ["AITER_CONFIG_GEMM_BF16"] = os.path.join(
    HERE, META["dispatch_config"]
)
h = _load("harness_lib", os.path.join(HERE, "harness_lib.py"))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.modules.pop("unittest", None)
cases = _load("dense_bf16_gemm_cases", os.path.join(HERE, "cases.py"))


def _enter_candidate_overlay():
    if os.environ.get("GEAK_ACTIVE_TASK_CANDIDATE") == HERE:
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


def _callable_identity(fn):
    module = getattr(fn, "__module__", "") or ""
    path = ""
    try:
        path = inspect.getsourcefile(fn) or ""
    except (OSError, TypeError):
        pass
    if not path and module:
        path = getattr(sys.modules.get(module), "__file__", "") or ""
    return {
        "module": module,
        "qualname": getattr(fn, "__qualname__", repr(fn)),
        "file": os.path.realpath(path) if path else "",
    }


def _verify_selection(case_id):
    path = os.path.join(HERE, META["selection_validation"]["path"])
    with open(path) as fh:
        selection = json.load(fh)
    if not selection.get("ok") or not selection.get("deepest_verified"):
        raise RuntimeError("standalone live dispatch selection has not passed for all five cases")
    if selection.get("target_callable") != META["target_callable"]:
        raise RuntimeError("selection evidence names a different callable")
    row = (selection.get("cases") or {}).get(case_id)
    if not row or not row.get("ok"):
        raise RuntimeError(f"selection evidence is missing or failed for {case_id}")
    return row


def _runtime_config(torch, tuned_gemm, case):
    config = tuned_gemm.get_GEMM_A16W16_config(
        int(case["m"]),
        int(case["n"]),
        int(case["k"]),
        False,
        str(torch.bfloat16),
        str(torch.bfloat16),
        False,
        False,
    )
    return config


def _config_matches(config, case):
    return (
        config.get("libtype") == case["expected_backend"]
        and int(config.get("solidx", -1)) == int(case["expected_solidx"])
        and int(config.get("splitK", 0) or 0) == int(case["expected_split_k"])
        and str(config.get("kernelName", "") or "") == case["expected_kernel"]
    )


def _install_backend_observer(tuned_gemm, backend):
    original = tuned_gemm.solMap[backend]
    state = {"calls": 0}

    def observed(*args, **kwargs):
        state["calls"] += 1
        return original(*args, **kwargs)

    tuned_gemm.solMap[backend] = observed
    return original, state


def _input_contract(torch, case):
    args = cases.make_args(case, seed=4000)
    a_before = args["A"].clone()
    b_before = args["B"].clone()
    first = cases.candidate_call(args)
    second = cases.candidate_call(args)
    ok = (
        first.dtype == torch.bfloat16
        and list(first.shape) == [int(case["m"]), int(case["n"])]
        and first.data_ptr() != second.data_ptr()
        and torch.equal(args["A"], a_before)
        and torch.equal(args["B"], b_before)
    )
    row = {
        "correct": bool(ok),
        "output_dtype": str(first.dtype),
        "output_shape": list(first.shape),
        "fresh_output": first.data_ptr() != second.data_ptr(),
        "inputs_unchanged": torch.equal(args["A"], a_before) and torch.equal(args["B"], b_before),
    }
    del args, a_before, b_before, first, second
    torch.cuda.empty_cache()
    return bool(ok), row


def _negative_check(case, eager, tol):
    def corrupt(args):
        out = cases.candidate_call(args).clone()
        out.reshape(-1)[0] += 1024
        return out

    ok, report = h.check_correct_multi(corrupt, [eager[0]], tol)
    payload = {
        "contract": "negative_control",
        "ledger_id": case["ledger_id"],
        "output_corruption_rejected": not ok,
        "report": report,
    }
    _update_case_file(os.path.join(HERE, META["negative_check"]), case["ledger_id"], {
        "status": "PASS" if not ok else "FAIL",
        **payload,
    })
    return not ok, payload


def _timing_smoke(case):
    timing = cases.timing_case(case)
    graph = h.deployment_graph_mode(META["regime"])
    baseline = h.time_op(
        lambda: cases.baseline_call(timing["args"]),
        warmup=2,
        repeats=3,
        inner=1,
        graph=graph,
        flush_cache=False,
        detail=True,
    )
    candidate = h.time_op(
        lambda: cases.candidate_call(timing["args"]),
        warmup=2,
        repeats=3,
        inner=1,
        graph=graph,
        flush_cache=False,
        detail=True,
    )
    row = {
        "sig": case["sig"],
        "regime": case["regime"],
        "m": int(case["m"]),
        "baseline_ms": (baseline or {}).get("ms"),
        "candidate_ms": (candidate or {}).get("ms"),
        "baseline_timer": (baseline or {}).get("timer"),
        "candidate_timer": (candidate or {}).get("timer"),
    }
    row["speedup"] = (
        row["baseline_ms"] / row["candidate_ms"]
        if row["baseline_ms"] and row["candidate_ms"]
        else None
    )
    return bool(row["baseline_ms"] and row["candidate_ms"]), row


def _run_case(case_id):
    started = time.time()
    selection = _verify_selection(case_id)
    case = cases.case_map(META)[case_id]
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("dense GEMM UT requires a ROCm/CUDA device")
    tuned_gemm = importlib.import_module("aiter.tuned_gemm")
    config = _runtime_config(torch, tuned_gemm, case)
    config_ok = _config_matches(config, case)
    if not config_ok:
        raise RuntimeError(f"live dispatcher config drift for {case_id}: {config}")

    baseline_identity = _callable_identity(torch.nn.functional.linear)
    candidate_identity = _callable_identity(tuned_gemm.gemm_a16w16)
    identity_ok = baseline_identity != candidate_identity
    if not identity_ok:
        raise RuntimeError("baseline and candidate resolved to the same callable identity")

    original_backend, engagement = _install_backend_observer(
        tuned_gemm, case["expected_backend"]
    )
    tol = float(case.get("tol", META["tol"]))
    eager = []
    replay = None
    baseline_outputs = {}
    try:
        baseline_outputs = cases.baseline_random_outputs(
            case, int(META["random_draws"]), seed=0
        )
        eager = cases.eager_cases(case)
        replay = cases.graph_replay_bundle(case)
        ok, correctness = h.run_correctness(
            META["regime"],
            eager_cases=eager,
            current_call=cases.candidate_call,
            random_shapes=cases.random_shapes(case),
            tol=tol,
            baseline_outputs=baseline_outputs,
            draws=int(META["random_draws"]),
            replay=replay,
        )
        graph_rows = correctness.get("graph_replay") or []
        graph_executed = bool(graph_rows) and all(
            "skipped:" not in str(row.get("note", "")) for row in graph_rows
        )
        ok = ok and graph_executed

        contract_ok, contract = _input_contract(torch, case)
        ok = ok and contract_ok
        negative_ok, negative = _negative_check(case, eager, tol)
        ok = ok and negative_ok
        timing_ok, timing = _timing_smoke(case)
        ok = ok and timing_ok
        engagement_ok = engagement["calls"] > 0
        ok = ok and engagement_ok and identity_ok and config_ok
    finally:
        tuned_gemm.solMap[case["expected_backend"]] = original_backend

    result = {
        "schema_version": 1,
        "ledger_id": case_id,
        "task_id": META["task_id"],
        "case_name": case["name"],
        "status": "PASS" if ok else "FAIL",
        "callable": META["target_callable"],
        "math_contract": META["math_contract"],
        "shape": {
            "M": int(case["m"]),
            "N": int(case["n"]),
            "K": int(case["k"]),
            "dtype": case["dtype"],
            "a_stride": case["a_stride"],
            "b_stride": case["b_stride"],
            "regime": case["regime"],
        },
        "backend": {
            "expected": case["expected_backend"],
            "runtime_config": {
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in config.items()
            },
            "backend_hook_calls": engagement["calls"],
            "device_selection_evidence": selection,
        },
        "oracle": {
            "policy": "runtime_frozen_torch_linear",
            "baseline_identity": baseline_identity,
            "candidate_identity": candidate_identity,
            "reference_io_sha256": "",
            "random_draws": int(META["random_draws"]),
            "tolerance": tol,
        },
        "gates": {
            "selection": bool(selection.get("ok")),
            "runtime_config_exact": config_ok,
            "backend_engaged": engagement_ok,
            "fixed_shape_baseline_parity": all(
                row.get("correct") for row in correctness.get("eager", [])
            ),
            "random_parity": all(
                row.get("correct") for row in correctness.get("random", [])
            ),
            "graph_replay": graph_executed and all(
                row.get("correct") for row in graph_rows
            ),
            "callable_contract": contract_ok,
            "negative_control": negative_ok,
            "distinct_baseline_candidate": identity_ok,
            "smoke_timing": timing_ok,
        },
        "correctness": correctness,
        "callable_contract": contract,
        "negative_check": negative,
        "timing": timing,
        "duration_sec": round(time.time() - started, 3),
    }
    ledger_path = os.path.join(RUN_ROOT, "reports", "ledger", f"{case_id}.json")
    _write_json(ledger_path, result)
    _update_case_file(os.path.join(HERE, "result.json"), case_id, result)
    print(
        f"CASE_RESULT {case_id} {result['status']} "
        f"backend={case['expected_backend']} graph_replay={result['gates']['graph_replay']}"
    )
    del baseline_outputs, replay, eager
    torch.cuda.empty_cache()
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", choices=META["ledger_ids"], default=None)
    args = parser.parse_args()
    _enter_candidate_overlay()
    selected = [args.case_id] if args.case_id else list(META["ledger_ids"])
    failed = []
    for case_id in selected:
        try:
            if _run_case(case_id) != 0:
                failed.append(case_id)
        except h.HarnessIncompleteError:
            failed.append(case_id)
        except Exception as exc:
            failed.append(case_id)
            result = {
                "schema_version": 1,
                "ledger_id": case_id,
                "task_id": META["task_id"],
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_json(
                os.path.join(RUN_ROOT, "reports", "ledger", f"{case_id}.json"),
                result,
            )
            _update_case_file(os.path.join(HERE, "result.json"), case_id, result)
            traceback.print_exc()
            print(f"CASE_RESULT {case_id} ERROR")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
