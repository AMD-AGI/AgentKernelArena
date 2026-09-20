#!/usr/bin/env python3
"""Callable UT for the live paged-attention ledger entry hk09."""

import argparse
import gc
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
RUN_ROOT = os.path.dirname(HERE)


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


with open(os.path.join(HERE, "meta.json")) as _fh:
    META = json.load(_fh)

h = _load("harness_lib", os.path.join(HERE, "harness_lib.py"))


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


def _identity(fn):
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


def _verify_provenance():
    with open(os.path.join(HERE, META["capture_meta"])) as fh:
        capture = json.load(fh)
    with open(os.path.join(HERE, "capture_telemetry.json")) as fh:
        telemetry = json.load(fh)
    expected = META["reference_io_sha256"]
    if not expected or capture.get("reference_io_sha256") != expected:
        raise RuntimeError("capture/meta reference checksum provenance mismatch")
    if telemetry.get("reference_io_sha256") != expected:
        raise RuntimeError("promoted oracle checksum evidence mismatch")
    if not telemetry.get("checksum_verified_once_after_promotion"):
        raise RuntimeError("promoted oracle lacks checksum verification evidence")
    transform = (capture.get("cases") or [{}])[0].get("capture_transform") or {}
    if transform.get("kind") != "paged_attention_referenced_pages":
        raise RuntimeError("attention oracle lacks referenced-page compaction evidence")
    if not capture.get("oracle_complete") or capture.get("budget_exceeded"):
        raise RuntimeError("live attention oracle is incomplete or over budget")
    return {
        "reference_io_sha256": "verified_at_promotion",
        "oracle_complete": True,
        "budget_exceeded": False,
        "capture_transform": transform,
        "total_calls_observed": capture["total_calls_observed"],
    }


def _run(case_id, cases):
    started = time.time()
    torch = h._torch()
    contract = META["case_contracts"][case_id]
    case = cases.load_live_case("cuda")
    provenance = _verify_provenance()

    baseline_identity = _identity(cases.baseline_callable())
    candidate_identity = _identity(cases.current_callable())
    identity_ok = baseline_identity != candidate_identity
    if not identity_ok:
        raise RuntimeError("baseline and candidate resolve to the same callable identity")

    baseline_outputs = cases.baseline_random_outputs(
        case, draws=int(META["random_draws"]), seed=0
    )
    replay = cases.graph_replay_bundle(h, case)
    ok, correctness = h.run_correctness(
        META["regime"],
        eager_cases=[case],
        current_call=cases.call,
        random_shapes=cases.random_shapes(case),
        tol=float(META["tol"]),
        baseline_outputs=baseline_outputs,
        replay=replay,
        draws=int(META["random_draws"]),
    )
    graph_rows = correctness.get("graph_replay") or []
    graph_executed = bool(graph_rows) and all(
        "skipped:" not in str(row.get("note", "")) for row in graph_rows
    )
    ok = ok and graph_executed

    contract_ok, callable_contract = cases.callable_contract(h, case)
    negative_ok, negative = cases.negative_check(h, case, float(META["tol"]))
    engagement_ok, engagement = cases.profile_engagement(
        h, case, contract["required_device_kernels"]
    )
    timing_ok, timing = cases.timing_smoke(h, case)
    ok = ok and contract_ok and negative_ok and engagement_ok and timing_ok

    selection = {
        "schema_version": 1,
        "task_id": META["task_id"],
        "ledger_id": case_id,
        "status": "PASS" if engagement_ok else "FAIL",
        "target_callable": META["target_callable"],
        "deepest_verified": True,
        **engagement,
    }
    _write_json(os.path.join(HERE, META["selection_validation"]["path"]), selection)
    _write_json(
        os.path.join(HERE, META["negative_check"]),
        {"schema_version": 1, "status": "PASS" if negative_ok else "FAIL", **negative},
    )

    positional = case["args"]["positional"]
    result = {
        "schema_version": 1,
        "ledger_id": case_id,
        "task_id": META["task_id"],
        "case_name": META["case_entrypoints"][case_id],
        "status": "PASS" if ok else "FAIL",
        "callable": META["target_callable"],
        "device_kernel": contract["required_device_kernels"][0],
        "math_contract": META["math_contract"],
        "shape": {
            "regime": "decode",
            "M": 64,
            "query": list(positional[2].shape),
            "key_cache_compact": list(positional[3].shape),
            "value_cache_compact": list(positional[4].shape),
            "kv_indptr": list(positional[6].shape),
            "kv_page_indices": list(positional[7].shape),
            "kv_last_page_lens": list(positional[8].shape),
            "output": list(positional[0].shape),
            "workspace": list(positional[1].shape),
            "kv_dtype": str(positional[3].dtype),
            "layout": positional[13],
        },
        "oracle": {
            "policy": "persistent_live_capture_referenced_pages",
            "reference_io_sha256": META["reference_io_sha256"],
            "random_draws": int(META["random_draws"]),
            "kv_metadata_scales_preserved": True,
        },
        "identity": {
            "baseline": baseline_identity,
            "candidate": candidate_identity,
            "distinct": identity_ok,
        },
        "gates": {
            "provenance": provenance,
            "live_oracle_eager": all(row.get("correct") for row in correctness.get("eager", [])),
            "random_parity": all(row.get("correct") for row in correctness.get("random", [])),
            "graph_replay": graph_executed and all(row.get("correct") for row in graph_rows),
            "callable_contract": contract_ok,
            "negative_control": negative_ok,
            "distinct_baseline_candidate": identity_ok,
            "device_kernel_engagement": engagement_ok,
            "smoke_timing": timing_ok,
        },
        "correctness": correctness,
        "callable_contract": callable_contract,
        "negative_check": negative,
        "device_engagement": engagement,
        "timing": timing,
        "duration_sec": round(time.time() - started, 3),
    }
    _write_json(os.path.join(HERE, META["result_path"]), result)
    _write_json(os.path.join(RUN_ROOT, "reports", "ledger", f"{case_id}.json"), result)
    print(
        f"CASE_RESULT {case_id} {result['status']} M=64 "
        f"graph_replay={result['gates']['graph_replay']} engagement={engagement_ok}"
    )
    del baseline_outputs, replay
    gc.collect()
    torch.cuda.empty_cache()
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", choices=META["ledger_ids"], default="hk09")
    args = parser.parse_args()
    _enter_candidate_overlay()

    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
    cases = _load("paged_attention_cases", os.path.join(HERE, "cases.py"))
    torch = h._torch()
    if not torch.cuda.is_available():
        raise RuntimeError("paged-attention UT requires a ROCm/CUDA device")
    try:
        return _run(args.case_id, cases)
    except h.HarnessIncompleteError:
        return 3
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "ledger_id": args.case_id,
            "task_id": META["task_id"],
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(os.path.join(HERE, META["result_path"]), payload)
        _write_json(os.path.join(RUN_ROOT, "reports", "ledger", f"{args.case_id}.json"), payload)
        traceback.print_exc()
        print(f"CASE_RESULT {args.case_id} ERROR")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
