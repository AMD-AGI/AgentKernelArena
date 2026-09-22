#!/usr/bin/env python3
"""Prove that each live GEMM shape reaches its configured AITER backend on MI355X."""

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path, payload):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


with open(os.path.join(HERE, "meta.json")) as fh:
    META = json.load(fh)

os.environ["AITER_CONFIG_GEMM_BF16"] = os.path.join(
    HERE, META["dispatch_config"]
)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.modules.pop("unittest", None)
cases = _load("dense_bf16_gemm_cases", os.path.join(HERE, "cases.py"))


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _profile_names(torch, call):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.profiler.ProfilerActivity, "CUDA"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as prof:
        out = call()
        torch.cuda.synchronize()
    names = []
    for event in prof.events():
        device_type = str(getattr(event, "device_type", ""))
        if "CPU" not in device_type:
            names.append(str(event.name))
    return out, sorted(set(names))


def validate_one(case):
    torch = importlib.import_module("torch")
    tuned_gemm = importlib.import_module("aiter.tuned_gemm")
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
    actual = {key: _plain(value) for key, value in config.items()}
    config_ok = (
        actual.get("libtype") == case["expected_backend"]
        and int(actual.get("solidx", -1)) == int(case["expected_solidx"])
        and int(actual.get("splitK", 0) or 0) == int(case["expected_split_k"])
        and str(actual.get("kernelName", "") or "") == case["expected_kernel"]
    )

    backend = case["expected_backend"]
    original = tuned_gemm.solMap[backend]
    calls = {"count": 0}

    def observed(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    tuned_gemm.solMap[backend] = observed
    args = cases.make_args(case, seed=7000)
    try:
        warm = cases.candidate_call(args)
        torch.cuda.synchronize()
        del warm
        output, device_events = _profile_names(
            torch, lambda: cases.candidate_call(args)
        )
        torch.cuda.synchronize()
        output_shape_ok = list(output.shape) == [int(case["m"]), int(case["n"])]
        del output
    finally:
        tuned_gemm.solMap[backend] = original
        del args
        torch.cuda.empty_cache()

    needle = case.get("profile_match", "")
    matching_events = [name for name in device_events if needle and needle in name]
    device_ok = bool(device_events) and (not needle or bool(matching_events))
    ok = config_ok and calls["count"] >= 2 and output_shape_ok and device_ok
    return {
        "ledger_id": case["ledger_id"],
        "shape": [int(case["m"]), int(case["n"]), int(case["k"])],
        "expected_backend": backend,
        "runtime_config": actual,
        "config_exact": config_ok,
        "backend_hook_calls": calls["count"],
        "output_shape_ok": output_shape_ok,
        "profile_match": needle,
        "matching_device_events": matching_events,
        "device_event_count": len(device_events),
        "device_events": device_events,
        "ok": ok,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", choices=META["ledger_ids"], default=None)
    args = parser.parse_args()
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("selection validation requires a ROCm/CUDA device")

    selected = args.case_id and [args.case_id] or META["ledger_ids"]
    rows = {}
    for case in cases.selected_cases(META, selected):
        row = validate_one(case)
        rows[case["ledger_id"]] = row
        print(
            f"SELECTION {case['ledger_id']} {'PASS' if row['ok'] else 'FAIL'} "
            f"backend={row['runtime_config'].get('libtype')} "
            f"hook_calls={row['backend_hook_calls']} "
            f"device_matches={len(row['matching_device_events'])}"
        )

    prior_path = os.path.join(HERE, "selection_validation.json")
    try:
        with open(prior_path) as fh:
            prior = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        prior = {}
    merged = dict(prior.get("cases") or {})
    merged.update(rows)
    required = set(META["ledger_ids"])
    all_ok = required.issubset(merged) and all(
        bool(merged[case_id].get("ok")) for case_id in required
    )
    payload = {
        "contract": "live_dispatch_and_device_engagement",
        "ok": all_ok,
        "deepest_verified": all_ok,
        "target_callable": META["target_callable"],
        "validated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(0),
        "arch": str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0],
        "dispatch_config": META["dispatch_config"],
        "cases": merged,
        "source_evidence": {
            "hk04_profiler_selection": "/shared_nfs/zepingl/qwen38_hyperloom_local_kernel_20260912T171040Z_093_multikernel_r10/hyperloom/Qwen3.8-2.4T-A95B-Quark-MXFP4/20260912T174806Z-6dfe2baa/geak/e2e_cycle0/kernels/Cijk_MT256x256x64_16384x4608x8192_task/selection_validation.json",
            "hk05_profiler_selection": "/shared_nfs/zepingl/qwen38_hyperloom_local_kernel_20260913T081758Z_093_multikernel_r11/hyperloom/Qwen3.8-2.4T-A95B-Quark-MXFP4/20260913T083154Z-7df2e82d/geak/e2e_cycle0/kernels/Cijk_MT16x16x1024_decode_skinny_gemm_task/selection_validation.json",
            "live_decode_config": "/shared_nfs/zepingl/qwen38_hyperloom_local_kernel_20260912T171040Z_093_multikernel_r10/baseline/aiter_tuned_configs/qwen38_24t_a95b_mxfp4_bf16_tuned_gemm.csv",
            "live_prefill_config": "/shared_nfs/zepingl/qwen38_hyperloom_local_kernel_20260912T171040Z_093_multikernel_r10/baseline/aiter_tuned_configs/qwen38_prefill_bf16_tuned_gemm.csv"
        },
        "note": "Each exact shape was resolved by the live AITER dispatcher, the selected solMap backend was observed executing, and a device-side profiler event was collected."
    }
    _write_json(prior_path, payload)
    return 0 if all(row["ok"] for row in rows.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
