#!/usr/bin/env python3
"""Literal native-routing contract and checked canonical _aka_benchmark worker.

The protected worker owns graph capture/validation. Every run retains its full
raw cases under build/native-routing-runs; no legacy event/median fallback runs.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(TASK_DIR, "build")
UT_DIR = os.path.join(TASK_DIR, "ut")
CONFIG = os.path.join(TASK_DIR, "config.yaml")
ROUTING_PROTOCOL = "qwen_native_route_graph_v1"
ROUTING_CONTRACT_SHA256 = "89dc2f7a012c442e911fd187847e64848d65ed4e05b63d798ce6b275bd83dcf4"


def load_config():
    import yaml
    return yaml.safe_load(Path(CONFIG).read_text())

def _routing_contract():
    path = Path(TASK_DIR) / "scripts/_routing_contract.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != ROUTING_CONTRACT_SHA256:
        raise ValueError("protected native-routing validator changed")
    spec = importlib.util.spec_from_file_location("_protected_dense_routing_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _routing_enabled():
    return hk(load_config(), "evaluation_protocol") == ROUTING_PROTOCOL

def _routing_cases(raw):
    contract = _routing_contract()
    task = Path(TASK_DIR)
    meta = contract.strict_json((task / "ut/meta.json").read_text())
    request = contract.strict_json((Path(BUILD_DIR) / "performance_request.json").read_text())
    contract.validate_request(request, task, "performance")
    rows = _validate_dense_payload(contract, raw, meta, request)
    return [{"test_case_id": row["sig"], "execution_time_ms": row["mean_ms"],
             "params": row["params"], "samples_ms": row["samples_ms"],
             "median_ms": row["median_ms"], "min_ms": row["min_ms"], "max_ms": row["max_ms"],
             "selected_route": row["selected_route"], "validation": row["validation"],
             **{key: value for key, value in row.items() if key.startswith("benchmark_")}}
            for row in rows]

def _validate_dense_payload(contract, raw, meta, request):
    rows = contract.validate_payload(raw, meta, request)
    for row, spec in zip(rows, contract.specs(meta)):
        validation = row["validation"]
        if request["mode"] == "performance":
            contract.require(all(type(value) is int for value in (raw.get("warmup"), raw.get("iters"),
                row.get("benchmark_warmup"), row.get("benchmark_samples"), row.get("benchmark_effective_repeats"))),
                "graph count metadata must contain integers, not boolean flags")
        binding = validation.get("native_binding", {})
        route = request["source"]["routes"][spec["sig"]]
        backend = spec["expected_backend"] if route == "recorded" else route
        names = {"torch": "torch_gemm", "triton": "triton_gemm", "hipblaslt": "hipb_gemm",
                 "asm": "asm_gemm", "flydsl": "flydsl_gemm"}
        contract.require(binding.get("native_source_sha256") == request["source"]["native_source_sha256"]
            and binding.get("backend") == backend and binding.get("function") == names[backend]
            and binding.get("editable_python_executed") is False,
            "actual native backend/source binding differs from the literal route")
        expected_config = ({"libtype": backend, "solidx": int(spec["expected_solidx"]),
                            "splitK": int(spec["expected_split_k"]), "kernelName": spec["expected_kernel"]}
                           if route == "recorded" else
                           {"libtype": backend, "solidx": 0, "splitK": 0, "kernelName": "native"})
        observed_config = binding.get("native_config", {})
        contract.require(observed_config == expected_config and all(type(observed_config[key]) is type(value)
                         for key, value in expected_config.items()), "native configuration differs from the route")
        contract.require(type(binding.get("python_invocations")) is int and binding["python_invocations"] >= 2,
                         "native backend was not engaged")
        if request["mode"] == "correctness":
            contract.require(validation.get("checked_invocations") == 2 + int(meta["random_draws"]) + 9
                and validation.get("graph_checked_invocations") == 9
                and validation.get("measured_graph_validated") is True
                and validation.get("prior_outputs_unchanged") is True
                and validation.get("independent_output_storage") is True,
                "dense correctness omitted original draws or checked graph replay")
    return rows

def _run_routing(mode, timeout):
    contract = _routing_contract()
    task, build = Path(TASK_DIR), Path(BUILD_DIR)
    request = None
    try:
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("positive phase timeout required")
        request = contract.make_request(task, mode, secrets.token_hex(24))
        meta = contract.strict_json((task / "ut/meta.json").read_text())
        request_path = build / (mode + "_request.json")
        request_path.write_text(json.dumps(request, indent=2) + "\n")
        if mode == "compile":
            write_report("compile_report.json", {"status": "ok", "request": request,
                "protocol": ROUTING_PROTOCOL, "parsed_sources": [contract.POLICY_FILE],
                "symbols_found": {"select_routes": contract.POLICY_FILE},
                "validation_kind": "literal route policy and frozen native source; native kernels remain image-owned",
                "editable_python_executed": False})
            print("Compilation: PASS (literal native-routing policy)")
            return True
        run_dir = build / "native-routing-runs" / request["run_id"]
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        raw_path = run_dir / "raw.json"
        command = [sys.executable, "-I", "-B", str(task / "scripts/_bench.py"),
                   "--ut", str(task / "ut"), "--out", str(raw_path), "--request", str(request_path),
                   "--warmup", "10", "--iters", "100"]
        if mode == "correctness":
            command.append("--correctness-only")
        env = dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", LD_PRELOAD="")
        with (run_dir / "worker.log").open("x") as log:
            process = subprocess.Popen(command, cwd=task, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
                raise RuntimeError("native-routing phase timed out; no fallback or retry")
        if code != 0:
            raise RuntimeError(f"native-routing worker failed ({code}); diagnostics: {run_dir}")
        contract.validate_request(request, task, mode)
        raw = contract.strict_json(raw_path.read_text())
        rows = _validate_dense_payload(contract, raw, meta, request)
        (build / "reports").mkdir(exist_ok=True)
        write_report("reports/native_selection_" + mode + ".json", {
            "status": "PASS", "request": request,
            "cases": [{"case_id": row["sig"], "selected_route": row["selected_route"],
                       "native_binding": row["validation"]["native_binding"]} for row in rows]})
        if mode == "correctness":
            write_report("reports/negative_check.json", {"status": "PASS", "request": request,
                "cases": [{"case_id": row["sig"], "controls": row["validation"]["negative_controls"]} for row in rows]})
        if mode == "performance":
            published_raw = build / "_bench_raw.json"
            published_raw.write_bytes(raw_path.read_bytes())
            cases = _routing_cases(raw)
            write_report("performance_report.json", {"status": "ok", "protocol": ROUTING_PROTOCOL,
                "timer": "cuda_graph", "warmup_iterations": 10, "benchmark_iterations": 100,
                "request": request, "test_cases": cases, "complete_raw_report": str(raw_path.relative_to(task)),
                "methodology": "Canonical graph replay, fresh CPU-owned inputs and checked outputs for every invocation; all100ordered samples retained",
                "optimization_surface": "constrained image-native routing", "standalone_gain_claim": False})
        else:
            write_report("correctness_report.json", {"status": "ok", "protocol": ROUTING_PROTOCOL,
                "request": request, "case_count": len(rows), "case_results": rows,
                "complete_raw_report": str(raw_path.relative_to(task)),
                "oracle": "independent F.linear after candidate output/input CPU snapshots"})
        print(f"{mode.capitalize()}: PASS ({len(rows)} complete cases)")
        return True
    except Exception as error:
        write_report(mode + "_report.json", {"status": "fail", "protocol": ROUTING_PROTOCOL,
            "request": request, "error": str(error), "test_cases": [], "fallback_used": False})
        print(f"{mode.capitalize()}: FAIL - {error}")
        return False

def hk(cfg, field, default=None):
    """Read a field from the task's ``headkernel:`` provenance block."""
    block = cfg.get("headkernel")
    if isinstance(block, dict):
        return block.get(field, default)
    return default

def write_report(name, payload):
    os.makedirs(BUILD_DIR, exist_ok=True)
    with open(os.path.join(BUILD_DIR, name), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

def candidate_overlay():
    """Build the GEAK candidate overlay so the timed callable resolves to source/.

    Returns the overlay dir, or None when this op has no rebind seam (then the
    live stack is timed directly, which is also what its own UT does).
    """
    meta_path = os.path.join(UT_DIR, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    meta = json.load(open(meta_path))
    if not (meta.get("candidate_bind") or {}).get("file"):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "geak_harness_lib", os.path.join(UT_DIR, "harness_lib.py"))
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    _base, cand = h.build_candidate_overlay(UT_DIR, meta)
    return cand

def _benchmark_cases(raw):
    if not _routing_enabled():
        raise ValueError("the protected native-routing contract is required")
    return _routing_cases(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "correctness", "performance"))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("HK_TASK_TIMEOUT", "1800")))
    args = parser.parse_args()
    os.makedirs(BUILD_DIR, exist_ok=True)
    config = load_config()
    if (not _routing_enabled() or config.get("source_file_path") != ["source/routing.py"]
            or config.get("target_kernel_functions") != ["select_routes"]):
        raise ValueError("unsupported native routing source contract")
    raise SystemExit(0 if _run_routing(args.mode, args.timeout) else 1)


if __name__ == "__main__":
    main()
