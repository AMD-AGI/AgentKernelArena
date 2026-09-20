#!/usr/bin/env python3
"""Protected compile, correctness and complete-case GPU benchmark entrypoints.

Performance delegates to scripts/_bench.py and its materialized _aka_benchmark
helper. Both run in the selected overlay's fresh worker process.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import secrets
import subprocess
import sys
import time

TASK_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = TASK_DIR / "build"
UT_DIR = TASK_DIR / "ut"
CONFIG = TASK_DIR / "config.yaml"
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


def load_config():
    import yaml
    with CONFIG.open() as handle:
        return yaml.safe_load(handle)


def write_report(name, payload):
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    path = BUILD_DIR / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


def source_abi(cfg):
    """Function argument names/defaults/annotations are part of the fixed ABI."""
    targets = set(cfg["target_kernel_functions"])
    signatures = {}
    for relative in cfg["source_file_path"]:
        path = TASK_DIR / relative
        tree = ast.parse(path.read_text(), filename=relative)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in targets:
                signatures.setdefault(node.name, []).append(
                    {"source": relative, "arguments": ast.dump(node.args, include_attributes=False)})
    return signatures


def run_compile(cfg):
    try:
        signatures = source_abi(cfg)
        missing = set(cfg["target_kernel_functions"]) - set(signatures)
        if missing:
            raise RuntimeError(f"target functions are not defined: {sorted(missing)}")
        expected = json.loads((TASK_DIR / "scripts/source_abi.json").read_text())
        if signatures != expected:
            raise RuntimeError("target function ABI changed: preserve argument names, order and defaults")
        # Preserve extra baseline entrypoints required by the package overlays.
        required = (cfg.get("headkernel") or {}).get("preserve_symbols") or []
        names = set()
        for relative in cfg["source_file_path"]:
            tree = ast.parse((TASK_DIR / relative).read_text(), filename=relative)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.add(node.id)
        if set(required) - names:
            raise RuntimeError(f"required harness symbols removed: {sorted(set(required) - names)}")
        write_report("compile_report.json", {"status": "ok", "validation": "python_ast_and_fixed_abi",
                                              "parsed_sources": cfg["source_file_path"]})
        return True, None
    except Exception as exc:
        write_report("compile_report.json", {"status": "fail", "error": str(exc)})
        return False, str(exc)


def verify_fixtures():
    """Verify every persistent artifact before any package can torch.load it."""
    entries = json.loads((TASK_DIR / "scripts/artifacts.json").read_text())
    meta = json.loads((UT_DIR / "meta.json").read_text())
    expected_keys = {"reference_io.pt": "reference_io_sha256",
                     "timing_geometry.pt": "timing_geometry_sha256"}
    declared = {key for key in expected_keys.values() if meta.get(key)}
    if {expected_keys[entry["filename"]] for entry in entries} != declared:
        raise RuntimeError("persistent fixture manifest disagrees with ut/meta.json")
    for entry in entries:
        filename = entry["filename"]
        if filename not in expected_keys or meta[expected_keys[filename]] != entry["sha256"]:
            raise RuntimeError("persistent fixture hash declaration differs from captured metadata")
        path = UT_DIR / filename
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"persistent fixture is absent or not a regular file: ut/{filename}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if before.st_size != entry["size_bytes"]:
                raise RuntimeError(f"fixture size mismatch: ut/{filename}")
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
        if ((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or digest.hexdigest() != entry["sha256"]):
            raise RuntimeError(f"fixture SHA-256 mismatch or changed during verification: ut/{filename}")


def load_harness():
    spec = importlib.util.spec_from_file_location("harness_lib", UT_DIR / "harness_lib.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["harness_lib"] = module
    spec.loader.exec_module(module)
    return module


def overlays():
    meta = json.loads((UT_DIR / "meta.json").read_text())
    if (meta.get("candidate_bind") or {}).get("file"):
        return load_harness().build_candidate_overlay(str(UT_DIR), meta)
    # The protected attention adapter constructs both functions directly. Its
    # candidate factory needs the captured reference's geometry helpers; the
    # installed serving callable is not a substitute for that reference.
    if (UT_DIR / "bindings.py").is_file():
        for relative in ("bindings.py", "baseline_ref/decode_attention.py.orig",
                         "kernel_src/geak_mla_stage1.py"):
            if not (UT_DIR / relative).is_file():
                raise RuntimeError(f"protected attention binding is missing: ut/{relative}")
        return None, None
    # The FlyDSL adapter loads the captured MoE dependency closure under two
    # independent namespaces. It deliberately bypasses package __init__ API
    # reexports, so the MoE module and manifest are the relevant prerequisites.
    if meta.get("entry_attr") in {"flydsl_moe_stage1", "flydsl_moe_stage2"}:
        for relative in ("flydsl_package.py", "dependency_manifest.json",
                         "baseline_src/flydsl/moe_kernels.py", "kernel_src/flydsl/moe_kernels.py"):
            if not (UT_DIR / relative).is_file():
                raise RuntimeError(f"independent Kimi package is missing: ut/{relative}")
        if (UT_DIR / "baseline_src/flydsl").resolve() == (UT_DIR / "kernel_src/flydsl").resolve():
            raise RuntimeError("frozen and editable FlyDSL packages resolve to the same source tree")
        return None, None
    raise RuntimeError("candidate_bind is missing; refusing to benchmark the unmodified runtime")


def worker_env(overlay, candidate):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # A task must not inherit another task's overlay or a user's capture hook.
    env.pop("PYTHONPATH", None)
    env.pop("GEAK_ACTIVE_TASK_CANDIDATE", None)
    if overlay:
        env["PYTHONPATH"] = str(overlay)
    if candidate:
        env["GEAK_ACTIVE_TASK_CANDIDATE"] = str(UT_DIR)
    return env


def run_process(command, timeout, env, cwd=TASK_DIR):
    """Kill the whole UT worker group when its command budget expires."""
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def worker_command(script, arguments, overlay, completion=None, nonce=None, attest_files=()):
    command = [sys.executable, "-u", str(TASK_DIR / "scripts/_trusted_worker.py"),
               "--task-root", str(TASK_DIR), "--script", str(script)]
    if overlay:
        command.extend(["--overlay", str(overlay)])
    if completion is not None:
        command.extend(["--completion", str(completion), "--nonce", nonce])
        for path in attest_files:
            command.extend(["--attest-file", str(path)])
    return command + ["--", *arguments]


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_worker(script, arguments, overlay, timeout, candidate, *, cwd=TASK_DIR, attest_files=()):
    """Require trusted worker finalization, not just exit zero and a JSON file."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32)
    completion = BUILD_DIR / f"_worker_completion_{nonce}.json"
    command = worker_command(script, arguments, overlay, completion, nonce, attest_files)
    try:
        proc = run_process(command, timeout, worker_env(None, candidate), cwd=cwd)
        if proc.returncode:
            return proc
        try:
            receipt = json.loads(completion.read_text())
            expected = {str(Path(path).resolve()): file_digest(path) for path in attest_files}
            if (receipt.get("nonce") != nonce or receipt.get("script") != str(Path(script).resolve())
                    or receipt.get("status") != "complete" or receipt.get("outputs") != expected):
                raise RuntimeError("worker completion or artifact digest does not match")
        except Exception as exc:
            return subprocess.CompletedProcess(command, 1, proc.stdout,
                                               proc.stderr + f"\nTrusted worker did not finalize: {exc}")
        return proc
    finally:
        completion.unlink(missing_ok=True)


def run_ut(timeout):
    _, candidate = overlays()
    started = time.monotonic()
    proc = run_worker(UT_DIR / "unittest.py", [], candidate, timeout, True, cwd=UT_DIR)
    return proc, time.monotonic() - started


def run_correctness(cfg, timeout):
    try:
        proc, seconds = run_ut(timeout)
        output = proc.stdout + proc.stderr
        if proc.returncode:
            raise RuntimeError(f"ut/unittest.py exited {proc.returncode}: {output[-3000:]}")
        write_report("correctness_report.json", {
            "status": "ok", "exit_code": 0, "duration_seconds": seconds,
            "oracle": (cfg.get("headkernel") or {}).get("oracle"),
            "stdout_tail": output.splitlines()[-30:],
        })
        return True, None
    except Exception as exc:
        write_report("correctness_report.json", {"status": "fail", "error": str(exc)})
        return False, str(exc)


def validate_performance_report(raw, expected_ids=None):
    rows = raw.get("test_cases") or []
    expected = raw.get("expected_case_ids") or []
    ids = [row.get("test_case_id") for row in rows]
    if raw.get("status") != "ok" or not expected or ids != expected or len(ids) != len(set(ids)):
        raise RuntimeError("benchmark report does not contain the complete expected case set")
    if expected_ids is not None and expected != expected_ids:
        raise RuntimeError("benchmark case set differs from the baseline worker's declaration")
    for row in rows:
        latency = row.get("execution_time_ms")
        if (not isinstance(latency, (int, float)) or isinstance(latency, bool)
                or not math.isfinite(latency) or latency <= 0):
            raise RuntimeError("benchmark report contains an invalid device time")
        if (row.get("benchmark_method") != "cuda_graph"
                or row.get("benchmark_output_validation") != "exact_timed_graph_replay"
                or row.get("benchmark_replay_probe") != "input_change_and_restore"
                or row.get("benchmark_state_restore") != "all_input_storages_before_each_replay"
                or row.get("benchmark_method_consistent") is not True
                or row.get("benchmark_samples") != BENCHMARK_ITERATIONS):
            raise RuntimeError("benchmark report lacks canonical timing or exact replay validation")
    return rows


def run_performance(cfg, timeout):
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    report = BUILD_DIR / "performance_report.json"
    raw_path = BUILD_DIR / "_bench_raw.json"
    reference = BUILD_DIR / "_benchmark_reference.pt"
    reference_cases = reference.with_suffix(".cases.json")
    for path in (report, raw_path, reference, reference_cases):
        path.unlink(missing_ok=True)
    started = time.monotonic()

    def remaining():
        budget = timeout - (time.monotonic() - started)
        if budget <= 0:
            raise TimeoutError("performance command budget exhausted")
        return budget

    try:
        # Standalone performance is gated too; a passing compile or an old UT
        # report never authorizes timing a candidate that fails correctness.
        ok, error = run_correctness(cfg, remaining())
        if not ok:
            raise RuntimeError(error)
        baseline, candidate = overlays()
        arguments = ["--ut", str(UT_DIR), "--out", str(raw_path),
                   "--reference", str(reference), "--warmup", str(WARMUP_ITERATIONS),
                   "--iters", str(BENCHMARK_ITERATIONS)]
        expected_ids = None
        for phase, overlay in (("reference", baseline), ("measure", candidate)):
            attest_files = (reference, reference_cases) if phase == "reference" else (raw_path,)
            proc = run_worker(TASK_DIR / "scripts/_bench.py", arguments + ["--phase", phase],
                              overlay, remaining(), phase == "measure", attest_files=attest_files)
            if proc.returncode:
                raise RuntimeError(f"{phase} worker exited {proc.returncode}: "
                                   f"{(proc.stdout + proc.stderr)[-3000:]}")
            if phase == "reference":
                # Hold this declaration in the trusted parent before candidate
                # execution; candidate-generated JSON cannot reduce the set.
                expected_ids = json.loads(reference_cases.read_text())
                if not expected_ids or len(expected_ids) != len(set(expected_ids)):
                    raise RuntimeError("baseline worker declared an invalid case set")
        raw = json.loads(raw_path.read_text())
        rows = validate_performance_report(raw, expected_ids)
        write_report("performance_report.json", raw)
        print(f"Performance: measured {len(rows)} complete test cases")
        return rows
    except Exception as exc:
        write_report("performance_report.json", {"status": "fail", "error": str(exc),
                                                  "test_cases": []})
        print(f"Performance: FAILED: {exc}")
        return []
    finally:
        # The transient live-baseline outputs are never reused or shipped.
        reference.unlink(missing_ok=True)
        reference_cases.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    parser.add_argument("--timeout", type=int)
    args = parser.parse_args()
    cfg = load_config()
    timeout = args.timeout or cfg.get(f"{args.mode}_timeout", 3600)
    if args.mode == "compile":
        ok, error = run_compile(cfg)
    else:
        try:
            verify_fixtures()
            from runtime_preflight import require_runtime
            require_runtime(cfg)
        except Exception as exc:
            write_report(f"{args.mode}_report.json", {"status": "fail", "error": str(exc),
                                                       "test_cases": []})
            print(f"Environment: FAILED: {exc}")
            return 1
        if args.mode == "correctness":
            ok, error = run_correctness(cfg, timeout)
        else:
            return 0 if run_performance(cfg, timeout) else 1
    print(f"{args.mode.capitalize()}: {'PASS' if ok else 'FAIL'}")
    if error:
        print(error)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
