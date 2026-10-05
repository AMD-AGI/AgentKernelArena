"""Trusted, CPU-only coordinator for isolated native evaluation workers."""

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

from native import source_identity
from oracle import load_cases


ROOT = Path(__file__).resolve().parents[1]
LEGS = ("production_native", "candidate_native")
WARMUP = 10
ITERATIONS = 100
SEEDS = (0, 1)


def package_identity():
    digest = hashlib.sha256()
    paths = [ROOT / "cases.json", ROOT / "config.yaml"]
    for directory in ("source", "ut", "scripts", "provenance"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("evaluation inputs must be regular in-workspace files")
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sample_statistics(samples):
    if not isinstance(samples, list) or len(samples) != ITERATIONS:
        raise ValueError("incomplete graph timing samples")
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in samples):
        raise ValueError("all device samples must be finite and positive")
    return {"mean_ms": math.fsum(samples) / len(samples),
            "min_ms": min(samples), "max_ms": max(samples), "samples_ms": samples}


def validate_worker(payload, request, pid, cases):
    for key in ("run_id", "mode", "leg", "package_sha256", "source_tree_sha256", "challenge_seed"):
        if payload.get(key) != request[key]:
            raise ValueError("stale or foreign worker result: " + key)
    if payload.get("status") != "ok" or payload.get("pid") != pid or pid == os.getpid():
        raise ValueError("native evaluation requires a successful isolated worker")
    proof = payload.get("native_build", {})
    if proof.get("leg") != request["leg"] or proof.get("production_namespace_rebound") is not False:
        raise ValueError("missing native engagement proof")
    extension = Path(proof.get("extension_path", ""))
    if not extension.is_absolute():
        extension = ROOT / extension
    if request["leg"] == "candidate_native":
        if (proof.get("fresh_compilation") is not True
                or proof.get("source_tree_sha256") != request["source_tree_sha256"]
                or not extension.resolve().is_relative_to((ROOT / "build/aiter_jit").resolve())
                or not proof.get("candidate_module", "").startswith("module_quant_aka_candidate_")
                or proof["candidate_module"] not in extension.name):
            raise ValueError("candidate must use its freshly compiled current source")
    elif extension.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("baseline extension must be supplied by the pinned runtime")
    if not extension.is_file() or hashlib.sha256(extension.read_bytes()).hexdigest() != proof.get("extension_sha256"):
        raise ValueError("native extension is missing or changed")
    rows = payload.get("results")
    expected = [] if request["mode"] == "compile" else [case["case_id"] for case in cases]
    if not isinstance(rows, list) or [row.get("case_id") for row in rows] != expected:
        raise ValueError("incomplete or duplicate workload coverage")
    for case, row in zip(cases, rows):
        if row.get("shape") != case["shape"] or row.get("trace_call_count") != case["trace_call_count"]:
            raise ValueError("worker changed workload shape or weight")
        if row.get("correct") is not True or row.get("input_immutable") is not True:
            raise ValueError("native oracle checks did not pass")
        if request["mode"] == "correctness":
            if row.get("seeds") != list(SEEDS) or row.get("negative_controls") is not True:
                raise ValueError("both seeds and negative controls are mandatory")
        elif request["mode"] == "performance":
            if (row.get("fresh_input_replays") != ITERATIONS
                    or row.get("poisoned_output_replays") != ITERATIONS
                    or row.get("oracle_checks") != ITERATIONS + 1
                    or row.get("warmup_iterations") != WARMUP
                    or row.get("benchmark_method") != "cuda_graph"):
                raise ValueError("graph replay freshness and full validation are mandatory")
            actual = row.get("timings", {})
            statistics = sample_statistics(actual.get("samples_ms"))
            if actual != statistics:
                raise ValueError("timing summary does not match its device samples")
    return payload


def run_workers(mode):
    manifest, cases = load_cases(ROOT / "cases.json")
    source_sha = source_identity()  # Enforce the GPU-only edit boundary before any subprocess.
    package_sha = package_identity()
    build = ROOT / "build"
    build.mkdir(parents=True, exist_ok=True)
    run_id = secrets.token_hex(24)
    # Both legs receive identical values. Fresh per-run seeds are independent of
    # the candidate; correctness retains mandatory public seeds 0 and 1 too.
    challenge_seed = secrets.randbelow(2**31)
    order = LEGS if secrets.randbits(1) else tuple(reversed(LEGS))
    results = {}
    with tempfile.TemporaryDirectory(prefix="native_eval_", dir=build) as temporary:
        directory = Path(temporary)
        for leg in order:
            request = {"run_id": run_id, "mode": mode, "leg": leg,
                       "package_sha256": package_sha, "source_tree_sha256": source_sha,
                       "challenge_seed": challenge_seed}
            request_path = directory / (leg + "_request.json")
            result_path = directory / (leg + "_result.json")
            request_path.write_text(json.dumps(request))
            # Keep worker/compiler output out of score-parsing stdout. A worker
            # cannot gain a score by printing a plausible timing or PASS line.
            log_path = build / (mode + "_" + leg + ".log")
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-I", str(ROOT / "scripts/worker.py"),
                     str(request_path), str(result_path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("native worker timed out: " + leg)
            if code != 0:
                raise RuntimeError(f"native worker {leg} failed ({code}); see {log_path.relative_to(ROOT)}")
            if not result_path.is_file() or result_path.is_symlink():
                raise ValueError("native worker produced no fresh result")
            if package_identity() != package_sha or source_identity() != source_sha:
                raise ValueError("task inputs changed during native evaluation")
            results[leg] = validate_worker(json.loads(result_path.read_text()), request, process.pid, cases)
    if len({result["pid"] for result in results.values()}) != 2:
        raise ValueError("baseline and candidate must execute in different processes")
    return manifest, cases, results, {"run_id": run_id, "package_sha256": package_sha,
                                      "source_tree_sha256": source_sha, "leg_order": list(order)}


def performance_report(manifest, cases, results, identity):
    paired = []
    for index, case in enumerate(cases):
        paired.append({"case_id": case["case_id"], "shape": case["shape"],
                       "trace_call_count": case["trace_call_count"], "graph_correctness": True,
                       "timings": {leg: results[leg]["results"][index]["timings"] for leg in LEGS}})
    weight_sum = sum(case["trace_call_count"] for case in cases)
    weighted = {leg: math.fsum(row["trace_call_count"] * row["timings"][leg]["mean_ms"]
                              for row in paired) / weight_sum for leg in LEGS}
    return {"status": "ok", "task_validator_status": "pending", **identity,
            "source_run": manifest["source_run"], "count_scope": manifest["count_scope"],
            "benchmark_method": "cuda_graph", "warmup_iterations": WARMUP,
            "benchmark_iterations": ITERATIONS, "isolated_native_processes": True,
            "leg_order_policy": "randomized per invocation; sequential isolated workers",
            "fresh_input_and_poisoned_output_each_replay": True,
            "native_builds": {leg: results[leg]["native_build"] for leg in LEGS},
            "worker_pids": {leg: results[leg]["pid"] for leg in LEGS},
            "weight_sum_per_rank": weight_sum, "weighted_mean_ms": weighted, "paired_cases": paired,
            # Arena scoring uses the arithmetic mean of matched per-case
            # speedup ratios. Observed-call-weighted timings are diagnostics.
            "test_cases": [{"test_case_id": row["case_id"],
                            "execution_time_ms": row["timings"]["candidate_native"]["mean_ms"],
                            "shape": row["shape"],
                            "params": {"trace_call_count": row["trace_call_count"]},
                            "metadata": {"benchmark_method": "cuda_graph"}}
                           for row in paired]}


def write_report(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
