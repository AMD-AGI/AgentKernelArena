"""Trusted parent controller: reference outputs never enter candidate workers."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import secrets
import time

PREFIX = "GLM_GENERATED_RESULT="


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_required(runner):
    contract = load("generated_contract", runner.UT_DIR / "generated_contract.py")
    contract.load_contract(runner.UT_DIR)


def parse_worker(proc, profile, seed, reference, expected_ids):
    if proc.returncode:
        raise RuntimeError(f"generated worker failed: {proc.stderr[-2000:]}")
    lines = [x[len(PREFIX):] for x in proc.stdout.splitlines() if x.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError("expected exactly one fresh generated worker result")
    value = json.loads(lines[0])
    if (value.get("schema_version") != 1 or value.get("profile") != profile
            or value.get("seed") != seed or value.get("reference") is not reference
            or [row.get("id") for row in value.get("rows", [])] != expected_ids):
        raise RuntimeError("worker result changed seed, role, or complete ordered case set")
    return value["rows"]


def compare_rows(reference, candidate, contract, tol, torch, *, timed=False):
    if [r["id"] for r in reference] != [r["id"] for r in candidate]:
        raise RuntimeError("candidate changed reference case identities")
    for left, right in zip(candidate, reference):
        if left["inputs"] != right["inputs"]:
            raise RuntimeError(f"candidate changed input metadata: {left['id']}")
        pairs = zip(left["checks"], right["checks"]) if timed else [(left, right)]
        if timed and (len(left.get("checks", [])) != 3 or len(right.get("checks", [])) != 3):
            raise RuntimeError("timed replay must return every A/B/A output")
        for observed, expected in pairs:
            if (observed["aliases"] != expected["aliases"]
                    or not contract.compare_output(observed["output"], expected["output"], tol, torch)):
                raise RuntimeError(f"generated {'timed replay' if timed else 'correctness'} mismatch: {left['id']}")


def run_pair(runner, profile, seed, expected, remaining, baseline, candidate):
    answers = []
    for reference, overlay in ((True, baseline), (False, candidate)):
        arguments = ["--ut", str(runner.UT_DIR), "--profile", profile, "--seed", str(seed)]
        if reference:
            arguments.append("--reference")
        proc = runner.run_worker(runner.TASK_DIR / "scripts/generated_worker.py", arguments,
                                 overlay, remaining(), not reference, cwd=runner.TASK_DIR)
        answers.append(parse_worker(proc, profile, seed, reference, expected))
    return answers


def run_correctness(runner, cfg, timeout):
    start = time.monotonic()
    seed = secrets.randbits(62)
    report = {"input_policy": "generated_numeric_preserved_capture_structure", "seed": seed, "profiles": []}
    try:
        import torch
        contract = load("generated_contract", runner.UT_DIR / "generated_contract.py")
        meta = json.loads((runner.UT_DIR / "meta.json").read_text())
        verify_required(runner)
        report["qualification_scope"] = meta["generated_inputs"].get("recorded_profile_scope", "generated semantic correctness")
        baseline, candidate = runner.overlays()
        def remaining():
            value = timeout - (time.monotonic() - start)
            if value <= 0:
                raise TimeoutError("generated correctness timeout")
            return value
        # The compact catalog contains all original recorded and random case IDs.
        expected = meta["generated_inputs"]["profiles"]
        for profile in ("recorded", "random", "semantic_call"):
            if not expected.get(profile):
                continue
            for draw in range(int(meta["random_draws"])):
                answers = run_pair(runner, profile, seed + draw, expected[profile], remaining, baseline, candidate)
                compare_rows(*answers, contract, float(meta["tol"]), torch)
                report["profiles"].append({"profile": profile, "draw": draw,
                                           "cases": len(expected[profile]), "correct": True})
                del answers
        report.update(status="ok", duration_seconds=time.monotonic()-start)
        runner.write_report("correctness_report.json", report)
        return True, None
    except Exception as exc:
        report.update(status="fail", error=str(exc))
        runner.write_report("correctness_report.json", report)
        return False, str(exc)


def run_performance(runner, cfg, timeout):
    start = time.monotonic()
    seed = secrets.randbits(62)
    def remaining():
        value = timeout - (time.monotonic()-start)
        if value <= 0:
            raise TimeoutError("generated performance timeout")
        return value
    try:
        import torch
        contract = load("generated_contract", runner.UT_DIR / "generated_contract.py")
        meta = json.loads((runner.UT_DIR / "meta.json").read_text())
        contract.require_complete_timing(meta)
        ok, error = run_correctness(runner, cfg, remaining())
        if not ok:
            raise RuntimeError(error)
        expected = meta["generated_inputs"]["profiles"]["timed"]
        baseline, candidate = runner.overlays()
        reference, measured = run_pair(runner, "timed", seed, expected, remaining, baseline, candidate)
        compare_rows(reference, measured, contract, float(meta["tol"]), torch, timed=True)
        raw = {"status": "ok", "expected_case_ids": expected, "seed": seed,
               "warmup_iterations": 10, "benchmark_iterations": 100,
               "test_cases": [row["timing"] for row in measured],
               "comparison_location": "independent_parent_memory"}
        rows = runner.validate_performance_report(raw, expected)
        runner.write_report("performance_report.json", raw)
        print(f"Performance: measured {len(rows)} complete generated-input cases")
        return rows
    except Exception as exc:
        runner.write_report("performance_report.json", {"status": "fail", "error": str(exc), "test_cases": []})
        print(f"Performance: FAILED: {exc}")
        return []
