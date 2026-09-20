"""Parent-only comparison of separately executed MiniMax reference/candidate outputs."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import secrets
import sys
import time

PREFIX = "MINIMAX_GENERATED_RESULT="


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_required(runner):
    contract = load("generated_contract", runner.UT_DIR / "generated_contract.py")
    contract.load_contract(runner.UT_DIR)
    meta = json.loads((runner.UT_DIR / "meta.json").read_text())["generated_inputs"]
    geometry = runner.UT_DIR / meta["geometry_file"]
    if contract.digest(geometry) != meta["geometry_sha256"]:
        raise RuntimeError("compact geometry SHA-256 mismatch")


def expected_profiles(ut, contract, meta):
    # This parent never loads a candidate. Install the same trusted helper object
    # used for contract checks so case geometry does not create an unchecked copy.
    sys.modules["generated_contract"] = contract
    # Protected case construction is imported without resolving a serving callable.
    cases = load("_minimax_profile_catalog", ut / "cases.py")
    h = load("_minimax_profile_harness", ut / "harness_lib.py")
    replay = [shape["sig"] for shape in cases.replay_shapes(h, meta)]
    if len(replay) < 2:
        raise RuntimeError("missing required replay boundary cases")
    return {"recorded": [record["sig"] for record in contract.load_contract(ut)["records"]],
            "random": [shape["sig"] for shape in cases.random_shapes(h, meta)],
            "replay": replay + replay[:1]}


def parse_worker(proc, profile, seed, reference, expected):
    if proc.returncode:
        raise RuntimeError(f"generated worker failed: {proc.stderr[-2000:]}")
    lines = [line[len(PREFIX):] for line in proc.stdout.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError("worker did not emit exactly one fresh result")
    value = json.loads(lines[0])
    if (value.get("schema_version") != 1 or value.get("profile") != profile
            or value.get("seed") != seed or value.get("reference") is not reference
            or [row.get("id") for row in value.get("rows", [])] != expected):
        raise RuntimeError("worker returned stale, incomplete or reordered cases")
    return value["rows"]


def compare_workers(reference, candidate, contract, tol, torch):
    if [row["id"] for row in reference] != [row["id"] for row in candidate]:
        raise RuntimeError("candidate changed the complete reference case set")
    for left, right in zip(candidate, reference):
        if not contract.compare_output(left["output"], right["output"], tol, torch):
            raise RuntimeError(f"generated correctness mismatch: {left['id']}")


def run_correctness(runner, cfg, timeout):
    start = time.monotonic()
    seed = secrets.randbits(62)
    report = {"input_policy": "generated_values_at_captured_contract", "seed": seed, "profiles": []}
    try:
        import torch
        contract = load("generated_contract", runner.UT_DIR / "generated_contract.py")
        meta = json.loads((runner.UT_DIR / "meta.json").read_text())
        verify_required(runner)
        expected = expected_profiles(runner.UT_DIR, contract, meta)
        baseline, candidate = runner.overlays()
        for profile in ("recorded", "random", "replay"):
            draws = int(meta["random_draws"]) if profile != "replay" else 1
            for draw in range(draws):
                current_seed = seed + draw
                answers = []
                for reference, overlay in ((True, baseline), (False, candidate)):
                    remaining = timeout - (time.monotonic() - start)
                    if remaining <= 0:
                        raise TimeoutError("generated correctness command budget exhausted")
                    arguments = ["--ut", str(runner.UT_DIR), "--profile", profile, "--seed", str(current_seed)]
                    if reference:
                        arguments.append("--reference")
                    # The only inputs passed to either worker are its contract/profile/seed.
                    # Reference results remain in this parent and are never written to disk.
                    proc = runner.run_worker(runner.TASK_DIR / "scripts/generated_worker.py", arguments,
                                             overlay, remaining, not reference, cwd=runner.TASK_DIR)
                    answers.append(parse_worker(proc, profile, current_seed, reference, expected[profile]))
                compare_workers(answers[0], answers[1], contract, float(meta["tol"]), torch)
                report["profiles"].append({"profile": profile, "draw": draw,
                                           "cases": len(expected[profile]), "correct": True})
                del answers
        report.update(status="ok", duration_seconds=time.monotonic() - start)
        runner.write_report("correctness_report.json", report)
        return True, None
    except Exception as exc:
        report.update(status="fail", error=str(exc), duration_seconds=time.monotonic() - start)
        runner.write_report("correctness_report.json", report)
        return False, str(exc)
