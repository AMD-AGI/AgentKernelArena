"""Parent-only Qwen comparison; expected output bytes never enter candidate workers."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import secrets
import sys
import time

PREFIX = "QWEN_GENERATED_RESULT="


def load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != Path(path).resolve():
            raise RuntimeError(f"trusted alias names another file: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_required(runner):
    contract = load("_qwen_parent_contract_" + runner.UT_DIR.parent.name,
                    runner.UT_DIR / "generated_contract.py")
    contract.load_contract(runner.UT_DIR)


def expected_profiles(meta):
    cases = meta["workload"]["cases"]
    ids = meta["generated_inputs"]["case_ids"]
    if len(ids) != meta["num_cases"] or len(ids) != len(set(ids)):
        raise ValueError("metadata does not identify every generated case exactly once")
    if set(ids) != {row["sig"] for row in cases}:
        raise ValueError("generated cases differ from the complete workload inventory")
    result = {"eager": [f"{case}|call{call}" for case in ids for call in range(2)]}
    if meta["generated_inputs"]["kernel"] == "fused_recurrent_gated_delta_rule_decode":
        descending = [row["sig"] for row in sorted(cases, key=lambda row: row["m"], reverse=True)]
        ascending = list(reversed(descending))
        result["replay"] = [f"{case}|replay{i}" for i, case in enumerate(descending + descending[:1])]
        result["transitions"] = [f"{case}|transition{i}" for i, case in enumerate(ascending + ascending[:1])]
    else:
        result["replay"] = [f"{case}|replay{i}" for case in ids for i in range(3)]
    return result


def parse_worker(proc, profile, seed, reference, expected):
    if proc.returncode:
        raise RuntimeError(f"generated worker failed: {proc.stderr[-2000:]}")
    lines = [line[len(PREFIX):] for line in proc.stdout.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError("worker did not emit exactly one completed result")
    value = json.loads(lines[0])
    if (value.get("schema_version") != 1 or value.get("profile") != profile
            or value.get("seed") != seed or value.get("reference") is not reference
            or [row.get("id") for row in value.get("rows", [])] != expected):
        raise RuntimeError("worker returned stale, incomplete or reordered cases")
    return value["rows"]


def compare_workers(reference, candidate, contract, tol, torch):
    if [row["id"] for row in reference] != [row["id"] for row in candidate]:
        raise RuntimeError("candidate changed the complete case set")
    for observed, expected in zip(candidate, reference):
        if not contract.compare_output(observed["output"], expected["output"], tol, torch):
            raise RuntimeError(f"generated correctness mismatch: {observed['id']}")


def run_correctness(runner, cfg, timeout):
    started = time.monotonic()
    seed = secrets.randbits(62)
    report = {"input_policy": "generated_values_at_captured_contract", "seed": seed, "profiles": []}
    try:
        import torch
        contract = load("_qwen_parent_contract_" + runner.UT_DIR.parent.name,
                        runner.UT_DIR / "generated_contract.py")
        meta = json.loads((runner.UT_DIR / "meta.json").read_text())
        verify_required(runner)
        expected = expected_profiles(meta)
        baseline, candidate = runner.overlays()
        for profile, ids in expected.items():
            draws = int(meta["random_draws"]) if profile == "eager" else 1
            for draw in range(draws):
                answers = []
                current_seed = seed + draw
                for reference, overlay in ((True, baseline), (False, candidate)):
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("generated correctness budget exhausted")
                    arguments = ["--ut", str(runner.UT_DIR), "--profile", profile, "--seed", str(current_seed)]
                    if reference:
                        arguments.append("--reference")
                    proc = runner.run_worker(runner.TASK_DIR / "scripts/generated_worker.py", arguments,
                                             overlay, remaining, not reference, cwd=runner.TASK_DIR)
                    answers.append(parse_worker(proc, profile, current_seed, reference, ids))
                compare_workers(answers[0], answers[1], contract, float(meta["tol"]), torch)
                report["profiles"].append({"profile": profile, "draw": draw, "cases": len(ids), "correct": True})
                del answers
        report.update(status="ok", duration_seconds=time.monotonic() - started)
        runner.write_report("correctness_report.json", report)
        return True, None
    except Exception as exc:
        report.update(status="fail", error=str(exc), duration_seconds=time.monotonic() - started)
        runner.write_report("correctness_report.json", report)
        return False, str(exc)
