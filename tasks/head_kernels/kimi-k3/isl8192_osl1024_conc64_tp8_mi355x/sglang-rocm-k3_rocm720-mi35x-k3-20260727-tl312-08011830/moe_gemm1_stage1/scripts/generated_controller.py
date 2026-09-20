"""Parent-only correctness and timing validation for generated Kimi inputs."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import secrets
import time

PREFIX = "KIMI_GENERATED_RESULT="


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_required(runner):
    load("generated_contract", runner.UT_DIR / "generated_contract.py").load_contract(runner.UT_DIR)


def expected_profiles(meta, compact):
    recorded = [record["sig"] for record in compact["records"]]
    if compact["kind"] == "attention":
        by_id = {case.get("sig") or case.get("name"): case for case in meta["cases"]}
        online = []
        for entry in meta["workload"]["cases"]:
            case = by_id[entry["name"]]
            if case.get("timing", True):
                online.append((entry["name"], entry.get("regime") or case.get("regime") or "decode"))
        random = [name for name, _ in online]
        replay = [json.dumps([name, regime], separators=(",", ":")) for name, regime in online]
    else:
        random = [case["sig"] for case in meta["case_specs"]]
        by_id = {case["sig"]: case for case in meta["case_specs"]}
        online = []
        for entry in meta["workload"]["cases"]:
            spec = entry.get("case_spec") or by_id[entry["name"]]
            online.append((spec["sig"], entry.get("regime") or spec.get("regime", "prefill")))
        replay = [case["sig"] for case in sorted(meta["replay_specs"], key=lambda row: int(row["token_num"]), reverse=True)]
        replay += replay[:1]
    sequence = [recorded[index] for index in meta.get("call_sequence_idx", [0, 1, 0])]
    return {"single_launch": [f"{spec['sig']}:single:{trial}" for spec in meta.get('case_specs', []) for trial in range(int(meta['random_draws']))], "recorded": recorded, "random": random, "sequence": sequence, "replay": replay,
            "performance": [json.dumps([name, regime], separators=(",", ":")) for name, regime in online]}


def parse_worker(proc, profile, seed, reference, expected, median):
    if proc.returncode:
        raise RuntimeError(f"generated worker failed: {proc.stderr[-2000:]}")
    lines = [line[len(PREFIX):] for line in proc.stdout.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise RuntimeError("generated worker did not emit exactly one complete result")
    data = json.loads(lines[0])
    if (data.get("schema_version") != 1 or data.get("profile") != profile or data.get("seed") != seed
            or data.get("reference") is not reference or [row.get("id") for row in data.get("rows", [])] != expected):
        raise RuntimeError("generated worker returned stale, omitted, reordered or invented cases")
    if median and profile != "single_launch" and (data.get("correctness_policy") != "elementwise_median_21"
                   or data.get("single_launch_correctness_established") is not False):
        raise RuntimeError("stage-1 median policy was confused with single-launch correctness")
    if profile == "single_launch" and data.get("correctness_policy") != "independent_torch_single_launch":
        raise RuntimeError("single-launch qualification was replaced by a median")
    return data["rows"]


def compare_rows(reference, candidate, contract, tol, torch):
    if [row["id"] for row in reference] != [row["id"] for row in candidate]:
        raise RuntimeError("candidate changed the expected case set")
    for observed, expected in zip(candidate, reference):
        if "outputs" in expected:
            if len(expected["outputs"]) != 3 or len(observed.get("outputs", [])) != 3:
                raise RuntimeError("timed/replayed output omitted its A/B/A value challenge")
            checks = zip(observed["outputs"], expected["outputs"])
        else:
            checks = [(observed["output"], expected["output"])]
        if not all(contract.compare_output(left, right, tol, torch) for left, right in checks):
            raise RuntimeError(f"independent generated reference mismatch: {expected['id']}")


def run_pair(runner, profile, seed, expected, meta, timeout, started):
    answers = []
    for reference in (True, False):
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("generated task command budget exhausted")
        arguments = ["--ut", str(runner.UT_DIR), "--profile", profile, "--seed", str(seed)]
        if reference:
            arguments.append("--reference")
        # No golden path or output is passed to either process. Expected values
        # remain only in this parent after the reference worker completes.
        proc = runner.run_worker(runner.TASK_DIR / "scripts/generated_worker.py", arguments,
                                 None, remaining, not reference, cwd=runner.TASK_DIR)
        answers.append(parse_worker(proc, profile, seed, reference, expected, meta.get("median_launches") == 21))
    return answers


def run_correctness(runner, cfg, timeout):
    started = time.monotonic()
    report = {"input_policy": "generated_numeric_values_at_captured_structure", "profiles": []}
    try:
        import torch
        contract = load("generated_contract", runner.UT_DIR / "generated_contract.py")
        compact = contract.load_contract(runner.UT_DIR)
        meta = json.loads((runner.UT_DIR / "meta.json").read_text())
        expected = expected_profiles(meta, compact)
        seed = secrets.randbits(62)
        report["seed"] = seed
        if meta.get("median_launches") == 21:
            report.update(correctness_policy="elementwise_median_21",
                          single_launch_correctness_established=False,
                          qualification_issue="upstream individual-launch errors remain unresolved")
        profiles = (["single_launch"] if meta.get("median_launches") == 21 else []) + ["recorded", "random", "sequence", "replay"]
        for profile in profiles:
            draws = int(meta["random_draws"]) if profile == "random" else 1
            for draw in range(draws):
                reference, candidate = run_pair(runner, profile, seed + draw, expected[profile], meta, timeout, started)
                compare_rows(reference, candidate, contract, float(meta["tol"]), torch)
                report["profiles"].append({"profile": profile, "draw": draw,
                                           "cases": len(expected[profile]), "correct": True})
                if profile == "single_launch":
                    report["single_launch_reference_gate_passed"] = True
                    report["single_launch_qualification_scope"] = "retained independent recipe; scored pre-shuffled bytes are not asserted identical"
                    report["independent_reference"] = "retained AITER torch_moe_stage1; every individual launch at original tolerance"
                del reference, candidate
        report.update(status="ok", duration_seconds=time.monotonic() - started)
        runner.write_report("correctness_report.json", report)
        return True, None
    except Exception as exc:
        report.update(status="fail", error=str(exc), duration_seconds=time.monotonic() - started)
        runner.write_report("correctness_report.json", report)
        return False, str(exc)


def run_performance(runner, cfg, timeout):
    started = time.monotonic()
    try:
        meta = json.loads((runner.UT_DIR / "meta.json").read_text())
        policy = meta.get("workload_scoring") or {}
        if not policy.get("enabled", False):
            raise RuntimeError("Workload scoring disabled: " + str(policy.get("reason", "exact scenario evidence missing")))
        import torch
        ok, error = run_correctness(runner, cfg, timeout)
        if not ok:
            raise RuntimeError(error)
        contract = load("generated_contract", runner.UT_DIR / "generated_contract.py")
        compact = contract.load_contract(runner.UT_DIR)
        meta = json.loads((runner.UT_DIR / "meta.json").read_text())
        expected = expected_profiles(meta, compact)["performance"]
        reference, candidate = run_pair(runner, "performance", secrets.randbits(62), expected, meta, timeout, started)
        compare_rows(reference, candidate, contract, float(meta["tol"]), torch)
        rows = [row["timing"] for row in candidate]
        report = {"status": "ok", "expected_case_ids": expected, "test_cases": rows,
                  "warmup_iterations": 10, "benchmark_iterations": 100,
                  "reference_policy": "fresh_independent_worker_parent_comparison"}
        if meta.get("median_launches") == 21:
            report.update(correctness_policy="elementwise_median_21", single_launch_correctness_established=False)
        runner.validate_performance_report(report, expected)
        runner.write_report("performance_report.json", report)
        print(f"Performance: measured {len(rows)} complete generated-input cases")
        return rows
    except Exception as exc:
        runner.write_report("performance_report.json", {"status": "fail", "error": str(exc), "test_cases": []})
        print(f"Performance: FAILED: {exc}")
        return []
