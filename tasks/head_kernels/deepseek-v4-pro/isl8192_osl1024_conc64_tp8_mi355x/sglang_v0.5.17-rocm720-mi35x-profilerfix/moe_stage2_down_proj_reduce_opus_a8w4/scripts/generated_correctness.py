"""Trusted parent compares separately executed frozen baseline and candidate."""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import secrets
import sys
import time

PREFIX = "DEEPSEEK_GENERATED_RESULT="


def load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != Path(path).resolve():
            raise RuntimeError(f"trusted helper alias names a different file: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_required(runner):
    helper = load("generated_contract", runner.UT_DIR / "generated_contract.py")
    data = helper.load_contract(runner.UT_DIR)
    metadata = json.loads((runner.UT_DIR / "meta.json").read_text())
    coverage = helper.profile_coverage(metadata, data)
    if coverage is not None:
        runner.write_report("sequence_coverage_report.json", coverage)
        if coverage["status"] != "complete":
            raise RuntimeError(
                "incomplete generated input contract: "
                f"{coverage['available_sequence_calls']}/"
                f"{coverage['required_sequence_calls']} mandatory sequence calls "
                f"have inputs; {coverage['missing_sequence_calls']} are missing; "
                "full correctness and performance qualification are blocked; "
                + json.dumps(coverage["missing_inputs"], sort_keys=True)
            )
    meta = metadata["generated_inputs"]
    if (
        helper.digest(runner.UT_DIR / meta["baseline_source"])
        != meta["baseline_sha256"]
    ):
        raise RuntimeError("frozen baseline source hash mismatch")


def parse_worker(proc, profile, index, seed, reference, expected):
    if proc.returncode:
        raise RuntimeError("generated worker failed: " + proc.stderr[-3000:])
    lines = [
        line[len(PREFIX) :]
        for line in proc.stdout.splitlines()
        if line.startswith(PREFIX)
    ]
    if len(lines) != 1:
        raise RuntimeError("worker did not emit exactly one fresh report")
    value = json.loads(lines[0])
    rows = value.get("rows", [])
    if (
        value.get("schema_version") != 1
        or value.get("profile") != profile
        or value.get("index") != index
        or value.get("seed") != seed
        or value.get("reference") is not reference
        or [row.get("id") for row in rows] != expected
    ):
        raise RuntimeError("worker output is stale, incomplete or reordered")
    objects = value.get("objects", {})
    for row in rows:
        if "output_ref" in row:
            if row["output_ref"] not in objects:
                raise RuntimeError("missing sequence output")
            row["output"] = objects[row["output_ref"]]
        if "output" not in row:
            raise RuntimeError("missing worker output")
    return rows


def run_correctness(runner, cfg, timeout):
    start = time.monotonic()
    seed = secrets.randbits(62)
    report = {
        "input_policy": "generated_numeric_values_at_captured_structural_contract",
        "seed": seed,
        "profiles": [],
    }
    try:
        helper = load("generated_contract", runner.UT_DIR / "generated_contract.py")
        meta = json.loads((runner.UT_DIR / "meta.json").read_text())
        verify_required(runner)
        import torch

        catalog = helper.profiles(meta)
        for profile in ("eager", "random", "sequence", "replay"):
            draws = int(meta["random_draws"]) if profile == "random" else 1
            indices = (
                range(len(catalog[profile])) if profile in ("eager", "random") else [0]
            )
            for draw in range(draws):
                for index in indices:
                    expected = (
                        [catalog[profile][index]]
                        if profile in ("eager", "random")
                        else catalog[profile]
                    )
                    answers = []
                    for reference in (True, False):
                        remaining = timeout - (time.monotonic() - start)
                        if remaining <= 0:
                            raise TimeoutError(
                                "generated correctness command budget exhausted"
                            )
                        arguments = [
                            "--ut",
                            str(runner.UT_DIR),
                            "--profile",
                            profile,
                            "--index",
                            str(index),
                            "--seed",
                            str(seed + draw),
                        ]
                        if reference:
                            arguments.append("--reference")
                        proc = runner.run_worker(
                            runner.TASK_DIR / "scripts/generated_worker.py",
                            arguments,
                            None,
                            remaining,
                            not reference,
                            cwd=runner.TASK_DIR,
                        )
                        answers.append(
                            parse_worker(
                                proc, profile, index, seed + draw, reference, expected
                            )
                        )
                    for observed, reference in zip(answers[1], answers[0]):
                        if not helper.compare_output(
                            observed["output"], reference["output"], meta["tol"], torch
                        ):
                            raise RuntimeError(
                                "generated correctness mismatch: " + observed["id"]
                            )
                    report["profiles"].append(
                        {
                            "profile": profile,
                            "index": index,
                            "draw": draw,
                            "cases": len(expected),
                            "correct": True,
                        }
                    )
                    del answers
        report.update(status="ok", duration_seconds=time.monotonic() - start)
        runner.write_report("correctness_report.json", report)
        return True, None
    except Exception as exc:
        report.update(
            status="fail", error=str(exc), duration_seconds=time.monotonic() - start
        )
        runner.write_report("correctness_report.json", report)
        return False, str(exc)
