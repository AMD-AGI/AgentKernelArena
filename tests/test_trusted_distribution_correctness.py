"""Exhaustive evidence is count-complete and request-bound; GPU calls are mocked."""
import copy
import json
from pathlib import Path
import subprocess

import pytest
from src.tools import trusted_task_eval as trusted


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ("moe_stage1_grouped_gemm_silu_flydsl", 180, 6000),
    ("moe_stage1_grouped_gemm_silu_opus_a8w4", 195, 6600),
    ("moe_stage2_down_proj_reduce_opus_a8w4", 375, 18000),
)


def contract_for(suffix):
    return trusted.package_contract(ROOT / "tasks/headkernel" / ("deepseek-v4-pro__" + suffix))


def example_report(contract, phase):
    manifest, groups = contract["manifest"], contract["distributions"]
    policy = manifest["measurement"]
    request = {"phase": phase, "request_id": "cpu-example-" + phase,
               "manifest_sha256": trusted.fingerprint(manifest), "challenge_seed": 123, "gpu": {}}
    if phase != "performance":
        request["distribution_correctness_mode"] = trusted.EXHAUSTIVE_CORRECTNESS
    report = {"schema_version": 1, "status": "ok", "request": request, "compiled": True,
              "compiled_specializations": [], "cases": []}
    for case in manifest["cases"]:
        receipt = {"case_id": case["case_id"], "invoked_and_synchronized": True}
        group = groups.get(case["case_id"])
        if group:
            receipt.update(distribution_compile_setting=group["histogram"][0]["num_valid_ids"][0],
                           correctness_mode=(trusted.EXHAUSTIVE_CORRECTNESS if phase != "performance"
                                             else group["correctness"]["default_mode"]),
                           all_observed_settings_correctness=False)
        report["compiled_specializations"].append(receipt)
        if phase == "compile":
            continue
        row = {"case": case, "correct": True}
        if phase == "correctness":
            row.update(seeds=policy["correctness_seeds"],
                       negative_controls={name: True for name in policy["negative_controls"]})
            if group:
                row["correctness_variants"] = [
                    {"variant_id": setting["variant_id"], "num_valid_ids": setting["num_valid_ids"],
                     "correct": True, "seeds": list(policy["correctness_seeds"]),
                     "negative_controls": {name: True for name in policy["negative_controls"]}}
                    for setting in group["histogram"]]
                row["observed_work_distribution"] = {
                    "histogram_sha256": group["histogram_sha256"],
                    "correctness_mode": trusted.EXHAUSTIVE_CORRECTNESS,
                    "correctness_settings_checked": len(group["histogram"]),
                    "observed_setting_count": len(group["histogram"]),
                    "all_observed_settings_correctness": True,
                    "all_observed_settings_timed": False,
                    "actual_other_rank_routing_recovered": False,
                    "reference_generated_outputs_not_parent_goldens": True}
        else:
            row.update(samples_ms=[1.0] * 100, warmup_iterations=10, fresh_input_resets=100,
                       output_initializations=100, oracle_checks=100, benchmark_method="cuda_graph")
        report["cases"].append(row)
    return report, request


def validate(contract, report, request):
    trusted.validate_report(report, contract["manifest"], request)
    trusted.validate_distribution_report(report, contract["manifest"], request,
                                         contract["distributions"], contract["distribution_correctness_mode"])


@pytest.mark.parametrize("suffix,count,limit", TASKS)
def test_committed_exhaustive_policy_preserves_counts_and_measurement(suffix, count, limit):
    contract = contract_for(suffix)
    policy = contract["manifest"]["measurement"]
    assert sum(len(group["histogram"]) for group in contract["distributions"].values()) == count
    assert policy["correctness_seeds"] == [42, 43, 44]
    assert policy["negative_controls"] == ["no_op", "wrong_output"]
    assert (policy["warmup_iterations"], policy["benchmark_iterations"]) == (10, 100)
    assert trusted.phase_timeout(contract["config"], "correctness", 20000) == limit
    assert trusted.phase_timeout(contract["config"], "correctness", 1000) == 1000
    for phase in trusted.PHASES:
        report, request = example_report(contract, phase)
        validate(contract, report, request)


@pytest.mark.parametrize("attack", [
    "missing_rare", "duplicate", "wrong_count", "seed", "no_op", "wrong_output", "boolean_as_integer",
    "targeted_only", "scope_false", "exhaustive_timing", "original_routing", "parent_goldens",
    "checked_count", "observed_count", "wrong_mode", "missing_request_mode", "compile_mode",
    "compile_overclaim", "missing_compile", "duplicate_compile",
])
def test_incomplete_or_overclaimed_exhaustive_evidence_rejected(attack):
    contract = contract_for(TASKS[1][0])
    report, request = copy.deepcopy(example_report(contract, "correctness"))
    row = report["cases"][-1]
    variants = row["correctness_variants"]
    proof = row["observed_work_distribution"]
    group = contract["distributions"][row["case"]["case_id"]]
    if attack == "missing_rare":
        rare = min(group["histogram"], key=lambda setting: setting["occurrences"])["variant_id"]
        variants[:] = [setting for setting in variants if setting["variant_id"] != rare]
    elif attack == "duplicate": variants[-1] = copy.deepcopy(variants[0])
    elif attack == "wrong_count": variants[0]["num_valid_ids"][0] += 64
    elif attack == "seed": variants[-1]["seeds"].pop()
    elif attack == "no_op": del variants[-1]["negative_controls"]["no_op"]
    elif attack == "wrong_output": variants[-1]["negative_controls"]["wrong_output"] = False
    elif attack == "boolean_as_integer": variants[-1]["correct"] = 1
    elif attack == "targeted_only":
        selected = set(group["correctness"]["targeted_variant_ids"])
        variants[:] = [setting for setting in variants if setting["variant_id"] in selected]
    elif attack == "scope_false": proof["all_observed_settings_correctness"] = False
    elif attack == "exhaustive_timing": proof["all_observed_settings_timed"] = True
    elif attack == "original_routing": proof["actual_other_rank_routing_recovered"] = True
    elif attack == "parent_goldens": proof["reference_generated_outputs_not_parent_goldens"] = False
    elif attack == "checked_count": proof["correctness_settings_checked"] -= 1
    elif attack == "observed_count": proof["observed_setting_count"] -= 1
    elif attack == "wrong_mode": proof["correctness_mode"] = "targeted_uncovered_behaviors"
    elif attack == "missing_request_mode": del request["distribution_correctness_mode"]
    elif attack == "compile_mode": report["compiled_specializations"][-1]["correctness_mode"] = "targeted_uncovered_behaviors"
    elif attack == "compile_overclaim": report["compiled_specializations"][-1]["all_observed_settings_correctness"] = True
    elif attack == "missing_compile": report["compiled_specializations"].pop()
    elif attack == "duplicate_compile": report["compiled_specializations"][-1] = report["compiled_specializations"][0]
    with pytest.raises(ValueError):
        validate(contract, report, request)


@pytest.mark.parametrize("phase", trusted.PHASES)
@pytest.mark.parametrize("opted_in", [False, True])
def test_real_phase_command_forwards_only_compile_and_correctness_mode(tmp_path, monkeypatch, phase, opted_in):
    contract = contract_for(TASKS[0][0])
    report, request = example_report(contract, phase)
    if not opted_in:
        request.pop("distribution_correctness_mode", None)
    staging = tmp_path / "staging"; staging.mkdir()
    output = tmp_path / "output"; output.mkdir()
    image = "test-image"
    monkeypatch.setattr(trusted, "docker_command", lambda *args: ["docker", "run", image, "runner"])
    monkeypatch.setattr(trusted, "command_with_binding", lambda command, *args: command)
    monkeypatch.setattr(trusted, "validate_preflight", lambda *args: None)
    monkeypatch.setattr(trusted, "preserve_diagnostics", lambda *args: None)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["docker", "run"]:
            build = staging / ("reference_" + phase + "_build")
            (build / "gpu_preflight.json").write_text("{}")
            (build / (phase + "_report.json")).write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", run)
    actual = trusted.run_phase(image, tmp_path / "task", staging, output, "reference", request, "render", 100)
    assert actual == report
    expected = [phase, "--request", "/evaluation-request.json"]
    if opted_in and phase != "performance": expected += ["--distribution-correctness-mode", trusted.EXHAUSTIVE_CORRECTNESS]
    assert commands[0][-len(expected):] == expected
    assert ("--distribution-correctness-mode" in commands[0]) == (opted_in and phase != "performance")


@pytest.mark.parametrize("mode", ["targeted_uncovered_behaviors", "arbitrary", True])
def test_unrecognized_committed_mode_fails_closed(mode):
    with pytest.raises(ValueError, match="unsupported distribution"):
        trusted.distribution_phase_arguments(mode, "correctness")


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "6000"])
def test_invalid_phase_timeout_fails_closed(limit):
    with pytest.raises(ValueError, match="positive integer"):
        trusted.phase_timeout({"correctness_timeout": limit}, "correctness", 20000)


def test_registry_digest_is_checked_before_using_histogram(tmp_path):
    manifest = copy.deepcopy(contract_for(TASKS[0][0])["manifest"])
    manifest["observed_work_distributions"]["path"] = "registry.json"
    (tmp_path / "registry.json").write_text("{}")
    with pytest.raises(ValueError, match="registry digest"):
        trusted.distribution_contract(tmp_path, manifest, trusted.EXHAUSTIVE_CORRECTNESS)
