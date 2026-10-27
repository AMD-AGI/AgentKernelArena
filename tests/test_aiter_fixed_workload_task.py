"""Source/CLI checks of the AITER task contract; no GPU results are claimed."""
import importlib.util
import hashlib
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from src.task_protocol import parse_command_result
from src.task_spec import load_task_spec


TASK_ID = "aiter_workloads/deepseek_v4_pro_tp8_ep1_c64_8k1k_fused_moe"
TASK = Path(__file__).resolve().parents[1] / "tasks" / TASK_ID


def test_task_contract_keeps_complete_call_editable_and_inputs_protected():
    spec = load_task_spec(TASK / "config.yaml", task_id=TASK_ID)
    assert spec.candidate.language == "gpu"
    assert spec.baseline.kind == "provided"
    assert spec.baseline.correctness_policy == "required"
    assert [(edit.path, edit.scope) for edit in spec.candidate.editable] == [("source", "tree")]
    source = spec.to_mapping()["workspace"]["sources"][0]
    assert source["url"] == "https://github.com/ROCm/aiter.git"
    assert source["revision"] == "2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0"
    assert source["destination"] == "upstream"
    aggregation = spec.to_mapping()["evaluation"]["aggregation"]
    assert aggregation["method"] == "weighted_total_time"
    assert aggregation["case_weights"] == {"prefill_m8192": 64, "decode_m64": 1024}
    workload = json.loads((TASK / "workloads.json").read_text())
    assert {case["test_case_id"] for case in workload["cases"]} == set(aggregation["case_weights"])
    provenance = json.loads((TASK / aggregation["source"]["path"]).read_text())
    assert provenance["status"] == "measured"
    assert {row["test_case_id"]: row["count"] for row in provenance["case_counts"]} == aggregation["case_weights"]
    assert workload["workload"]["tp"] == 8
    assert workload["workload"]["ep"] == 1
    assert workload["workload"]["concurrency"] == 64
    assert workload["workload"]["input_tokens"] == 8192
    assert workload["workload"]["output_tokens"] == 1024


@pytest.mark.parametrize("role,action", [("task", "validate-task"), *[
    (role, action) for role in ("baseline", "candidate")
    for action in ("compile", "correctness", "performance")
]])
def test_unbound_task_never_runs_or_scores_a_default_sweep(tmp_path, role, action):
    workspace = tmp_path / "task"
    shutil.copytree(TASK, workspace)
    workload_path = workspace / "workloads.json"
    workload = json.loads(workload_path.read_text())
    workload["binding_status"] = "pending_native_runtime_qualification"
    workload["runtime_binding"] = None
    workload_path.write_text(json.dumps(workload))
    # Poison dependency imports: a source-only failure must precede GPU work.
    for name in ("torch", "aiter"):
        (workspace / "scripts" / f"{name}.py").write_text(
            'raise RuntimeError("unexpected_gpu_import")\n')
    argv = [action] if role == "task" else [role, action]
    result = subprocess.run([sys.executable, "scripts/evaluate.py", *argv], cwd=workspace,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert "unexpected_gpu_import" not in result.stderr
    report = parse_command_result(result.stdout, role=role, action=action, returncode=result.returncode)
    assert not report.passed
    assert "not yet bound" in report.reason
    assert report.cases == ()


def test_setting_frozen_alone_cannot_authorize_unbound_cases(tmp_path, monkeypatch):
    workspace = tmp_path / "task"
    shutil.copytree(TASK, workspace)
    module_spec = importlib.util.spec_from_file_location("_aiter_task_evaluator", workspace / "scripts/evaluate.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.syspath_prepend(str(workspace / "scripts"))
    workload = json.loads((workspace / "workloads.json").read_text())
    workload["cases"] = []  # Explicit unbound fixture; the task now has captured cases.
    workload["binding_status"] = "frozen"
    (workspace / "workloads.json").write_text(json.dumps(workload))
    with pytest.raises(ValueError, match="nonempty"):
        module.read_binding()
    workload["cases"] = [{"test_case_id": "unbound", "checks": ["correctness", "performance"]}]
    (workspace / "workloads.json").write_text(json.dumps(workload))
    with pytest.raises(ValueError, match="benchmark_method"):
        module.read_binding()
    workload["cases"][0]["params"] = {"benchmark_method": "cuda_graph"}
    (workspace / "workloads.json").write_text(json.dumps(workload))
    with pytest.raises(ValueError, match="weights"):
        module.read_binding()


def test_pinned_source_identity_agrees_with_acquisition():
    config = yaml.safe_load((TASK / "config.yaml").read_text())
    identity = json.loads((TASK / "source_identity.json").read_text())
    assert identity["revision"] == config["workspace"]["sources"][0]["revision"]
    assert "aiter/fused_moe.py" in identity["aiter_files"]
    for relative, digest in identity["aiter_files"].items():
        assert relative.startswith("aiter/")
        assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def test_candidate_can_use_normal_package_helpers(tmp_path):
    workspace = tmp_path / "task"
    shutil.copytree(TASK, workspace)
    (workspace / "source/helpers.py").write_text("def value():\n    return 7\n")
    (workspace / "source/kernel.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from .helpers import value\n"
        "@dataclass\n"
        "class Result:\n"
        "    number: int\n"
        "def fused_moe():\n"
        "    return Result(value())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, 'scripts'); "
         "from evaluate import candidate_callable; "
         "assert candidate_callable()().number == 7"],
        cwd=workspace, text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def enrollment(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("_moe_enrollment", TASK / "scripts/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    inputs = {"path": "inputs/call.pt", "sha256": "1" * 64,
              "weights": {"path": "inputs/weights.pt", "sha256": "2" * 64},
              "current_value_replay": {"path": "inputs/replay.pt", "sha256": "3" * 64}}
    case = {"test_case_id": "decode", "shape": [64, 7168, 384, 384, 6],
            "dtype": "bfloat16", "params": {"benchmark_method": "cuda_graph"},
            "inputs": inputs, "checks": ["correctness", "performance"]}
    receipt = {"exported_cases": [{**deepcopy(case), "count": 12}]}
    def bind_receipt(value):
        body = json.dumps(value).encode()
        (tmp_path / "capture.json").write_bytes(body)
        return {"reference": "capture.json", "sha256": hashlib.sha256(body).hexdigest()}
    provenance = {"status": "measured", "enrolled_for_scoring": True,
                  "capture": bind_receipt(receipt),
                  "case_counts": [{"test_case_id": "decode", "count": 12}]}
    return module, provenance, {"decode": 12}, [case], receipt, bind_receipt


def test_measured_enrollment_joins_the_protected_case_export(enrollment):
    module, provenance, weights, cases, _, _ = enrollment
    module.check_enrollment(provenance, weights, cases)


def test_measured_enrollment_requires_a_current_value_replay_carrier(enrollment):
    module, provenance, weights, cases, receipt, bind_receipt = enrollment
    cases[0]["inputs"].pop("current_value_replay")
    receipt["exported_cases"][0]["inputs"].pop("current_value_replay")
    provenance["capture"] = bind_receipt(receipt)
    with pytest.raises(ValueError, match="current-value replay carriers"):
        module.check_enrollment(provenance, weights, cases)


@pytest.mark.parametrize("change", ["scenario", "not_enrolled", "wrong_count", "duplicate_count", "missing_carrier"])
def test_scenario_or_unmatched_counts_cannot_enroll(enrollment, change):
    module, provenance, weights, cases, _, _ = enrollment
    if change == "scenario":
        provenance["status"] = "scenario"
    elif change == "not_enrolled":
        provenance["enrolled_for_scoring"] = False
    elif change == "wrong_count":
        provenance["case_counts"][0]["count"] += 1
    elif change == "duplicate_count":
        provenance["case_counts"] *= 2
    else:
        cases[0].pop("inputs")
    with pytest.raises(ValueError):
        module.check_enrollment(provenance, weights, cases)


@pytest.mark.parametrize("change", ["payload", "replay", "shape", "count", "duplicate", "missing"])
def test_hash_valid_capture_must_match_enrolled_case(enrollment, change):
    module, provenance, weights, cases, receipt, bind_receipt = enrollment
    row = receipt["exported_cases"][0]
    if change == "payload":
        row["inputs"]["sha256"] = "3" * 64
    elif change == "replay":
        row["inputs"]["current_value_replay"]["sha256"] = "4" * 64
    elif change == "shape":
        row["shape"][0] = 32
    elif change == "count":
        row["count"] = 13
    elif change == "duplicate":
        receipt["exported_cases"] *= 2
    else:
        receipt["exported_cases"] = []
    provenance["capture"] = bind_receipt(receipt)
    with pytest.raises(ValueError, match="Capture|capture"):
        module.check_enrollment(provenance, weights, cases)


def test_missing_or_changed_capture_receipt_is_rejected(enrollment):
    module, provenance, weights, cases, _, _ = enrollment
    (module.ROOT / "capture.json").write_text("{}")
    with pytest.raises(ValueError, match="content differs"):
        module.check_enrollment(provenance, weights, cases)
    (module.ROOT / "capture.json").unlink()
    with pytest.raises(FileNotFoundError):
        module.check_enrollment(provenance, weights, cases)
