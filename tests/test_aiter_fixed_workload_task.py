"""Source/CLI checks for the unqualified AITER task; no GPU results are claimed."""
import importlib.util
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
    assert spec.aggregation is None  # No invented frequency stands in for missing evidence.
    workload = json.loads((TASK / "workloads.json").read_text())
    assert workload["cases"] == []
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
