"""CPU-only checks for the configuration-derived MiniMax dense projection task."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/aiter_workloads/minimax_m3_tp8_c64_8k1k_bf16_mlp_down"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"minimax_task_{name}", TASK / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_and_exact_case_geometry():
    from src.task_spec import load_task_spec

    load_task_spec(TASK / "config.yaml", task_id=TASK.relative_to(ROOT / "tasks").as_posix())
    data = load_script("contract").load_contract()
    assert [case["shape"] for case in data["cases"]] == [[8192, 6144, 1536], [64, 6144, 1536]]
    assert data["workload"]["ep"] is None
    assert data["operation"]["includes_all_reduce"] is False


def test_validate_task_emits_framework_parseable_case_statuses(monkeypatch):
    from src.task_protocol import CaseManifest, parse_command_result

    for name in ("contract", "inputs", "reference", "runtime"):
        monkeypatch.setitem(sys.modules, name, load_script(name))
    runner = load_script("evaluate")
    monkeypatch.setattr(runner, "verify_runtime", lambda: {})
    monkeypatch.setattr(runner, "candidate", lambda: None)
    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.tuned_gemm", types.SimpleNamespace(gemm_a16w16=lambda: None))
    report, code = runner.invoke(["validate-task"])
    result = parse_command_result("ARENA_EVAL_RESULT=" + json.dumps(report),
                                  role="task", action="validate-task", returncode=code)
    CaseManifest.from_result(result)
    assert [row["status"] for row in report["cases"]] == ["PASS", "PASS"]


@pytest.mark.parametrize("mutation", ["drop_case", "change_k", "unweighted", "changed_model"])
def test_contract_rejects_changed_scope(tmp_path, mutation):
    task = tmp_path / "task"
    shutil.copytree(TASK, task)
    if mutation == "changed_model":
        model = json.loads((task / "model_config.json").read_text())
        model["text_config"]["dense_intermediate_size"] = 24576
        (task / "model_config.json").write_text(json.dumps(model))
    elif mutation == "unweighted":
        config = yaml.safe_load((task / "config.yaml").read_text())
        del config["evaluation"]["aggregation"]
        (task / "config.yaml").write_text(yaml.safe_dump(config))
    else:
        data = json.loads((task / "workloads.json").read_text())
        if mutation == "drop_case":
            data["cases"].pop()
        else:
            data["cases"][0]["shape"][2] = 3072
        (task / "workloads.json").write_text(json.dumps(data))
    with pytest.raises((ValueError, KeyError)):
        load_script("contract").load_contract(task)


def test_small_oracle_and_incorrect_outputs():
    torch = pytest.importorskip("torch")
    module = load_script("reference")
    x = torch.tensor([[1, 2, -1], [-2, 0, 3]], dtype=torch.bfloat16)
    weight = torch.tensor([[2, -1, 4], [1, 3, -2]], dtype=torch.bfloat16)
    expected = torch.tensor([[-4, 9], [8, -8]], dtype=torch.bfloat16)
    actual = module.reference(x, weight)
    assert torch.equal(actual, expected)
    tolerance = {"rtol": 0.01, "atol": 0.01}
    module.check_output(actual, expected, tolerance)
    for wrong in (torch.zeros_like(actual), actual.T.contiguous(), actual.float(),
                  torch.full_like(actual, float("nan"))):
        with pytest.raises(AssertionError):
            module.check_output(wrong, expected, tolerance)


def test_generated_values_and_input_mutation():
    torch = pytest.importorskip("torch")
    module = load_script("inputs")
    a = module.make_inputs({"shape": [2, 5, 3]}, 42, device="cpu")
    same = module.make_inputs({"shape": [2, 5, 3]}, 42, device="cpu")
    other_m = module.make_inputs({"shape": [4, 5, 3]}, 42, device="cpu")
    changed = module.make_inputs({"shape": [2, 5, 3]}, 1042, device="cpu")
    assert all(torch.equal(x, y) for x, y in zip(a, same))
    assert torch.equal(a[1], other_m[1])
    assert all(not torch.equal(x, y) for x, y in zip(a, changed))
    snapshot = module.snapshot_inputs(a)
    module.check_inputs_unchanged(snapshot)
    a[1][0, 0] += 1
    with pytest.raises(AssertionError):
        module.check_inputs_unchanged(snapshot)


def test_candidate_imports_relative_helpers_and_dataclasses(tmp_path):
    task = tmp_path / "task"
    shutil.copytree(TASK, task)
    (task / "source" / "helper.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Offset:\n"
        "    value: int\n"
    )
    (task / "source" / "kernel.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from .helper import Offset\n"
        "@dataclass\n"
        "class Parameters:\n"
        "    offset: Offset\n"
        "def mlp_down(x, weight):\n"
        "    return x + weight + Parameters(Offset(3)).offset.value\n"
    )
    # A fresh action process also verifies that no earlier test's imported
    # source package supplies the candidate or dataclass module context.
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         "import sys; sys.path.insert(0, 'scripts'); "
         "import evaluate; f = evaluate.candidate(); "
         "assert f.__module__ == 'source.kernel'; assert f(2, 7) == 12"],
        cwd=task, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
