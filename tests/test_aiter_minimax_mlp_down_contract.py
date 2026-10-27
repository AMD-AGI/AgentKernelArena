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


@pytest.mark.parametrize("role", ["baseline", "candidate"])
@pytest.mark.parametrize("offsets", [(0, 0), (1, 0), (0, 1), (1, 1), (7, 3)])
def test_upstream_alignment_preparation_uses_current_values(monkeypatch, role, offsets):
    torch = pytest.importorskip("torch")
    inputs = load_script("inputs")
    oracle = load_script("reference")
    values = inputs.make_inputs({"shape": [2, 5, 3]}, 42, device="cpu", offsets=offsets)
    observed = []

    def upstream(x, weight):
        assert x.data_ptr() % 16 == weight.data_ptr() % 16 == 0
        observed.append((x, weight))
        return oracle.reference(x, weight)

    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.tuned_gemm", types.SimpleNamespace(gemm_a16w16=upstream))
    if role == "baseline":
        operation = load_script("baseline").mlp_down
    else:
        spec = importlib.util.spec_from_file_location("minimax_initial", TASK / "source/kernel.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        operation = module.mlp_down
    for iteration in range(2):
        if iteration:
            for old, fresh in zip(values, inputs.make_inputs({"shape": [2, 5, 3]}, 1042, device="cpu")):
                old.copy_(fresh)
        frozen = inputs.snapshot_inputs(values)
        assert torch.equal(operation(*values), oracle.reference(*values))
        inputs.check_inputs_unchanged(frozen)
        for original, dispatched in zip(values, observed[-1]):
            assert (original.data_ptr() == dispatched.data_ptr()) == (original.data_ptr() % 16 == 0)


def test_unaligned_storage_prefix_is_read_only():
    torch = pytest.importorskip("torch")
    inputs = load_script("inputs")
    values = inputs.make_inputs({"shape": [2, 5, 3]}, 42, device="cpu", offsets=(1, 1))
    assert all(value.is_contiguous() and value.data_ptr() % 16 == 2 for value in values)
    frozen = inputs.snapshot_inputs(values)
    values[0].as_strided((1,), (1,), storage_offset=0).zero_()
    with pytest.raises(AssertionError, match="inputs must remain unchanged"):
        inputs.check_inputs_unchanged(frozen)


@pytest.mark.parametrize("stale_operand", [None, 0, 1])
def test_unaligned_graph_checks_reject_stale_preparation(monkeypatch, stale_operand):
    pytest.importorskip("torch")
    for name in ("contract", "inputs", "reference", "runtime"):
        monkeypatch.setitem(sys.modules, name, load_script(name))
    runner = load_script("evaluate")
    inputs = load_script("inputs")
    monkeypatch.setattr(runner, "make_inputs", lambda case, seed, **kwargs:
                        inputs.make_inputs(case, seed, device="cpu", **kwargs))
    captures = []

    def capture(launch, warmup):
        captures.append(warmup)
        return launch

    monkeypatch.setattr(runner, "capture_replay", capture)
    prepared = {}

    def operation(x, weight):
        values = [x, weight]
        if stale_operand is not None:
            key = id(values[stale_operand])
            prepared.setdefault(key, values[stale_operand].clone())
            values[stale_operand] = prepared[key]
        return runner.reference(*values)

    data = load_script("contract").load_contract()
    if stale_operand is None:
        result = runner.check_unaligned(operation, {"shape": [2, 5, 3]}, data)
        assert len(result) == len(captures) == 3
    else:
        with pytest.raises(AssertionError):
            runner.check_unaligned(operation, {"shape": [2, 5, 3]}, data)
