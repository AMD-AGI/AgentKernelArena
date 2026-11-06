"""Schema-v3 task contracts: validation rules and per-language resolution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest
import yaml

from src.task_contract import (
    TARGET_LANGUAGES, case_seeds, hip_op_schema, load_task_contract, render_workspace, validate_task_contract,
)
from src.task_spec import TaskConfigError, TaskSpec, load_task_spec

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "tasks/normalization/rms_norm/bf16_h8192"
PILOT_ID = "normalization/rms_norm/bf16_h8192"


def _raw():
    raw = yaml.safe_load((PILOT / "task.yaml").read_text())
    workloads = json.loads((PILOT / "workloads.json").read_text())
    return raw, workloads


def _validate(raw, workloads):
    return validate_task_contract(raw, workloads, task_id=PILOT_ID)


def test_pilot_task_loads():
    contract = load_task_contract(PILOT / "task.yaml", task_id=PILOT_ID)
    assert contract.baseline_language == "triton"
    assert contract.checked_tensors == ["out"]


@pytest.mark.parametrize("language", TARGET_LANGUAGES)
def test_every_target_resolves_to_a_v2_execution_spec(language, tmp_path):
    spec = load_task_spec(PILOT / "task.yaml", task_id=PILOT_ID, target_language=language)
    assert spec.candidate.language == language
    assert spec.target_language == language
    assert spec.candidate.initial_state == ("implemented" if language == "triton" else "unimplemented")
    assert [e.path for e in spec.candidate.editable] == ["solution"]
    workspace = tmp_path / "ws"
    shutil.copytree(PILOT, workspace)
    generated = render_workspace(PILOT, workspace, task_id=PILOT_ID, target_language=language)
    assert "config.yaml" in generated and "_aka_eval/contract.json" in generated
    rendered = TaskSpec.from_mapping(yaml.safe_load((workspace / "config.yaml").read_text()), task_id=PILOT_ID)
    assert rendered.to_mapping() == spec.to_mapping()
    entry = workspace / spec.candidate.entrypoints[0].file
    assert entry.is_file()
    if language == "triton":
        assert entry.read_text() == (PILOT / "baseline/kernel.py").read_text()
    else:
        assert "AKA-UNIMPLEMENTED-TEMPLATE" in entry.read_text()


def test_task_spec_round_trips_the_contract_record():
    spec = load_task_spec(PILOT / "task.yaml", task_id=PILOT_ID, target_language="hip")
    contract = spec.contract
    assert contract["schema_version"] == 3 and contract["task_file"] == "task.yaml"
    assert contract["target_language"] == "hip" and contract["baseline_language"] == "triton"
    assert set(contract) <= {"schema_version", "task_file", "target_language", "baseline_language", "operator"}
    mapping = spec.to_mapping()
    assert mapping["contract"] == contract
    again = TaskSpec.from_mapping(mapping, task_id=PILOT_ID)
    assert again.contract == contract and again.target_language == "hip"
    assert again.to_mapping() == mapping
    spec.contract["target_language"] = "triton"   # a copy: the spec cannot be mutated through it
    assert spec.contract["target_language"] == "hip"


@pytest.mark.parametrize("update", [
    {"target_language": "triton"},          # disagrees with candidate.language
    {"schema_version": 2},
    {"task_file": "config.yaml"},
    {"unexpected": True},
])
def test_task_spec_rejects_an_inconsistent_contract_record(update):
    mapping = load_task_spec(PILOT / "task.yaml", task_id=PILOT_ID, target_language="hip").to_mapping()
    mapping["contract"].update(update)
    with pytest.raises(TaskConfigError):
        TaskSpec.from_mapping(mapping, task_id=PILOT_ID)


def test_schema_v2_spec_has_no_contract_record(tmp_path):
    mapping = load_task_spec(PILOT / "task.yaml", task_id=PILOT_ID, target_language="hip").to_mapping()
    del mapping["contract"]
    spec = TaskSpec.from_mapping(mapping, task_id=PILOT_ID)
    assert spec.contract is None and spec.target_language is None
    assert "contract" not in spec.to_mapping()


def test_target_language_is_required_and_checked():
    with pytest.raises(TaskConfigError):
        load_task_spec(PILOT / "task.yaml", task_id=PILOT_ID)
    with pytest.raises(TaskConfigError):
        load_task_spec(PILOT / "task.yaml", task_id=PILOT_ID, target_language="cuda")


def test_task_yaml_cannot_name_a_target_language():
    raw, workloads = _raw()
    raw["candidate"] = {"language": "triton"}
    with pytest.raises(TaskConfigError, match="unknown fields"):
        _validate(raw, workloads)


def test_performance_case_count_and_span_are_enforced():
    raw, workloads = _raw()
    too_few = copy.deepcopy(workloads)
    for case in too_few["cases"]:
        if case["id"] in ("t2048", "t8192"):
            case["checks"] = ["correctness"]
    with pytest.raises(TaskConfigError, match="performance cases"):
        _validate(raw, too_few)
    narrow = copy.deepcopy(workloads)
    narrow["cases"] = [c for c in narrow["cases"] if c["id"] != "t204800"]
    narrow["cases"].append({"id": "t65536", "axes": {"n_tokens": 65536}, "checks": ["correctness", "performance"]})
    with pytest.raises(TaskConfigError, match="span"):
        _validate(raw, narrow)


def test_boundary_correctness_cases_are_required():
    raw, workloads = _raw()
    for dropped in ("t1", "t1000"):
        reduced = copy.deepcopy(workloads)
        reduced["cases"] = [c for c in reduced["cases"] if c["id"] != dropped]
        with pytest.raises(TaskConfigError, match="boundary"):
            _validate(raw, reduced)


def test_axis_domain_is_enforced():
    raw, workloads = _raw()
    workloads["cases"][0]["axes"]["n_tokens"] = 300000
    with pytest.raises(TaskConfigError, match="outside its declared domain"):
        _validate(raw, workloads)


def test_baseline_relative_k_and_violation_limits():
    raw, workloads = _raw()
    raw["compare"]["out"] = {"bound": "baseline_relative", "k": 20}
    with pytest.raises(TaskConfigError, match="between 1 and 8"):
        _validate(raw, workloads)
    raw["compare"]["out"] = {"bound": "baseline_relative", "max_violation_fraction": 0.5}
    with pytest.raises(TaskConfigError, match="at most 0.01"):
        _validate(raw, workloads)


def test_integer_outputs_compare_exactly():
    raw, workloads = _raw()
    raw["interface"]["outputs"]["out"]["dtype"] = "int32"
    with pytest.raises(TaskConfigError, match="compare exactly"):
        _validate(raw, workloads)
    raw["compare"]["out"] = {"bound": "exact"}
    _validate(raw, workloads)


def test_provenance_and_baseline_entry_are_required():
    raw, workloads = _raw()
    bad = copy.deepcopy(raw)
    del bad["baseline"]["provenance"]["license"]
    with pytest.raises(TaskConfigError, match="missing fields"):
        _validate(bad, workloads)
    bad = copy.deepcopy(raw)
    bad["baseline"]["entry"] = "kernel.py"
    with pytest.raises(TaskConfigError, match="baseline/kernel.py"):
        _validate(bad, workloads)


def test_seeds_are_framework_derived_and_distinct():
    seeds = case_seeds(PILOT_ID, "t1")
    assert seeds == case_seeds(PILOT_ID, "t1")
    assert len(set(seeds)) == 2 and seeds != case_seeds(PILOT_ID, "t3")


def test_hip_schema_marks_written_tensors():
    contract = load_task_contract(PILOT / "task.yaml", task_id=PILOT_ID)
    assert hip_op_schema(contract) == "run(Tensor x, Tensor weight, float epsilon, Tensor(a!) out) -> ()"


def test_generated_and_solution_names_are_reserved(tmp_path):
    package = tmp_path / "task"
    shutil.copytree(PILOT, package)
    (package / "solution").mkdir()
    (package / "solution" / "kernel.py").write_text("")
    with pytest.raises(TaskConfigError, match="cannot contain solution"):
        load_task_spec(package / "task.yaml", task_id=PILOT_ID, target_language="triton")


def test_an_in_place_operator_may_declare_only_mutable_inputs():
    raw, workloads = _raw()
    raw["interface"]["outputs"] = {}
    raw["interface"]["mutable"] = ["x"]
    raw["compare"] = {"x": {"bound": "baseline_relative"}}
    _validate(raw, workloads)
    raw["interface"]["mutable"] = []
    raw["compare"] = {}
    with pytest.raises(TaskConfigError, match="at least one output or mutable input"):
        _validate(raw, workloads)


def test_contract_json_keeps_the_declared_order(tmp_path):
    # A reference may return a tuple in declared order and derived axes may build on earlier ones,
    # so the evaluator must see the task.yaml order, not an alphabetical one.
    workspace = tmp_path / "ws"
    shutil.copytree(PILOT, workspace)
    render_workspace(PILOT, workspace, task_id=PILOT_ID, target_language="triton")
    resolved = json.loads((workspace / "_aka_eval/contract.json").read_text())
    declared = yaml.safe_load((PILOT / "task.yaml").read_text())["interface"]
    assert list(resolved["inputs"]) == list(declared["inputs"])
    assert list(declared["inputs"]) != sorted(declared["inputs"])


def test_dot_product_bound_is_not_offered():
    raw, workloads = _raw()
    raw["compare"]["out"] = {"bound": "dot_product", "length": 8192}
    with pytest.raises(TaskConfigError):
        _validate(raw, workloads)


def test_package_digest_ignores_umask_bits_but_not_the_executable_bit(tmp_path):
    from src.task_materialization import _Deadline, _tree_manifest, package_digest

    def digest(root):
        return package_digest(_tree_manifest(root, _Deadline(60)))

    for name, file_mode, dir_mode in (("a", 0o644, 0o755), ("b", 0o664, 0o775)):
        root = tmp_path / name
        shutil.copytree(PILOT, root, ignore=shutil.ignore_patterns("__pycache__"))
        for path in [root, *root.rglob("*")]:
            path.chmod(dir_mode if path.is_dir() else file_mode)
    assert digest(tmp_path / "a") == digest(tmp_path / "b")
    (tmp_path / "b" / "reference.py").chmod(0o775)
    assert digest(tmp_path / "a") != digest(tmp_path / "b")
