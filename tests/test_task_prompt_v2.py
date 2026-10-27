import logging

import pytest
import yaml

from src.prompt_builder import prompt_builder
from src.task_prompt import build_task_prompt
from src.task_spec import TaskSpec


def config():
    return {"schema_version": 2, "description": "Compute every declared GEMM case.",
            "candidate": {"language": "flydsl", "initial_state": "unimplemented",
                          "editable": ["source/kernel.py"],
                          "entrypoints": [{"file": "source/kernel.py", "kind": "builder", "symbol": "build_actual"}]},
            "instructions": ["docs/operator.md"],
            "evaluation": {"runner": ["python3", "scripts/evaluate.py"]}}


def test_shared_prompt_uses_contract_without_agent_or_task_type_routing(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/operator.md").write_text("All output elements must satisfy this task's comparison.")
    (tmp_path / "README.md").write_text("Operator layout and allowed dependencies.")
    spec = TaskSpec.from_mapping(config(), task_id="custom_family/no_backend_in_name")
    prompt = build_task_prompt(spec, tmp_path, target_gpu="MI355X")
    assert "Required final implementation backend: flydsl" in prompt
    assert "Initial candidate state: unimplemented" in prompt
    assert "source/kernel.py:build_actual" in prompt
    assert "All output elements" in prompt
    assert "Operator layout and allowed dependencies" in prompt
    assert "python3 scripts/evaluate.py candidate correctness" in prompt
    assert "separate frozen baseline workspace" in prompt
    assert "Do not write task_result.yaml" in prompt
    for leaked_agent_requirement in ("forge_driver.py", "PORT", "OPTIMIZE", "Codex", "Claude"):
        assert leaked_agent_requirement not in prompt


def test_existing_public_prompt_builder_dispatches_v2_once(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/operator.md").write_text("Task-specific SQNR rule.")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config()))
    prompt = prompt_builder(str(path), tmp_path, {"target_gpu_model": "MI355X", "_task_id": "suite/test"}, logging.getLogger(__name__))
    assert "Task-specific SQNR rule" in prompt
    assert "Required final implementation backend: flydsl" in prompt
    assert "Task: suite/test" in prompt
    assert "update that file to use" not in prompt


def test_missing_declared_instructions_cannot_silently_disappear(tmp_path):
    spec = TaskSpec.from_mapping(config(), task_id="suite/test")
    with pytest.raises(ValueError):
        build_task_prompt(spec, tmp_path, target_gpu="MI355X")


def test_every_agent_gets_the_declared_workload_objective_in_shared_prompt(tmp_path):
    data = config()
    data.pop("instructions")
    data["evaluation"]["aggregation"] = {
        "method": "weighted_total_time", "case_weights": {"rare": 1, "frequent": 100},
    }
    spec = TaskSpec.from_mapping(data, task_id="suite/workload")
    prompt = build_task_prompt(spec, tmp_path, target_gpu="MI355X")
    assert "sum(case_weight * baseline_ms) / sum(case_weight * candidate_ms)" in prompt
    assert '"rare": 1' in prompt and '"frequent": 100' in prompt
    assert "Per-case speedup means are diagnostics" in prompt
    assert "ARENA_TASK_CONTEXT" in prompt


def test_serving_bundle_prompt_is_model_wide_with_optional_profiling():
    from pathlib import Path
    from src.task_spec import load_task_spec
    from src.task_prompt import build_task_prompt
    task=Path(__file__).resolve().parents[1]/'tasks/e2e/qwen3_next_80b_sglang'
    spec=load_task_spec(task/'config.yaml',task_id='e2e/qwen3_next_80b_sglang')
    prompt=build_task_prompt(spec,task,target_gpu='MI355X')
    assert 'choose any GPU backend' in prompt
    assert 'fusion across' in prompt
    assert 'Profiling is recommended but optional' in prompt
    assert 'source: implementation subtree' in prompt
    assert 'only symbols' not in prompt
