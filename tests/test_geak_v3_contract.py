"""GEAK on schema-v3 tasks: target languages, author/optimize mode and the task layout (CPU only)."""
import json
from pathlib import Path

import pytest

from agents.geak.bridge import TaskContext
from agents.geak.compatibility import author_mode, task_layout
from src.task_spec import TaskSpec, load_task_spec

ROOT = Path(__file__).resolve().parents[1]
RMS_NORM = ROOT / "tasks/normalization/rms_norm/bf16_h8192/task.yaml"


def v3_spec(language: str, baseline: str = "triton") -> TaskSpec:
    entry = "kernel.hip" if language == "hip" else "kernel.py"
    state = "implemented" if language == baseline else "unimplemented"
    candidate = {"language": language, "initial_state": state,
                 "editable": [{"path": "solution", "scope": "tree"}],
                 "entrypoints": [{"file": f"solution/{entry}", "kind": "function", "symbol": "run"}]}
    if state == "implemented":
        candidate["initial_language"] = language
    return TaskSpec.from_mapping({
        "schema_version": 2, "candidate": candidate,
        "baseline": {"kind": "provided", "language": baseline, "source_files": ["baseline/kernel.py"]},
        "evaluation": {"runner": ["python3", "-m", "_aka_eval.evaluate"]},
        "contract": {"schema_version": 3, "task_file": "task.yaml", "target_language": language,
                     "baseline_language": baseline, "operator": {"name": "rms_norm", "category": "normalization"}},
    }, task_id="normalization/rms_norm/bf16_h8192")


def write_context(tmp_path: Path, spec: TaskSpec) -> Path:
    workspace, baseline = tmp_path / "workspace", tmp_path / "baseline"
    workspace.mkdir()
    baseline.mkdir()
    path = tmp_path / "context.json"
    path.write_text(json.dumps({
        "version": 1, "task_id": spec.task_id, "task_config": spec.to_mapping(),
        "workspace": str(workspace), "baseline_workspace": str(baseline),
        "manifest": {"protocol": "arena-eval-v1", "role": "task", "action": "validate-task", "status": "PASS",
                     "metadata": {"candidate_state": spec.candidate.initial_state},
                     "cases": [{"test_case_id": "t1", "shape": [1], "dtype": "bfloat16", "status": "PASS",
                                "checks": ["correctness", "performance"]}]}}))
    return path


@pytest.mark.parametrize("language", ["triton", "gluon", "hip", "flydsl"])
def test_every_run_target_language_is_accepted(tmp_path, language):
    context = TaskContext.load(write_context(tmp_path, v3_spec(language)))
    assert context.spec.target_language == language


@pytest.mark.parametrize("language", ["tilelang", "gpu"])
def test_other_candidate_languages_fail_closed_with_a_reason(tmp_path, language):
    spec = TaskSpec.from_mapping({"schema_version": 2,
                                  "candidate": {"language": language, "editable": ["kernel.py"]},
                                  "evaluation": {"runner": ["python3", "run.py"]}}, task_id="x/y")
    with pytest.raises(ValueError, match=f"candidate is {language}"):
        TaskContext.load(write_context(tmp_path, spec))


@pytest.mark.parametrize("language,author", [("triton", False), ("gluon", True), ("hip", True), ("flydsl", True)])
def test_mode_follows_the_starting_state_from_the_baseline_language(language, author):
    spec = v3_spec(language, baseline="triton")
    assert author_mode(spec) is author
    layout = task_layout(spec)
    assert "Edit only solution/" in layout and "_aka_eval/CANDIDATE.md" in layout
    assert "baseline/ (triton)" in layout
    assert ("unimplemented template" in layout) is author
    assert "config.yaml, README.md if present" not in layout


@pytest.mark.parametrize("language", ["triton", "gluon"])
def test_a_real_v3_task_resolves_for_the_runs_target_language(language):
    # The launcher compares the declared task.yaml, resolved for the context's
    # target language, with the framework's execution record.
    spec = load_task_spec(RMS_NORM, task_id="normalization/rms_norm/bf16_h8192", target_language=language)
    assert spec.target_language == language
    assert author_mode(spec) is (language != spec.contract["baseline_language"])
