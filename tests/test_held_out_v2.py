"""CPU subprocess fixtures exercise held-out contracts, never GPU performance."""
import logging
from pathlib import Path
import sys

import pytest
import yaml

from src.held_out.generate_heldout import build_prompt, discover_tasks, generate_for_task
from src.held_out.injection import apply_injection
from src.held_out.run_heldout_eval import evaluate_single_task
from src.task_run import _state_directory
from tests.test_task_run_v2 import package, run, cpu_runtime  # noqa: F401
from tests.test_task_session_v2 import RUNNER


LOG = logging.getLogger(__name__)
RUNNER_WITH_SHAPES = RUNNER.replace(
    '"shape": [2]', '"shape": json.loads(pathlib.Path("workload.json").read_text())["shape"]'
).replace(
    'case.update(execution_time_ms=0.1, benchmark_method="cuda_graph")',
    'case.update(execution_time_ms=0.2 if role == "baseline" else 0.1, '
    'benchmark_method="cuda_event_fallback" if "event_path" in source else "cuda_graph")'
)


def completed(tmp_path, *, empty=False, diagnostic=False, event=False, bad_heldout=False,
              mutation=False, missing_case=False):
    path = package(tmp_path, empty=empty, provided=4 if diagnostic else 3 if empty else None,
                   diagnostic=diagnostic)
    text = RUNNER_WITH_SHAPES
    if bad_heldout:
        text = text.replace('value != 3', '(value != 3 or role == "candidate" and case["shape"] == [3])')
    if mutation:
        text = text.replace('print("ARENA_EVAL_RESULT="',
                            'if role == "candidate" and action == "correctness" and case["shape"] == [3]:\n'
                            '    pathlib.Path("kernel.py").write_text("def compute(): return 30\\n")\n'
                            'print("ARENA_EVAL_RESULT="')
    if missing_case:
        text = text.replace('print("ARENA_EVAL_RESULT="',
                            'if role == "candidate" and action == "correctness" and case["shape"] == [3]:\n'
                            '    case["test_case_id"] = "different"\n'
                            'print("ARENA_EVAL_RESULT="')
    (path.parent / "evaluate.py").write_text(text)
    (path.parent / "workload.json").write_text('{"shape": [2]}\n')

    def launcher(**kw):
        (Path(kw["workspace"]) / "kernel.py").write_text(
            'def compute(): return 1 + 2\n' + ('# event_path\n' if event else ''))

    ok, workspace = run(tmp_path, path, launcher)
    assert ok
    return workspace, path.parent


def shapes(shape=3):
    return {"injections": [{"file": "workload.json", "find_marker": "raw_replace",
                            "old_code": '[2]', "replacement_code": f'[{shape}]'}]}


def evaluate(workspace, task_dir, output=None, injections=None):
    output = output or workspace.parent.parent / "heldout"
    return evaluate_single_task(workspace, output, injections or shapes(), task_dir, LOG)


def test_uses_saved_baseline_and_shared_actions_instead_of_current_task_checkout(tmp_path):
    workspace, task_dir = completed(tmp_path)
    (task_dir / "kernel.py").write_text("def compute(): return -100\n")
    result = evaluate(workspace, task_dir)
    assert result["generalization_status"] == "both_pass", result
    assert result["speedup_ratio"] == 2.0
    assert result["score"] == 320.0
    orig = tmp_path / "heldout" / "orig" / "kernel.py"
    assert "return 3" in orig.read_text()
    assert "return 1 + 2" in (workspace / "kernel.py").read_text()
    assert '[2]' in (workspace / "workload.json").read_text()
    assert len(list((tmp_path / "heldout" / "actions").glob("*.json"))) == 7


def test_generation_task_keeps_initial_stub_and_independent_provided_baseline(tmp_path):
    workspace, task_dir = completed(tmp_path, empty=True)
    result = evaluate(workspace, task_dir)
    assert result["generalization_status"] == "both_pass", result
    assert 'NotImplementedError' in (tmp_path / "heldout" / "orig" / "kernel.py").read_text()
    assert result["opt_pass_correctness"]


def test_colocated_shape_function_can_change_without_replacing_candidate_helpers(tmp_path):
    path = package(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["candidate"]["editable"] = [{"path": "kernel.py", "scope": "symbols",
                                         "symbols": ["compute"], "allow_new_helpers": True}]
    path.write_text(yaml.safe_dump(config))
    (path.parent / "kernel.py").write_text(
        "def compute(): return 3\ndef get_inputs(): return [2]\n")
    text = RUNNER_WITH_SHAPES.replace(
        'case = {', 'namespace = {}\nexec(source, namespace)\ncase = {'
    ).replace('json.loads(pathlib.Path("workload.json").read_text())["shape"]',
              'namespace["get_inputs"]()')
    (path.parent / "evaluate.py").write_text(text)

    def launcher(**kw):
        (Path(kw["workspace"]) / "kernel.py").write_text(
            "def compute(): return helper()\ndef helper(): return 3\ndef get_inputs(): return [2]\n")

    _, workspace = run(tmp_path, path, launcher)
    injection = {"injections": [{"file": "kernel.py", "find_marker": "def get_inputs",
                                  "replacement_code": "def get_inputs(): return [3]"}]}
    result = evaluate(workspace, path.parent, injections=injection)
    assert result["generalization_status"] == "both_pass", result
    assert "def helper(): return 3" in (tmp_path / "heldout" / "opt" / "kernel.py").read_text()


def test_numeric_failure_is_a_generalization_regression_not_a_clean_score(tmp_path):
    workspace, task_dir = completed(tmp_path, bad_heldout=True)
    result = evaluate(workspace, task_dir)
    assert result["generalization_status"] == "opt_regression", result
    assert result["orig_heldout_pass_correctness"]
    assert not result["opt_pass_correctness"]
    assert result["speedup_ratio"] == 0
    assert not list((tmp_path / "heldout" / "actions").glob("*-candidate-performance.json"))


def test_diagnostic_baseline_is_not_relabelled_as_correct_on_heldout(tmp_path):
    workspace, task_dir = completed(tmp_path, empty=True, diagnostic=True)
    result = evaluate(workspace, task_dir)
    assert result["generalization_status"] == "opt_improvement", result
    assert not result["orig_heldout_pass_correctness"]
    assert result["opt_pass_correctness"]
    assert result["speedup_ratio"] == 0


@pytest.mark.parametrize("fault", ["mutation", "missing_case"])
def test_invalid_heldout_evidence_is_an_error_and_never_a_shape_failure(tmp_path, fault):
    workspace, task_dir = completed(tmp_path, **{fault: True})
    result = evaluate(workspace, task_dir)
    assert result["generalization_status"] == "evaluation_error", result
    assert result["score"] == result["speedup_ratio"] == 0
    assert result["error"]


def test_event_candidate_cannot_replace_graph_baseline_policy(tmp_path):
    # Original performance is graph; a new shape exposes candidate-only Events.
    path = package(tmp_path)
    text = RUNNER_WITH_SHAPES.replace('"event_path" in source',
                                      'role == "candidate" and case["shape"] == [3]')
    (path.parent / "evaluate.py").write_text(text)
    (path.parent / "workload.json").write_text('{"shape": [2]}')
    _, candidate = run(tmp_path, path, lambda **_: None)
    result = evaluate(candidate, path.parent)
    assert result["generalization_status"] == "both_pass", result
    assert result["orig_heldout_execution_time"] == 0.2
    assert result["opt_execution_time"] == 0.1
    assert not result["benchmark_method_consistent"]
    assert result["speedup_ratio"] == 0


def test_same_inputs_with_a_new_id_are_not_heldout(tmp_path):
    workspace, task_dir = completed(tmp_path)
    injections = {"injections": [{"file": "evaluate.py", "find_marker": "raw_replace",
                                   "old_code": '"test_case_id": "one"',
                                   "replacement_code": '"test_case_id": "renamed"'}]}
    result = evaluate(workspace, task_dir, injections=injections)
    assert "reuse original inputs" in result["error"]
    assert result["score"] == 0


@pytest.mark.parametrize("target", ["kernel.py", "config.yaml"])
def test_shape_injections_cannot_replace_implementation_or_contract(tmp_path, target):
    workspace, task_dir = completed(tmp_path)
    old, new = ('return 3', 'return 0') if target == 'kernel.py' else ('language: hip', 'language: triton')
    injection = {"injections": [{"file": target, "find_marker": "raw_replace",
                                  "old_code": old, "replacement_code": new}]}
    result = evaluate(workspace, task_dir, injections=injection)
    assert result["generalization_status"] == "evaluation_error"
    assert result["score"] == 0


def test_stale_candidate_evidence_and_existing_outputs_are_preserved_and_rejected(tmp_path):
    workspace, task_dir = completed(tmp_path)
    (workspace / "kernel.py").write_text("def compute(): return 3\n# changed\n")
    result = evaluate(workspace, task_dir)
    assert "completion and source evidence" in result["error"]
    output = tmp_path / "heldout"
    before = (output / "heldout_task_result.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        evaluate(workspace, task_dir)
    assert (output / "heldout_task_result.yaml").read_bytes() == before


def test_v2_discovery_and_prompt_do_not_depend_on_task_family(tmp_path):
    path = package(tmp_path)
    assert discover_tasks(tmp_path) == [("task-package", path.parent)]
    prompt = build_prompt("anything/new-family", "held_out_shapes.yaml", {"schema_version": 2})
    assert "candidate declarations" in prompt
    assert "evaluation.workloads" in prompt
    assert "Do not change config.yaml" in prompt


@pytest.mark.parametrize("missing", [True, False])
def test_runtime_provenance_must_exist_and_match(tmp_path, monkeypatch, missing):
    workspace, task_dir = completed(tmp_path)
    if missing:
        (_state_directory(workspace) / "runtime_identity.json").unlink()
    else:
        monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {"gpu_arch": "gfx942"})
    result = evaluate(workspace, task_dir)
    assert "runtime" in result["error"]
    assert result["generalization_status"] == "evaluation_error"


def test_shape_generation_does_not_launch_an_agent_in_committed_task_sources(tmp_path, monkeypatch):
    path = package(tmp_path)
    observed = []

    def launcher(prompt, workspace, timeout, log, model=None):
        scratch = Path(workspace)
        assert scratch != path.parent
        observed.append(scratch)
        (scratch / "kernel.py").write_text("accidental edit in disposable copy")
        (scratch / "held_out_shapes.yaml").write_text(yaml.safe_dump(shapes()))
        return "fixture generated shapes"

    monkeypatch.setitem(__import__("src.held_out.generate_heldout", fromlist=["BACKENDS"]).BACKENDS,
                        "fixture", launcher)
    output = tmp_path / "private-cases"
    assert generate_for_task("family/example", path.parent, output, "fixture", 10)
    assert "def compute" in (path.parent / "kernel.py").read_text()
    assert not observed[0].exists()
    assert not generate_for_task("family/example", path.parent, output, "fixture", 10)
    assert len(observed) == 1


@pytest.mark.parametrize("kind", ["absolute", "traversal", "symlink"])
def test_injection_rejects_escape_paths_without_touching_external_files(tmp_path, kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("SHAPES = [2]\n")
    (workspace / "link.py").symlink_to(outside)
    path = str(outside) if kind == "absolute" else "../outside.py" if kind == "traversal" else "link.py"
    assert not apply_injection(workspace, {"file": path, "find_marker": "raw_replace",
                                          "old_code": "[2]", "replacement_code": "[3]"})
    assert outside.read_text() == "SHAPES = [2]\n"
