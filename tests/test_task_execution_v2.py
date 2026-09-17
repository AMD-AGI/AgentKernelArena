import json
from pathlib import Path
import sys
import time

import pytest

from src.task_execution import TaskExecutionError, run_action
from src.task_protocol import CaseManifest
from src.task_spec import TaskSpec


RUNNER = '''import json, os, sys
args = sys.argv[1:]
role, action = ("task", "validate-task") if args == ["validate-task"] else args
case = {"test_case_id": "one", "shape": [2], "dtype": "float32", "status": "PASS"}
if role == "task": case["checks"] = ["correctness", "performance"]
if action == "performance": case.update(execution_time_ms=0.1, benchmark_method="cuda_graph")
print("ARENA_EVAL_RESULT=" + json.dumps({"protocol": "arena-eval-v1", "role": role,
 "action": action, "status": "PASS", "cases": [] if action == "compile" else [case],
 "metadata": {"phase": os.environ.get("ARENA_EVAL_PHASE"), "cwd": os.getcwd()}}))
'''


def task(tmp_path, evaluation=None):
    (tmp_path / "evaluate.py").write_text(RUNNER)
    return TaskSpec.from_mapping({"schema_version": 2, "candidate": {"language": "hip", "editable": ["kernel.hip"]},
                                 "evaluation": evaluation or {"runner": [sys.executable, "evaluate.py"]}},
                                 task_id="suite/test")


def test_runs_all_seven_actions_with_phase_and_workspace_binding(tmp_path):
    spec = task(tmp_path)
    checked = run_action(spec, tmp_path, role="task", action="validate-task", phase="task_validation")
    manifest = CaseManifest.from_result(checked.result)
    for role in ("baseline", "candidate"):
        for action in ("compile", "correctness", "performance"):
            result = run_action(spec, tmp_path, role=role, action=action,
                                phase="candidate_evaluation", manifest=manifest)
            assert result.result.passed
            assert result.invocation_id != checked.invocation_id
            metadata = result.result.metadata["commands"][0]
            assert metadata["phase"] == "candidate_evaluation"
            assert Path(metadata["cwd"]) == tmp_path


def test_does_not_accept_old_result_file_or_a_success_exit_alone(tmp_path):
    spec = task(tmp_path, {"runner": [sys.executable, "-c", "print('PASS')"]})
    (tmp_path / "task_result.yaml").write_text("pass_correctness: true\n")
    with pytest.raises(TaskExecutionError, match="exactly one") as error:
        run_action(spec, tmp_path, role="candidate", action="compile", phase="candidate_evaluation")
    assert error.value.commands[0].stdout.strip() == "PASS"


def test_correctness_needs_independent_manifest_and_context_cannot_be_overridden(tmp_path):
    spec = task(tmp_path)
    with pytest.raises(TaskExecutionError, match="manifest"):
        run_action(spec, tmp_path, role="candidate", action="correctness", phase="candidate_evaluation")
    with pytest.raises(TaskExecutionError, match="cannot override"):
        run_action(spec, tmp_path, role="candidate", action="compile", phase="candidate_evaluation",
                   extra_env={"ARENA_EVAL_PHASE": "task_validation"})


def test_timeout_is_shared_across_commands(tmp_path):
    report = {"protocol": "arena-eval-v1", "role": "candidate", "action": "compile", "status": "PASS", "cases": []}
    code = "import time; time.sleep(0.65); print(" + repr("ARENA_EVAL_RESULT=" + json.dumps(report)) + ")"
    command = [sys.executable, "-c", code]
    spec = task(tmp_path, {"runner": [sys.executable, "evaluate.py"],
                          "candidate": {"compile": {"commands": [command, command], "timeout_s": 1}}})
    started = time.monotonic()
    with pytest.raises(TaskExecutionError, match="action deadline"):
        run_action(spec, tmp_path, role="candidate", action="compile", phase="candidate_evaluation")
    assert time.monotonic() - started < 2.5


def test_failure_stops_following_commands(tmp_path):
    report = {"protocol": "arena-eval-v1", "role": "candidate", "action": "compile",
              "status": "FAIL", "reason": "Build failed", "cases": []}
    fail = [sys.executable, "-c", "import sys; print(" + repr("ARENA_EVAL_RESULT=" + json.dumps(report)) + "); sys.exit(1)"]
    forbidden = [sys.executable, "-c", "open('should-not-run', 'w').write('bad')"]
    spec = task(tmp_path, {"runner": [sys.executable, "evaluate.py"],
                          "candidate": {"compile": {"commands": [fail, forbidden]}}})
    result = run_action(spec, tmp_path, role="candidate", action="compile", phase="candidate_evaluation")
    assert not result.result.passed
    assert len(result.commands) == 1
    assert not (tmp_path / "should-not-run").exists()


def test_argv_is_not_interpreted_as_shell(tmp_path):
    literal = "$(touch accidental-shell-execution)"
    code = "import sys; assert sys.argv[1] == " + repr(literal) + "; " + RUNNER.replace("args = sys.argv[1:]", "args = ['candidate', 'compile']")
    spec = task(tmp_path, {"runner": [sys.executable, "evaluate.py"],
                          "candidate": {"compile": {"commands": [[sys.executable, "-c", code, literal]]}}})
    assert run_action(spec, tmp_path, role="candidate", action="compile", phase="candidate_evaluation").result.passed
    assert not (tmp_path / "accidental-shell-execution").exists()
