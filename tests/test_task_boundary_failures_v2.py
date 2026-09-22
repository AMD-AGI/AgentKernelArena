"""CPU subprocess regressions for malformed state and exporter evidence loss."""
import json
import signal
import sys
import time

import pytest
import yaml

from src.task_run import task_run_is_complete
from tests.test_task_run_v2 import package, read_report, run


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {
        "gpu_arch": "gfx950", "gpu_name": "CPU protocol fixture, no GPU", "torch": "fixture"})
    monkeypatch.setenv("AGENT_KERNEL_ARENA_PYTHON", sys.executable)


@pytest.mark.parametrize("metadata", [
    [{"candidate_state": []}],
    [{"candidate_state": {}}],
    [{"candidate_state": "implemented"}, {"candidate_state": []}],
    [{"candidate_state": {}}, {"candidate_state": "implemented"}],
    [{"candidate_state": "implemented"}, {"candidate_state": ["implemented"]}],
    [{"candidate_state": "implemented"}, {"candidate_state": None}],
    [{"candidate_state": "implemented"}, {"candidate_state": "unimplemented"}],
    [None, {"candidate_state": []}],
])
def test_malformed_state_records_initial_failure_and_final_report(tmp_path, metadata):
    path = package(tmp_path)
    config = yaml.safe_load(path.read_text())
    commands = []
    for index, item in enumerate(metadata):
        result = {"protocol": "arena-eval-v1", "role": "task", "action": "validate-task",
                  "status": "PASS", "cases": [{"test_case_id": str(index), "status": "PASS",
                  "shape": [2], "checks": ["correctness", "performance"]}]}
        if item is not None:
            result["metadata"] = item
        commands.append([sys.executable, "-c", "print(" + repr(
            "ARENA_EVAL_RESULT=" + json.dumps(result)) + ")"])
    config["evaluation"]["task"] = {"commands": commands}
    path.write_text(yaml.safe_dump(config))

    def forbidden_launcher(**_):
        pytest.fail("Malformed initial state must not start optimization")

    completed, workspace = run(tmp_path, path, forbidden_launcher)
    report = read_report(workspace)
    state = workspace.parent / ".task-sessions" / workspace.name
    initial = json.loads((state / "initial_validation.json").read_text())
    assert completed and task_run_is_complete(workspace, "suite/protocol_fixture", "codex")
    assert initial["accepted"] is False
    assert "verify candidate_state" in initial["errors"][0]
    assert report["initial_task_validation"] == initial
    assert report["agent_execution"]["status"] == "NOT_RUN"
    assert report["candidate_accepted"] is False and report["score"] == 0
    assert report["delivery_status"] == "NOT_ACCEPTED"
    actions = list(state.glob("action-*.json"))
    assert len(actions) == 1  # No baseline or candidate actions follow bad state.
    evidence = json.loads(actions[0].read_text())
    assert len(evidence["commands"]) == len(commands)
    assert evidence["result"]["metadata"]["commands"] == metadata


@pytest.mark.parametrize("artifact,mutation", [
    ("missing", None), ("unsafe", None),
    ("missing", "candidate"), ("unsafe", "candidate"),
    ("missing", "protected"), ("unsafe", "protected"),
    ("present", "candidate_escape"), ("missing", "report"),
])
def test_export_process_evidence_survives_output_errors_and_mutations(tmp_path, artifact, mutation):
    outside = tmp_path / "outside.py"
    original_outside = "raise AssertionError('external source must not execute')\n"
    outside.write_text(original_outside)
    code = """import os, pathlib, sys
print('EXPORT_STDOUT_DIAGNOSTIC')
print('EXPORT_STDERR_DIAGNOSTIC', file=sys.stderr)
output = pathlib.Path(os.environ['ARENA_EXPORT_PATH'])
"""
    if artifact != "missing":
        code += "output.parent.mkdir(parents=True, exist_ok=True)\n"
        code += ("output.symlink_to(" + repr(str(outside)) + ")\n" if artifact == "unsafe"
                 else "output.write_text('{}')\n")
    if mutation == "candidate":
        code += "pathlib.Path('kernel.py').write_text('def compute(): return 999\\n')\n"
    elif mutation == "candidate_escape":
        code += "pathlib.Path('kernel.py').unlink()\n"
        code += "pathlib.Path('kernel.py').symlink_to(" + repr(str(outside)) + ")\n"
    elif mutation == "protected":
        code += "p=pathlib.Path('evaluate.py')\np.write_text(p.read_text()+'\\n# changed harness\\n')\n"
    elif mutation == "report":
        code += "pathlib.Path(os.environ['ARENA_FINAL_RESULT_PATH']).write_text('{}')\n"
    code += "sys.exit(7)\n"
    path = package(tmp_path, exporter=code)
    _, workspace = run(tmp_path, path, lambda **_: None)
    report = read_report(workspace)
    state = workspace.parent / ".task-sessions" / workspace.name
    export = report["exports"][0]
    saved = json.loads((state / "exports.json").read_text())["exports"][0]
    assert export == saved
    assert export["status"] == "FAIL" and report["delivery_status"] == "INCOMPLETE"
    assert export["command"]["argv"] == [sys.executable, "export.py"]
    assert export["command"]["returncode"] == 7
    assert export["command"]["stdout"] == "EXPORT_STDOUT_DIAGNOSTIC\n"
    assert export["command"]["stderr"] == "EXPORT_STDERR_DIAGNOSTIC\n"
    assert export["command"]["elapsed_s"] >= 0
    assert report["pass_correctness"] and report["score"] == 220
    if mutation in ("candidate", "candidate_escape"):
        assert export["candidate_unchanged"] is False
        assert report["candidate_accepted"] is False
    elif mutation == "protected":
        assert export["protected_state_unchanged"] is False
        assert report["candidate_accepted"] is False
    else:
        assert report["candidate_accepted"] is True
    assert outside.read_text() == original_outside


def test_export_preflight_failure_does_not_inherit_previous_process_evidence(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text('{}')
    code = """import os, pathlib
output = pathlib.Path(os.environ['ARENA_EXPORT_PATH'])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('{}')
""" + "pathlib.Path('artifacts/unsafe.json').symlink_to(" + repr(str(outside)) + ")\n"
    path = package(tmp_path, exporter=code)
    config = yaml.safe_load(path.read_text())
    config["exports"].append({"format": "fixture", "output": "artifacts/unsafe.json",
                              "command": [sys.executable, "-c", "raise AssertionError('must not run')"]})
    path.write_text(yaml.safe_dump(config))
    _, workspace = run(tmp_path, path, lambda **_: None)
    first, second = read_report(workspace)["exports"]
    assert first["status"] == "PASS" and first["command"]["returncode"] == 0
    assert second["status"] == "FAIL" and "command" not in second
    assert outside.read_text() == '{}'


@pytest.mark.parametrize("partial_artifact", [False, True])
def test_export_timeout_retains_partial_process_evidence_in_final_report(tmp_path, partial_artifact):
    code = """import os, pathlib, sys, time
print('EXPORT_TIMEOUT_STDOUT', flush=True)
print('EXPORT_TIMEOUT_STDERR', file=sys.stderr, flush=True)
"""
    if partial_artifact:
        code += """output = pathlib.Path(os.environ['ARENA_EXPORT_PATH'])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('{"partial": true}')
"""
    code += "time.sleep(10)\n"
    path = package(tmp_path, exporter=code)
    config = yaml.safe_load(path.read_text())
    config["exports"][0]["timeout_s"] = 1
    path.write_text(yaml.safe_dump(config))
    started = time.monotonic()
    completed, workspace = run(tmp_path, path, lambda **_: None)
    elapsed = time.monotonic() - started
    report = read_report(workspace)
    state = workspace.parent / ".task-sessions" / workspace.name
    export = report["exports"][0]
    assert export == json.loads((state / "exports.json").read_text())["exports"][0]
    assert completed and task_run_is_complete(workspace, "suite/protocol_fixture", "codex")
    assert export["status"] == "FAIL" and export["timed_out"] is True
    assert "TimeoutExpired" in export["error"]
    assert export["command"]["argv"] == [sys.executable, "export.py"]
    assert export["command"]["returncode"] == -signal.SIGKILL
    assert export["command"]["stdout"] == "EXPORT_TIMEOUT_STDOUT\n"
    assert export["command"]["stderr"] == "EXPORT_TIMEOUT_STDERR\n"
    assert 1 <= export["command"]["elapsed_s"] <= elapsed
    assert report["delivery_status"] == "INCOMPLETE"
    assert report["candidate_accepted"] is True and report["score"] == 220
    if partial_artifact:
        assert (workspace / "artifacts/solution.json").read_text() == '{"partial": true}'
    else:
        assert not (workspace / "artifacts/solution.json").exists()


def test_timeout_evidence_and_flag_do_not_leak_between_exports(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text('{}')
    code = """import os, pathlib, sys, time
mode = sys.argv[1]
print(mode + '-stdout', flush=True)
print(mode + '-stderr', file=sys.stderr, flush=True)
output = pathlib.Path(os.environ['ARENA_EXPORT_PATH'])
output.parent.mkdir(parents=True, exist_ok=True)
if mode == 'timeout':
    pathlib.Path('artifacts/unsafe.json').symlink_to(""" + repr(str(outside)) + """)
    time.sleep(10)
output.write_text('{}')
"""
    path = package(tmp_path, exporter=code)
    config = yaml.safe_load(path.read_text())
    config["exports"] = [
        {"format": "fixture", "output": "artifacts/" + mode + ".json",
         "command": [sys.executable, "export.py", mode], "timeout_s": 1}
        for mode in ("before", "timeout", "unsafe", "after")
    ]
    path.write_text(yaml.safe_dump(config))
    _, workspace = run(tmp_path, path, lambda **_: None)
    report = read_report(workspace)
    state = workspace.parent / ".task-sessions" / workspace.name
    assert report["exports"] == json.loads((state / "exports.json").read_text())["exports"]
    before, timeout, unsafe, after = report["exports"]
    assert timeout["status"] == "FAIL" and timeout["timed_out"] is True
    assert timeout["command"]["returncode"] == -signal.SIGKILL
    assert timeout["command"]["argv"] == [sys.executable, "export.py", "timeout"]
    assert timeout["command"]["stdout"] == "timeout-stdout\n"
    assert timeout["command"]["stderr"] == "timeout-stderr\n"
    assert unsafe["status"] == "FAIL" and "command" not in unsafe and "timed_out" not in unsafe
    for name, record in (("before", before), ("after", after)):
        assert record["status"] == "PASS" and "timed_out" not in record
        assert record["command"]["returncode"] == 0
        assert record["command"]["argv"] == [sys.executable, "export.py", name]
        assert record["command"]["stdout"] == name + "-stdout\n"
        assert record["command"]["stderr"] == name + "-stderr\n"
    assert report["delivery_status"] == "INCOMPLETE" and report["candidate_accepted"] is True
    assert outside.read_text() == '{}'
