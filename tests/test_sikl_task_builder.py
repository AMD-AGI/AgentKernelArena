"""SIKL import fidelity and acceptance gates; no model or GPU needed."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from agents.sikl_task_builder.bundle import ImportProblem, inspect_bundle
from agents.sikl_task_builder.config import Config
from agents.sikl_task_builder.execution import run_process
from agents.sikl_task_builder.materialize import check_contract, materialize_task, task_digest
from agents.sikl_task_builder.orchestrator import install_task, run
from agents.sikl_task_builder import validation
from agents.task_validator.report_schema import finalize_report
from test_task_validator import _valid_raw_report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle with spaces"
    definition = {
        "name": "tiny_gemm", "op_type": "gemm",
        "axes": {"m": {"type": "var"}, "n": {"type": "const", "value": 3},
                 "k": {"type": "const", "value": 4}},
        "inputs": {"a": {"shape": ["m", "k"], "dtype": "float32"},
                   "b": {"shape": ["n", "k"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["m", "n"], "dtype": "float32"}},
        "constraints": ["m <= k"],
    }
    write_json(root / "definitions/gemm.json", definition)
    rows = [{"definition": "tiny_gemm", "workload": {
        "axes": {"m": m}, "uuid": f"case-{m}",
        "inputs": {"a": {"type": "random"}, "b": {"type": "random"}}},
        "solution": None, "evaluation": None} for m in (1, 2)]
    path = root / "workloads/gemm.jsonl"
    path.parent.mkdir()
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for role in ("baseline", "reference"):
        solution = {"name": role, "definition": "tiny_gemm", "spec": {
            "language": "python", "destination_passing_style": False,
            "entry_point": "main.py::run", "target_hardware": ["MI355X"]},
            "sources": [{"path": "main.py", "content": "from .helper import matmul\ndef run(a, b):\n    return matmul(a, b)\n"},
                        {"path": "helper.py", "content": "def matmul(a, b):\n    return a @ b.T\n"}]}
        write_json(root / f"solutions/{role}/gemm.json", solution)
    return root


@pytest.fixture
def emitted(bundle, tmp_path):
    task = inspect_bundle(bundle)[0]
    cfg = Config(str(bundle))
    draft = tmp_path / "draft"
    materialize_task(task, cfg, draft)
    return task, cfg, draft


def test_links_cases_without_executing_source(bundle):
    solution = bundle / "solutions/reference/gemm.json"
    data = json.loads(solution.read_text())
    data["sources"][0]["content"] = "raise RuntimeError('must not import')\ndef run(a, b): return a\n"
    write_json(solution, data)
    tasks = inspect_bundle(bundle)
    assert len(tasks) == 1
    assert [r["workload"]["uuid"] for r in tasks[0].rows] == ["case-1", "case-2"]
    assert tasks[0].reference == data


@pytest.mark.parametrize("problem", ["duplicate_uuid", "mixed_definition", "unknown_axis", "constraint", "input_descriptor", "shape_expression", "duplicate_key"])
def test_rejects_cases_it_cannot_preserve(bundle, problem):
    workload = bundle / "workloads/gemm.jsonl"
    rows = [json.loads(l) for l in workload.read_text().splitlines()]
    definition = json.loads((bundle / "definitions/gemm.json").read_text())
    if problem == "duplicate_uuid":
        rows[1]["workload"]["uuid"] = "case-1"
    elif problem == "mixed_definition":
        rows[1]["definition"] = "other"
    elif problem == "unknown_axis":
        rows[1]["workload"]["axes"]["extra"] = 1
    elif problem == "constraint":
        definition["constraints"] = ["m % 2 == 0"]
    elif problem == "shape_expression":
        definition["inputs"]["a"]["shape"] = ["m", "k*2"]
    elif problem == "input_descriptor":
        rows[0]["workload"]["inputs"]["a"] = {"type": "file", "path": "a.bin"}
    if problem == "duplicate_key":
        workload.write_text(workload.read_text().replace('"m": 1', '"m": 1, "m": 2'))
    else:
        workload.write_text("\n".join(json.dumps(r) for r in rows))
    write_json(bundle / "definitions/gemm.json", definition)
    with pytest.raises(ImportProblem):
        inspect_bundle(bundle)


def test_ambiguous_solution_requires_selection(bundle):
    path = bundle / "solutions/baseline/gemm.json"
    other = json.loads(path.read_text())
    other["name"] = "alternative"
    write_json(path.with_name("other.json"), other)
    with pytest.raises(ImportProblem, match="choose exactly one"):
        inspect_bundle(bundle)
    assert inspect_bundle(bundle, {"tiny_gemm": {"baseline": "baseline"}})[0].baseline["name"] == "baseline"


@pytest.mark.parametrize("source_path", ["../escape.py", "/absolute.py", "./main.py", "sub/../main.py", "a\\b.py"])
def test_rejects_path_traversal(bundle, source_path):
    path = bundle / "solutions/baseline/gemm.json"
    data = json.loads(path.read_text())
    data["sources"][0]["path"] = source_path
    write_json(path, data)
    with pytest.raises(ImportProblem):
        inspect_bundle(bundle)


def test_rejects_symlinks(bundle):
    (bundle / "linked").symlink_to(bundle / "definitions/gemm.json")
    with pytest.raises(ImportProblem, match="symbolic"):
        inspect_bundle(bundle)


def test_emission_preserves_all_source_and_contract(emitted):
    task, cfg, draft = emitted
    assert check_contract(task, cfg, draft)["ok"]
    workload = json.loads((draft / "scripts/workload.json").read_text())
    assert workload["rows"] == task.rows
    for role in ("baseline", "reference"):
        for source in getattr(task, role)["sources"]:
            assert (draft / "scripts" / role / source["path"]).read_text() == source["content"]
    assert yaml.safe_load((draft / "config.yaml").read_text())["task_type"] == "instruction2triton"
    (draft / "scripts/task_inputs.py").write_text("def make_inputs(*args): return {}\n")
    assert check_contract(task, cfg, draft)["ok"]  # semantic check belongs to GPU validator
    (draft / "scripts/workload.json").write_text("{}")
    assert not check_contract(task, cfg, draft)["ok"]


def test_emitted_task_runs_without_repository_imports(emitted):
    _, _, draft = emitted
    script = '''import json
import torch
from pathlib import Path
from scripts.task_api import load_solution, assert_outputs, validate_inputs, clone_inputs, assert_unmodified
from scripts.task_inputs import make_inputs
from source.kernel import run
root = Path.cwd()
c = json.loads((root / 'scripts/workload.json').read_text())
reference = load_solution(root / 'scripts/reference', c['reference_spec']['entry_point'])
for row in c['rows']:
    values = make_inputs(c['definition'], row, c['policy'], device='cpu')
    again = make_inputs(c['definition'], row, c['policy'], device='cpu')
    assert torch.equal(values['a'], again['a'])
    validate_inputs(values, c['definition'], row, 'cpu')
    assert_outputs(run(**values), reference(**values), c['definition'], row, c['policy'], 'cpu')
    original = clone_inputs(values)
    values['a'].add_(1)
    try: assert_unmodified(original, values)
    except ValueError: pass
    else: raise AssertionError('mutation accepted')
(root / 'source/kernel.py').write_text('raise RuntimeError("candidate failure")')
import importlib.util
spec = importlib.util.spec_from_file_location('runner', root / 'scripts/task_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
try: runner.candidate()
except RuntimeError: pass
else: raise AssertionError('candidate silently fell back to baseline')
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=draft,
                            env={**os.environ, "PYTHONPATH": ""}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("status,finalized,command_ok,changed", [
    ("PASS", True, True, False), ("WARN", True, True, False),
    ("FAIL", True, True, False), ("PASS", False, True, False),
    ("PASS", True, False, False), ("PASS", True, True, True),
])
def test_validation_requires_commands_finalized_pass_and_same_files(emitted, tmp_path, monkeypatch, status, finalized, command_ok, changed):
    _, cfg, draft = emitted
    monkeypatch.setattr(validation, "runtime_identity", lambda c: {"arch": "gfx950"})
    def process(argv, cwd, log, timeout, env):
        if "--mode" in argv:
            return {"ok": command_ok, "exit_code": 0 if command_ok else 1}
        request = json.loads(Path(argv[-1]).read_text())
        workspace = Path(request["workspace"])
        raw = _valid_raw_report("workspace")
        if status != "PASS":
            raw["checks"]["correctness"]["status"] = status
        (workspace / "validation_report.yaml").write_text(yaml.safe_dump(raw))
        if finalized:
            finalize_report(workspace, expected_task_name="workspace")
        if changed:
            (workspace / "scripts/task_inputs.py").write_text("# validator tampering\n")
        return {"ok": True, "exit_code": 0}
    monkeypatch.setattr(validation, "run_process", process)
    result = validation.validate_task(draft, tmp_path / "validation", cfg)
    assert result["ok"] == (status == "PASS" and finalized and command_ok and not changed)


def test_install_rejects_stale_evidence_and_conflicts(emitted, tmp_path):
    _, _, draft = emitted
    evidence = {"ok": True, "task_digest": task_digest(draft)}
    output = tmp_path / "tasks"
    destination = install_task(draft, output, "task", evidence)
    assert install_task(draft, output, "task", evidence) == destination
    (draft / "scripts/task_inputs.py").write_text("# repaired\n")
    with pytest.raises(ImportProblem, match="changed"):
        install_task(draft, output, "task", evidence)
    evidence["task_digest"] = task_digest(draft)
    with pytest.raises(ImportProblem, match="overwrite"):
        install_task(draft, output, "task", evidence)


def test_controller_repairs_revalidates_installs_and_resumes(bundle, tmp_path):
    repo = tmp_path / "repo"
    cfg = Config(str(bundle))
    calls = []
    def generator(config, draft, root, task, feedback, log, timeout):
        calls.append("generate")
        if len(calls) > 1:
            assert not feedback["validation"]["ok"]
            with (draft / "scripts/task_inputs.py").open("a") as f:
                f.write("\n# input repair\n")
        return {"ok": True}
    def validator(draft, artifacts, config, timeout):
        calls.append("validate")
        return {"ok": len(calls) == 4, "task_digest": task_digest(draft), "validation_id": "accepted"}
    fake_runtime = lambda cfg: {"arch": "gfx950"}
    result = run(cfg, repo=repo, generator=generator, validator=validator, runtime=fake_runtime)
    assert result["ok"] and calls == ["generate", "validate"] * 2
    assert (repo / "tasks/SIKL-task/tiny_gemm/config.yaml").is_file()
    resumed = run(cfg, result["run_id"], repo=repo, generator=generator, validator=validator, runtime=fake_runtime)
    assert resumed["ok"] and len(calls) == 4
    cfg.policy["seed"] += 1
    with pytest.raises(ValueError, match="Resume refused"):
        run(cfg, result["run_id"], repo=repo, runtime=fake_runtime)


def test_failed_generation_cannot_install_even_with_valid_contract(bundle, tmp_path):
    def validator(*args, **kwargs):
        pytest.fail("Failed generation must not reach acceptance")
    result = run(Config(str(bundle), max_repair_attempts=1), repo=tmp_path / "repo",
                 generator=lambda *a: {"ok": False}, validator=validator, runtime=lambda c: {})
    assert not result["ok"]
    assert result["tasks"]["tiny_gemm"]["attempts"] == 2
    assert not (tmp_path / "repo/tasks/SIKL-task/tiny_gemm").exists()


def test_resume_checks_snapshot_and_keeps_deadline(bundle, tmp_path):
    cfg = Config(str(bundle))
    repo = tmp_path / "repo"
    def crash(*args):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run(cfg, repo=repo, generator=crash, runtime=lambda c: {})
    root = next((repo / cfg.artifact_root).iterdir())
    state = json.loads((root / "state.json").read_text())
    state["tasks"]["tiny_gemm"]["deadline_at"] = time.time() - 1
    write_json(root / "state.json", state)
    result = run(cfg, root.name, repo=repo, generator=crash, runtime=lambda c: {})
    assert result["tasks"]["tiny_gemm"]["state"] == "failed"
    (root / "bundle/workloads/gemm.jsonl").write_text("{}")
    with pytest.raises(ImportProblem, match="snapshot"):
        run(cfg, root.name, repo=repo, runtime=lambda c: {})


def test_timeout_stops_process_group(tmp_path):
    marker = tmp_path / "survived"
    child = "import time; from pathlib import Path; time.sleep(0.8); Path(%r).touch()" % str(marker)
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(10)" % child
    result = run_process([sys.executable, "-c", parent], tmp_path, tmp_path / "process.log", 0.2)
    assert not result["ok"] and result["timed_out"]
    time.sleep(0.9)
    assert not marker.exists()


@pytest.mark.parametrize("kwargs", [{"policy": None}, {"generator": []}, {"max_repair_attempts": -1},
                                    {"output_dir": ""}, {"tasks": "tiny_gemm"}])
def test_bad_config_fails_before_agent_execution(kwargs):
    with pytest.raises(ValueError):
        Config("bundle", **kwargs)


def test_definition_output_order_survives_emission(emitted, tmp_path):
    from agents.sikl_task_builder.templates.task_api import load_solution
    task, cfg, _ = emitted
    task.definition['outputs'] = {'z_output': {'shape': ['m', 'n'], 'dtype': 'float32'},
                                  'a_output': {'shape': ['n', 'm'], 'dtype': 'float32'}}
    draft = tmp_path / 'ordered'
    materialize_task(task, cfg, draft)
    contract = json.loads((draft / 'scripts/workload.json').read_text())
    assert list(contract['definition']['outputs']) == ['z_output', 'a_output']
    # Baseline/reference packages cannot accidentally reuse each other's helper.
    for role, number in [('baseline', 2), ('reference', 5)]:
        root = tmp_path / role
        root.mkdir()
        (root / '__init__.py').write_text('from .helper import value\ndef run(): return value\n')
        (root / 'helper.py').write_text(f'value = {number}\n')
        assert load_solution(root, '__init__.py::run')() == number


def test_metadata_cli_does_not_import_torch():
    result = subprocess.run([sys.executable, '-c',
        'import sys; import agents.sikl_task_builder.__main__; assert "torch" not in sys.modules'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_graph_replay_poison_rejects_stale_and_aliased_output(emitted):
    import torch
    from agents.sikl_task_builder.templates.task_api import poison_outputs, assert_outputs
    task, cfg, _ = emitted
    row = task.rows[0]
    expected = torch.ones((1, 3))
    captured = expected.clone()
    poison_outputs(captured, expected, {}, task.definition, row, 'cpu')
    with pytest.raises(ValueError, match='finite'):
        assert_outputs(captured, expected, task.definition, row, cfg.policy, 'cpu')
    captured.copy_(expected)  # exact graph writes its captured allocation
    assert_outputs(captured, expected, task.definition, row, cfg.policy, 'cpu')
    with pytest.raises(ValueError, match='aliasing'):
        poison_outputs(captured, expected, {'input': captured}, task.definition, row, 'cpu')
