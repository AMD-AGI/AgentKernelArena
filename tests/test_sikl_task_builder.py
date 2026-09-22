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
from tests.test_task_validator_v2 import context as validation_context, draft as validation_draft
from agents.task_validator.report_v2 import DRAFT_FILENAME


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
    from src.task_spec import load_task_spec
    spec = load_task_spec(draft / "config.yaml", task_id="SIKL-task/tiny_gemm")
    assert spec.candidate.language == "triton"
    assert spec.candidate.initial_language == "python"
    assert spec.baseline.kind == "provided"
    assert len(spec.actions) == 7
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


@pytest.mark.parametrize("status,finalized,command_ok,changed,stale", [
    ("PASS", True, True, False, False), ("WARN", True, True, False, False),
    ("FAIL", True, True, False, False), ("PASS", False, True, False, False),
    ("PASS", True, False, False, False), ("PASS", True, True, True, False),
    ("PASS", True, True, False, True),
])
def test_validation_requires_commands_finalized_pass_and_same_files(emitted, tmp_path, monkeypatch, status, finalized, command_ok, changed, stale):
    _, cfg, draft = emitted
    check_name = "correctness_implementation_review"
    monkeypatch.setattr(validation, "runtime_identity", lambda c: {"arch": "gfx950"})
    def process(argv, cwd, log, timeout, env):
        request = json.loads(Path(argv[-1]).read_text())
        workspace = Path(request["workspace"])
        if stale:
            request["validation_id"] = "different-validation-attempt"
        ctx = validation_context(tmp_path / "captured", state="implemented", numerical_fail=not command_ok)
        raw = validation_draft(ctx, request_id=request["validation_id"])
        ctx["task_id"] = request["task_id"]
        ctx["workspace"] = str(workspace)
        raw["task_name"] = request["task_id"]
        from agents.task_validator.trusted_evidence import snapshot_task_evidence
        raw["task_evidence_sha256"] = snapshot_task_evidence(ctx, task_id=request["task_id"]).sha256
        if status != "PASS":
            raw["checks"][check_name]["status"] = status
            raw["checks"][check_name]["evidence"] = [{"path": "scripts/workload.json", "finding": "Test diagnostic"}]
        (workspace / DRAFT_FILENAME).write_text(yaml.safe_dump(raw))
        if finalized:
            finalize_report(workspace, expected_task_name=request["task_id"], trusted_task_evidence=ctx,
                            validation_request_id=request["validation_id"], task_schema_version=2)
        if changed:
            (workspace / "scripts/task_inputs.py").write_text("# validator tampering\n")
        return {"ok": True, "exit_code": 0}
    monkeypatch.setattr(validation, "run_process", process)
    result = validation.validate_task(draft, tmp_path / "validation", cfg)
    assert result["ok"] == (status == "PASS" and finalized and command_ok and not changed and not stale)
    if status != "PASS" and finalized and command_ok:
        assert any(d["check"] == check_name and d["status"] == status for d in result["diagnostics"])


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


def test_validation_copies_only_files_that_will_be_installed(emitted, tmp_path):
    _, _, draft = emitted
    (draft / "source/__pycache__").mkdir()
    (draft / "source/__pycache__/kernel.pyc").write_bytes(b"stale bytecode")
    (draft / "build").mkdir()
    (draft / "build/undeclared.so").write_bytes(b"old build artifact")
    workspace = tmp_path / "clean-validation"
    validation.copy_task_files(draft, workspace)
    installed = install_task(draft, tmp_path / "output", "task", {"ok": True, "task_digest": task_digest(draft)})
    assert task_digest(workspace) == task_digest(installed)
    assert not (workspace / "source/__pycache__").exists()
    assert not (workspace / "build").exists()


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


@pytest.fixture
def bundle_v2(bundle):
    definition = json.loads((bundle / "definitions/gemm.json").read_text())
    definition.update(
        schema_version=2,
        reference="def run(a, b):\n    return (a.double() @ b.double().T).to(a.dtype)\n",
        initialize=("import torch\nfrom dataclasses import dataclass\n"
                    "@dataclass\nclass Settings:\n    scale: float = 0.5\n"
                    "def run(inputs, seed=0):\n"
                    "    rng = torch.Generator(device=inputs['a'].device).manual_seed(seed)\n"
                    "    for value in inputs.values():\n"
                    "        value.normal_(std=Settings().scale, generator=rng)\n"
                    "    return inputs\n"),
        compare=("import torch\ndef run(actual, expected):\n"
                 "    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)\n"),
    )
    write_json(bundle / "definitions/gemm.json", definition)
    (bundle / "solutions/reference/gemm.json").unlink()
    baseline = json.loads((bundle / "solutions/baseline/gemm.json").read_text())
    baseline["spec"].pop("target_hardware")
    baseline["spec"]["target"] = [{"arch": "gfx950", "hardware_id": "MI355X"}]
    write_json(bundle / "solutions/baseline/gemm.json", baseline)
    return bundle


def test_v2_callbacks_are_exact_and_not_authoring_editable(bundle_v2, tmp_path):
    task = inspect_bundle(bundle_v2)[0]
    cfg = Config(str(bundle_v2), target_language="flydsl")
    draft = tmp_path / "draft"
    info = materialize_task(task, cfg, draft)
    assert info["editable"] == []
    for name in ("reference", "initialize", "compare"):
        assert (draft / f"scripts/{name}/main.py").read_text() == task.definition[name]
    assert task.origins["reference"] == "definitions/gemm.json#reference"
    from src.task_spec import load_task_spec
    spec = load_task_spec(draft / "config.yaml", task_id="SIKL-task/tiny_gemm")
    assert spec.candidate.language == "flydsl"
    assert check_contract(task, cfg, draft)["ok"]
    (draft / "scripts/task_inputs.py").write_text("# replace distribution\n")
    assert not check_contract(task, cfg, draft)["ok"]


@pytest.mark.parametrize("version", [0, 3, True, "2", None])
def test_unknown_schema_rejected(bundle_v2, version):
    path = bundle_v2 / "definitions/gemm.json"
    definition = json.loads(path.read_text())
    definition["schema_version"] = version
    write_json(path, definition)
    with pytest.raises(ImportProblem, match="schema"):
        inspect_bundle(bundle_v2)


@pytest.mark.parametrize("name", ["reference", "initialize", "compare"])
def test_callbacks_need_syntax_and_run_without_importing(bundle_v2, name):
    path = bundle_v2 / "definitions/gemm.json"
    data = json.loads(path.read_text())
    data[name] = "raise RuntimeError('metadata must not execute')\ndef run(*args, **kw): pass\n"
    write_json(path, data)
    assert inspect_bundle(bundle_v2)[0].definition[name] == data[name]
    data[name] = "def no_run(): pass\n"
    write_json(path, data)
    with pytest.raises(ImportProblem, match="run"):
        inspect_bundle(bundle_v2)


def test_v2_embedded_reference_cannot_be_overridden(bundle_v2):
    with pytest.raises(ImportProblem, match="owned by the definition"):
        inspect_bundle(bundle_v2, {"tiny_gemm": {"reference": "other"}})


def test_target_arch_is_authoritative(bundle_v2, tmp_path):
    path = bundle_v2 / "solutions/baseline/gemm.json"
    data = json.loads(path.read_text())
    data["spec"]["target"][0]["arch"] = "gfx942"
    write_json(path, data)
    task = inspect_bundle(bundle_v2)[0]
    with pytest.raises(ImportProblem, match="does not match"):
        materialize_task(task, Config(str(bundle_v2)), tmp_path / "draft")


def test_v2_callbacks_execute_and_refill_preserves_storage(bundle_v2, tmp_path):
    task = inspect_bundle(bundle_v2)[0]
    draft = tmp_path / "draft"
    cfg = Config(str(bundle_v2))
    cfg.policy.update(rtol=100, atol=100)  # Must not override the bundle's comparison.
    materialize_task(task, cfg, draft)
    script = '''import json
from pathlib import Path
import torch
from scripts.task_api import assert_outputs, load_solution
from scripts.task_inputs import make_inputs, refill_inputs
from source.kernel import run
c = json.loads(Path('scripts/workload.json').read_text())
d, row, policy = c['definition'], c['rows'][0], c['policy']
x = make_inputs(d, row, policy, device='cpu')
y = make_inputs(d, row, policy, device='cpu')
assert all(torch.equal(x[k], y[k]) for k in x)
expected = load_solution(Path('scripts/reference'), 'main.py::run')(**x)
assert_outputs(run(**x), expected, d, row, policy, 'cpu')
try: assert_outputs(expected + 1, expected, d, row, policy, 'cpu')
except AssertionError: pass
else: raise AssertionError('bundle comparison was bypassed')
pointers = {k: v.data_ptr() for k, v in x.items()}
refill_inputs(x, d, row, policy, device='cpu')
assert all(x[k].data_ptr() == pointers[k] for k in x)
assert any(not torch.equal(x[k], y[k]) for k in x)
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=draft,
                            env={**os.environ, "PYTHONPATH": ""}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("malformed", ["missing_result", "missing_case", "wrong_role"])
def test_diagnostic_tool_rejects_false_success(emitted, tmp_path, monkeypatch, malformed):
    task, cfg, draft = emitted
    monkeypatch.setattr(validation, "runtime_identity", lambda c: {})
    def process(argv, cwd, log, timeout, env):
        from agents.sikl_task_builder.bundle import case_manifest
        cases = case_manifest(task.definition, task.rows)
        if malformed == "missing_case":
            cases.pop()
        obj = {"protocol": "arena-eval-v1", "role": "baseline" if malformed == "wrong_role" else "candidate",
               "action": "correctness", "status": "PASS", "cases": cases}
        log.write_text("PASS\n" if malformed == "missing_result" else "ARENA_EVAL_RESULT=" + json.dumps(obj))
        return {"ok": True, "exit_code": 0, "timed_out": False, "log": str(log)}
    monkeypatch.setattr(validation, "run_process", process)
    assert not validation.check_task(draft, tmp_path / "check", cfg, "correctness")["ok"]


def test_refilled_replay_rejects_precomputed_result_and_vacuous_compare(bundle_v2, tmp_path):
    task = inspect_bundle(bundle_v2)[0]
    draft = tmp_path / "draft"
    materialize_task(task, Config(str(bundle_v2)), draft)
    script = '''import json, sys, types
from pathlib import Path
from scripts import task_runner as runner
c = json.loads(Path('scripts/workload.json').read_text())
d, row, policy = c['definition'], c['rows'][0], c['policy']
reference = runner.load_solution(Path('scripts/reference'), 'main.py::run')
x = runner.make_inputs(d, row, policy, device='cpu')
assert runner.validate_case(d, row, policy, reference, x, device='cpu')['wrong_output_rejected']
for stale in [False, True]:
    def benchmark(fn, *, timed_run, **kwargs):
        cached = fn().clone()
        output = cached.clone()
        def replay():
            output.copy_(cached if stale else fn())
            return output
        timed_run._bind(replay, output)
        return 0.01, {'benchmark_method': 'cuda_graph'}  # CPU replay fixture, not device evidence.
    sys.modules['_aka_benchmark'] = types.SimpleNamespace(benchmark_cuda_graph_or_events=benchmark)
    x = runner.make_inputs(d, row, policy, device='cpu')
    try: runner.measure_case(reference, reference, x, d, row, policy, device='cpu')
    except AssertionError:
        assert stale
    else:
        assert not stale, 'cached result passed after input refill'
compare = runner.load_solution(Path('scripts/compare'), 'main.py::run')
sys.modules[compare.__module__].run = lambda *args: None
x = runner.make_inputs(d, row, policy, device='cpu')
try: runner.validate_case(d, row, policy, reference, x, device='cpu')
except ValueError as error: assert 'incorrect finite' in str(error)
else: raise AssertionError('vacuous comparison accepted')
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=draft,
                            env={**os.environ, "PYTHONPATH": ""}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_baseline_action_does_not_compile_or_load_candidate(emitted):
    _, _, draft = emitted
    (draft / 'source/kernel.py').write_text('broken python !!!\n')
    script = '''from scripts import task_runner as runner
# CPU lifecycle fixture; only the device boundary is replaced.
runner.torch.cuda.is_available = lambda: True
runner.torch.version.hip = 'CPU fixture'
runner.torch.cuda.synchronize = lambda: None
make_inputs, validate_inputs, outputs = runner.make_inputs, runner.validate_inputs, runner.outputs
runner.make_inputs = lambda d, r, p: make_inputs(d, r, p, device='cpu')
runner.validate_inputs = lambda v, d, r, device: validate_inputs(v, d, r, 'cpu')
runner.outputs = lambda v, d, r, device: outputs(v, d, r, 'cpu')
assert runner.main(['baseline', 'compile']) == 0
assert runner.main(['candidate', 'compile']) == 1
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=draft,
                            env={**os.environ, 'PYTHONPATH': ''}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_deadline_cleans_framework_action_in_separate_session(tmp_path):
    marker = tmp_path / 'escaped-session'
    started = tmp_path / 'started'
    child = (
        'from pathlib import Path; import time; '
        f'Path({str(started)!r}).touch(); time.sleep(1.5); Path({str(marker)!r}).touch()'
    )
    parent = (
        'from src.task_execution import _run_process; from pathlib import Path; import os,sys; '
        f'_run_process((sys.executable, "-c", {child!r}), Path.cwd(), dict(os.environ), 30)'
    )
    result = run_process([sys.executable, '-c', parent], Path(__file__).resolve().parents[1],
                         tmp_path / 'deadline.log', 0.8)
    assert started.is_file(), (tmp_path / 'deadline.log').read_text()
    assert result['timed_out'] and not result['ok']
    time.sleep(1.6)
    assert not marker.exists(), 'framework action outlived the authoring deadline'


def test_report_identity_follows_output_suite():
    assert Config('bundle', output_dir='tasks/imported/subsuite').arena_task_id('gemm') == 'imported/subsuite/gemm'
    assert Config('bundle', output_dir='tasks/../tasks/imported').arena_task_id('gemm') == 'imported/gemm'
    root = Path(__file__).resolve().parents[1]
    assert Config('bundle', output_dir=str(root / 'tasks/imported')).arena_task_id('gemm') == 'imported/gemm'
