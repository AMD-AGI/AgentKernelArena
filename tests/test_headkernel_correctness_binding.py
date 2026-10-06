"""CPU behavior tests for remaining legacy bound runners and child interpreters.

Opted-in trusted_evaluation tasks use test_task_contract and their task-specific
source/oracle tests; they no longer implement this legacy overlay protocol.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml


TASKS = Path(__file__).resolve().parents[1] / "tasks/headkernel"
BOUND_TASKS = (
    "deepseek-v4-pro__dsa_sparse_mla_attn",
    "deepseek-v4-pro__moe_stage1_grouped_gemm_silu_flydsl",
    "deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4",
    "glm-5.3-flash__elementwise_copy_cluster",
    "glm-5.3-flash__fused_moe_kernel",
    "minimax-m3__decode_score_kernel",
    "minimax-m3__gqa_share_sparse_decode_kernel",
    "minimax-m3__gqa_share_sparse_fwd_kernel",
)
BOUND_TASKS = tuple(name for name in BOUND_TASKS
                    if not yaml.safe_load((TASKS/name/"config.yaml").read_text()).get("trusted_evaluation"))
pytestmark = pytest.mark.skipif(not BOUND_TASKS, reason="all scoped tasks use the guarded-case protocol")


def fixture_runner(tmp_path, monkeypatch, name, candidate="def kernel(): return 41\n"):
    original = TASKS / name
    task = tmp_path / "task"
    for directory in ("scripts", "source", "ut/kernel_src", "ut/baseline_overlay"):
        (task / directory).mkdir(parents=True, exist_ok=True)
    for relative in ("scripts/task_runner.py", "ut/harness_lib.py", "ut/overlay_setup.py"):
        (task / relative).write_bytes((original / relative).read_bytes())
    source = task / "source/candidate.py"
    source.write_text(candidate)
    (task / "ut/kernel_src/candidate.py").symlink_to("../../source/candidate.py")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "production.py").write_text("def kernel(): return 41\n")

    old_bind = json.loads((original / "ut/meta.json").read_text())["candidate_bind"]
    bind = {"kind": old_bind["kind"], "file": "kernel_src/candidate.py"}
    if bind["kind"] == "module":
        bind["module"] = "production"
    else:
        bind.update(target="production:kernel", impl_module="candidate", impl_attr="kernel")
    meta = {"candidate_bind": bind, "target_callable": "production:kernel"}
    (task / "ut/meta.json").write_text(json.dumps(meta))
    (task / "config.yaml").write_text("headkernel:\n  tol: 0\n")
    tree = ast.parse((task / "ut/overlay_setup.py").read_text())
    sitecustomize = next(ast.literal_eval(node.value) for node in tree.body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "SITECUSTOMIZE"
                                 for t in node.targets))
    base = task / "ut/baseline_overlay"
    (base / "sitecustomize.py").write_text(sitecustomize + "\nos.environ['BINDING_LEG'] = os.path.basename(_HERE)\n")
    (base / "_overlay_manifest.json").write_text(json.dumps(
        {"modules": [], "rebinds": [], "captures": [], "markers": []}))

    # The actual frozen _run_leg/baseline_random_outputs helpers run this small
    # CPU leg. No mocks replace their environment handling or process boundary.
    (task / "ut/leg_runner.py").write_text("""
import argparse, inspect, json, os
from pathlib import Path
import production
p = argparse.ArgumentParser()
p.add_argument('--task'); p.add_argument('--mode'); p.add_argument('--seed')
p.add_argument('--out'); p.add_argument('--draws'); p.add_argument('--bucket')
a = p.parse_args()
result = {'value': production.kernel(), 'overlay': os.environ.get('BINDING_LEG')}
if a.out:
    Path(a.out).write_text(json.dumps(result))
if a.mode == 'resolve':
    result = {'file': inspect.getsourcefile(production.kernel),
              'module': production.kernel.__module__, 'qualname': 'kernel'}
elif a.mode == 'list':
    result = {'sigs': ['case']}
elif a.mode == 'time':
    result = {'cases': [{'sig': 'case', 'ms': float(production.kernel()),
                         'regime': 'cpu-test', 'm': 1}],
              'overlay': os.environ.get('BINDING_LEG')}
print(json.dumps(result))
""")
    (task / "ut/unittest.py").write_text("""
import importlib.util, json, os
from pathlib import Path
from types import SimpleNamespace
import production
here = Path(__file__).resolve().parent
(here / 'UT-RAN').write_text('ran')
spec = importlib.util.spec_from_file_location('frozen_harness', here / 'harness_lib.py')
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
h._torch = lambda: SimpleNamespace(load=lambda path, **kw: json.loads(Path(path).read_text()))
assert '_cand_overlay' not in os.environ.get('PYTHONPATH', '')
baseline = h.baseline_random_outputs(str(here), {}, draws=1, timeout=10)
assert baseline == {'value': 41, 'overlay': 'baseline_overlay'}, baseline
# measure_legs uses the same explicit _run_leg override for its baseline timing.
timed = h._run_leg(str(here), str(here / 'baseline_overlay'), 'time', timeout=10)
assert timed['cases'][0]['ms'] == 41 and timed['overlay'] == 'baseline_overlay', timed
assert os.environ['BINDING_LEG'] == '_cand_overlay'
assert production.kernel() == baseline['value'], 'candidate numerical mismatch'
meta = json.loads((here / 'meta.json').read_text())
paired = h.measure_legs(str(here), meta, timeout=10, max_reps=1)
assert paired[0]['baseline_ms'] == 41 and paired[0]['optimized_ms'] == 41, paired
print('CORRECTNESS: PASS')
""")
    # Include a stale candidate PYTHONPATH entry deliberately; run_ut must remove
    # it from the environment inherited by the explicit baseline subprocesses.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(task / "ut/_cand_overlay"), str(runtime)]))
    spec = importlib.util.spec_from_file_location("bound_runner", task / "scripts/task_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner, task


@pytest.mark.parametrize("name", BOUND_TASKS)
def test_candidate_and_independent_baseline_execute_in_correct_legs(tmp_path, monkeypatch, name):
    runner, task = fixture_runner(tmp_path, monkeypatch, name)
    ok, error = runner.run_correctness({}, 20)
    assert ok, error
    report = json.loads((task / "build/correctness_report.json").read_text())
    binding = report["candidate_binding"]
    assert binding["status"] == "ok"
    assert binding["baseline_environment_restored"] is True
    assert "_cand_overlay" in binding["resolved_source"]
    assert Path(binding["candidate_source"]) == task / "source/candidate.py"
    completion = report["correctness_completion"]
    assert completion["status"] == "complete"
    assert completion["run_id"] == binding["run_id"]
    assert completion["source_sha256"] == binding["source_sha256"]


@pytest.mark.parametrize("name", BOUND_TASKS)
def test_wrong_candidate_cannot_receive_baseline_correctness_pass(tmp_path, monkeypatch, name):
    runner, task = fixture_runner(tmp_path, monkeypatch, name, "def kernel(): return -999\n")
    ok, error = runner.run_correctness({}, 20)
    assert not ok
    report = json.loads((task / "build/correctness_report.json").read_text())
    assert report["candidate_binding"]["status"] == "ok"
    assert "candidate numerical mismatch" in "\n".join(report["stdout_tail"])
    assert (task / "ut/UT-RAN").exists()


@pytest.mark.parametrize("name", BOUND_TASKS[:2])
def test_swallowed_overlay_import_error_fails_before_ut(tmp_path, monkeypatch, name):
    runner, task = fixture_runner(tmp_path, monkeypatch, name, "raise RuntimeError('broken candidate import')\n")
    ok, error = runner.run_correctness({}, 20)
    assert not ok
    report = json.loads((task / "build/correctness_report.json").read_text())
    assert report["candidate_binding"]["status"] == "fail"
    assert not (task / "ut/UT-RAN").exists()


def test_removed_binding_cannot_reuse_an_earlier_pass(tmp_path, monkeypatch):
    runner, task = fixture_runner(tmp_path, monkeypatch, BOUND_TASKS[0])
    assert runner.run_correctness({}, 20)[0]
    (task / "ut/UT-RAN").unlink()
    (task / "ut/meta.json").write_text(json.dumps({"target_callable": "production:kernel"}))
    ok, error = runner.run_correctness({}, 20)
    assert not ok and "candidate_bind" in error
    assert not (task / "ut/UT-RAN").exists()
    assert not (task / "build/correctness_binding.json").exists()
    assert json.loads((task / "build/correctness_report.json").read_text())["status"] == "fail"


def test_candidate_exit_zero_before_attestation_is_not_a_pass(tmp_path, monkeypatch):
    runner, task = fixture_runner(tmp_path, monkeypatch, BOUND_TASKS[0],
                                  "import os\nos._exit(0)\n")
    ok, error = runner.run_correctness({}, 20)
    assert not ok
    report = json.loads((task / "build/correctness_report.json").read_text())
    assert report["exit_code"] != 0
    assert report["candidate_binding"]["status"] == "fail"
    assert not (task / "ut/UT-RAN").exists()


@pytest.mark.parametrize("name", BOUND_TASKS)
@pytest.mark.parametrize("exit_body", ["import os; os._exit(0)", "raise SystemExit(0)"])
def test_candidate_exit_zero_after_binding_is_not_completion(tmp_path, monkeypatch, name, exit_body):
    runner, task = fixture_runner(tmp_path, monkeypatch, name,
                                  "def kernel():\n    " + exit_body + "\n")
    ok, error = runner.run_correctness({}, 20)
    assert not ok
    report = json.loads((task / "build/correctness_report.json").read_text())
    assert report["candidate_binding"]["status"] == "ok"
    assert report["correctness_completion"]["status"] != "complete"
    assert report["exit_code"] != 0
    assert (task / "ut/UT-RAN").exists()


def test_successful_ut_system_exit_writes_completion(tmp_path, monkeypatch):
    runner, task = fixture_runner(tmp_path, monkeypatch, BOUND_TASKS[0])
    unit = task / "ut/unittest.py"
    unit.write_text(unit.read_text() + "\nimport sys\nsys.exit(0)\n")
    ok, error = runner.run_correctness({}, 20)
    assert ok, error
    assert json.loads((task / "build/correctness_completion.json").read_text())["status"] == "complete"


def test_early_exit_cannot_reuse_previous_completion(tmp_path, monkeypatch):
    runner, task = fixture_runner(tmp_path, monkeypatch, BOUND_TASKS[0])
    assert runner.run_correctness({}, 20)[0]
    old = json.loads((task / "build/correctness_completion.json").read_text())
    (task / "source/candidate.py").write_text("def kernel():\n    import os\n    os._exit(0)\n")
    assert not runner.run_correctness({}, 20)[0]
    assert not (task / "build/correctness_completion.json").exists()
    current = json.loads((task / "build/correctness_binding.json").read_text())
    assert old["run_id"] != current["run_id"]


def test_candidate_cannot_forge_exit_origin_through_argv(tmp_path, monkeypatch):
    runner, task = fixture_runner(tmp_path, monkeypatch, BOUND_TASKS[0],
        "def kernel():\n    import sys\n    sys.argv[0] = __file__\n    raise SystemExit(0)\n")
    assert not runner.run_correctness({}, 20)[0]
    receipt = json.loads((task / "build/correctness_completion.json").read_text())
    assert receipt["status"] == "fail"


def test_source_overlay_mismatch_is_rejected(tmp_path, monkeypatch):
    runner, task = fixture_runner(tmp_path, monkeypatch, BOUND_TASKS[0])
    build = runner.candidate_overlay
    def corrupt_overlay():
        directory = build()
        (Path(directory) / "_patched/production.py").write_text("def kernel(): return 41 # detached\n")
        return directory
    monkeypatch.setattr(runner, "candidate_overlay", corrupt_overlay)
    ok, error = runner.run_correctness({}, 20)
    assert not ok and "this run's source" in error
    assert not (task / "ut/UT-RAN").exists()


def test_ut_fallback_cannot_keep_old_measurement_after_binding_failure(tmp_path, monkeypatch):
    runner, task = fixture_runner(tmp_path, monkeypatch, BOUND_TASKS[0])
    (task / "build").mkdir()
    path = task / "build/performance_report.json"
    path.write_text(json.dumps({"status": "ok", "test_cases": [{"execution_time_ms": 1}]}))
    (task / "ut/meta.json").write_text(json.dumps({"target_callable": "production:kernel"}))
    assert runner.run_performance_via_ut({}, 20, "none", "unsupported") == []
    report = json.loads(path.read_text())
    assert report["status"] == "fail" and report["test_cases"] == []
