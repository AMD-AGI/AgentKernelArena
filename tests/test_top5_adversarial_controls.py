"""CPU regressions for attacks that must not produce a head-kernel score.

These tests exercise real guard/parser/scoring boundaries using small task
packages. They do not execute a GPU kernel or establish that the task-local
timers, random-value oracles, or deployment replay are correct.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.testcases import TestCaseResult as CaseResult
from src.testcases import calculate_average_speedup


IMMUTABLE_INPUTS = (
    "scripts/task_runner.py",
    "scripts/_bench.py",
    "ut/cases.py",
    "ut/harness_lib.py",
    "ut/task_contract.py",
    "ut/meta.json",
    "ut/reference_io.pt",
    "ut/baseline_ref/kernel.py.orig",
    "ut/model_config/config.json",
    "ut/runtime/model.patch",
    "session_cases.json",
)


@pytest.fixture
def task_package(tmp_path):
    task = tmp_path / "original"
    workspace = tmp_path / "workspace"
    task.mkdir()
    config = {
        "task_type": "triton2triton",
        "source_file_path": ["source/kernel.py"],
        "target_kernel_functions": ["kernel"],
        "performance_command": ["python3 scripts/task_runner.py performance"],
    }
    (task / "config.yaml").write_text(yaml.safe_dump(config))
    (task / "source").mkdir()
    (task / "source/kernel.py").write_text("def kernel(value):\n    return value + 1\n")
    for relative in IMMUTABLE_INPUTS:
        path = task / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"immutable capture or harness fixture\n")
    shutil.copytree(task, workspace)
    return task, workspace


@pytest.mark.parametrize("relative", IMMUTABLE_INPUTS)
@pytest.mark.parametrize("operation", ("replace", "delete"))
def test_shipped_task_input_tampering_rejects_score(task_package, relative, operation):
    task, workspace = task_package
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    attacked = workspace / relative
    if operation == "delete":
        attacked.unlink()
    else:
        attacked.write_bytes(b"weaker oracle, reduced case set, or fabricated measurement\n")

    with pytest.raises(RuntimeError, match="kernel score is rejected"):
        verify_workspace_harness(snapshot)


def test_implementation_change_preserves_full_task_input_guard(task_package):
    task, workspace = task_package
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    (workspace / "source/kernel.py").write_text(
        "def kernel(value):\n    return 1 + value\n"
    )

    verify_workspace_harness(snapshot)


def test_missing_capture_before_agent_execution_fails_closed(task_package):
    task, workspace = task_package
    (workspace / "ut/reference_io.pt").unlink()

    with pytest.raises(RuntimeError, match="Task inputs missing before agent execution"):
        snapshot_workspace_harness(workspace, task_root=task)


def test_capture_mutation_during_candidate_evaluation_is_detectable(task_package):
    task, workspace = task_package
    snapshot = snapshot_workspace_harness(workspace, task_root=task)
    verify_workspace_harness(snapshot)
    # Models a source module writing to a frozen file only when imported by the
    # evaluator, after the initial guard check. The post-evaluation check is
    # necessary to catch this persistent mutation.
    (workspace / "ut/reference_io.pt").write_bytes(b"candidate's desired answer")

    with pytest.raises(RuntimeError, match="kernel score is rejected"):
        verify_workspace_harness(snapshot)


def _case(case_id, milliseconds=10.0, method="cuda_graph", **params):
    return CaseResult(
        test_case_id=case_id,
        execution_time_ms=milliseconds,
        shape=[16, 128],
        metadata={
            "benchmark_method": method,
            "dtype": "torch.bfloat16",
            "params": {"case": case_id, "stride": [128, 1], **params},
        },
    )


@pytest.mark.parametrize("method", (None, "cuda_event", "wall", "mixed:cuda_graph,cuda_event_fallback"))
def test_unattested_or_ambiguous_timer_cannot_earn_speedup(method):
    baseline = [_case("decode", method=method)]
    candidate = [_case("decode", 0.01, method=method)]

    assert calculate_average_speedup(baseline, candidate) == 0.0


def test_same_aggregate_methods_cannot_hide_swapped_case_timers():
    baseline = [_case("decode", method="cuda_graph"),
                _case("prefill", method="cuda_event_fallback")]
    candidate = [_case("decode", 0.01, "cuda_event_fallback"),
                 _case("prefill", 0.01, "cuda_graph")]

    assert calculate_average_speedup(baseline, candidate) == 0.0


@pytest.mark.parametrize("attack", ("missing_case", "extra_case", "duplicate_case"))
def test_partial_or_duplicated_candidate_sweep_cannot_earn_speedup(attack):
    baseline = [_case("decode"), _case("prefill")]
    candidate = [_case("decode", 0.01), _case("prefill", 0.01)]
    if attack == "missing_case":
        candidate.pop()
    elif attack == "extra_case":
        candidate.append(_case("invented_fast_case", 0.01))
    else:
        candidate.append(deepcopy(candidate[0]))

    assert calculate_average_speedup(baseline, candidate) == 0.0


@pytest.mark.parametrize("attack", ("stride", "dtype", "shape", "remove_params"))
def test_same_case_id_cannot_hide_changed_workload(attack):
    baseline = [_case("decode")]
    candidate = [_case("decode", 0.01)]
    if attack == "stride":
        candidate[0].metadata["params"]["stride"] = [1, 16]
    elif attack == "dtype":
        candidate[0].metadata["dtype"] = "torch.float16"
    elif attack == "shape":
        candidate[0].shape = [1, 128]
    else:
        del candidate[0].metadata["params"]

    assert calculate_average_speedup(baseline, candidate) == 0.0


@pytest.mark.parametrize("milliseconds", (0.0, -1.0, float("nan"), float("inf")))
def test_invalid_sample_cannot_be_dropped_to_improve_score(milliseconds):
    baseline = [_case("decode"), _case("prefill")]
    candidate = [_case("decode", 0.01), _case("prefill", milliseconds)]

    assert calculate_average_speedup(baseline, candidate) == 0.0


def test_real_source_only_gain_still_scores_with_complete_contract():
    baseline = [_case("decode", 10.0), _case("prefill", 20.0)]
    candidate = [_case("prefill", 10.0), _case("decode", 5.0)]

    assert calculate_average_speedup(baseline, candidate) == pytest.approx(2.0)


def test_stale_published_report_is_not_reused_after_empty_success(tmp_path, monkeypatch):
    from src import performance

    report = tmp_path / "build/performance_report.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"test_cases": [
        {"test_case_id": "decode", "execution_time_ms": 0.0001,
         "benchmark_method": "cuda_graph"}
    ]}))
    monkeypatch.setattr(performance, "force_jit_rebuild", lambda *_args: {})
    monkeypatch.setattr(performance, "run_command", lambda *_args, **_kwargs: (True, "", ""))

    cases = performance.measure_performance(
        tmp_path,
        {"task_type": "triton2triton", "performance_command": ["fixture no-output"]},
    )

    assert cases == []
    assert not report.exists()


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "tasks/head_kernels/_support/task_runner.py"
    spec = importlib.util.spec_from_file_location("adversarial_head_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bench():
    path = Path(__file__).resolve().parents[1] / "tasks/head_kernels/_support/_bench.py"
    spec = importlib.util.spec_from_file_location("adversarial_head_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_report():
    return {
        "status": "ok",
        "expected_case_ids": ["decode", "prefill"],
        "test_cases": [
            {"test_case_id": name, "execution_time_ms": 1.0,
             "benchmark_method": "cuda_graph", "benchmark_samples": 100,
             "benchmark_replay_probe": "input_change_and_restore",
             "benchmark_state_restore": "all_input_storages_before_each_replay",
             "benchmark_method_consistent": True,
             "benchmark_output_validation": "exact_timed_graph_replay"}
            for name in ("decode", "prefill")
        ],
    }


def test_runner_accepts_complete_measurement_contract(runner):
    raw = _raw_report()
    assert runner.validate_performance_report(raw, raw["expected_case_ids"]) == raw["test_cases"]


@pytest.mark.parametrize("attack", (
    "partial", "duplicate", "unexpected", "nonfinite", "method", "sample_count", "unvalidated_output",
))
def test_runner_rejects_incomplete_or_weak_measurement(runner, attack):
    raw = _raw_report()
    row = raw["test_cases"][0]
    if attack == "partial":
        raw["test_cases"].pop()
    elif attack == "duplicate":
        raw["test_cases"][1]["test_case_id"] = row["test_case_id"]
    elif attack == "unexpected":
        row["test_case_id"] = "invented"
    elif attack == "nonfinite":
        row["execution_time_ms"] = float("nan")
    elif attack == "method":
        row["benchmark_method"] = "cuda_event_fallback"
    elif attack == "sample_count":
        row["benchmark_samples"] = 1
    else:
        del row["benchmark_output_validation"]

    with pytest.raises(RuntimeError):
        runner.validate_performance_report(raw)


def test_complete_manifest_keeps_every_long_signature(bench):
    rows = [{"sig": "same_prefix_" + "x" * 70 + str(index), "regime": "decode"}
            for index in range(30)]

    identities = bench.validate_cases(rows)

    assert len(identities) == 30
    assert len(set(identities)) == 30
    assert all(row["sig"] in identity for row, identity in zip(rows, identities))


def test_runner_does_not_reuse_private_stale_result(runner, tmp_path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    raw = build / "_bench_raw.json"
    raw.write_text(json.dumps(_raw_report()))
    monkeypatch.setattr(runner, "TASK_DIR", tmp_path)
    monkeypatch.setattr(runner, "BUILD_DIR", build)
    monkeypatch.setattr(runner, "UT_DIR", tmp_path / "ut")
    monkeypatch.setattr(runner, "run_correctness", lambda *_args: (True, None))
    monkeypatch.setattr(runner, "overlays", lambda: (None, None))
    monkeypatch.setattr(runner, "run_process", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=0, stdout="", stderr=""))

    assert runner.run_performance({}, 5) == []
    assert not raw.exists()
    assert json.loads((build / "performance_report.json").read_text())["status"] == "fail"


def test_runner_refuses_performance_after_correctness_failure(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "BUILD_DIR", tmp_path)
    monkeypatch.setattr(runner, "run_correctness", lambda *_args: (False, "deliberate wrong output"))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("performance worker ran after correctness failed")

    monkeypatch.setattr(runner, "run_process", forbidden)

    assert runner.run_performance({}, 5) == []
    report = json.loads((tmp_path / "performance_report.json").read_text())
    assert report["status"] == "fail"
    assert "deliberate wrong output" in report["error"]


@pytest.mark.parametrize("replay_value", (-999, 123))
def test_timed_replay_must_compute_correct_output_and_respond_to_new_values(
    bench, monkeypatch, replay_value,
):
    # The normal callable is correct, but the exact graph retained by the timer
    # is either wrong or memoizes its first output. Both must be rejected.
    state = SimpleNamespace(probe_enabled=False, restore=lambda: None)
    monkeypatch.setattr(bench, "InputState", lambda *_args: state)
    monkeypatch.setattr(bench, "replay_probe", lambda *_args: None)
    oracle = SimpleNamespace(
        to_device_like=lambda value, _device: value,
        correct=lambda actual, expected, _tol: (actual == expected, abs(actual - expected)),
    )

    def fake_benchmark(_call, **kwargs):
        kwargs["timed_run"]._bind(lambda: replay_value, None)
        return [1.0] * 3, {"benchmark_method": "cuda_graph"}

    with pytest.raises(RuntimeError, match="timed graph output mismatch"):
        bench.measure_case(
            {"sig": "decode", "args": {}}, lambda _args: 123,
            {"base": 123, "probe": 456}, SimpleNamespace(), oracle,
            {"tol": 0.01}, SimpleNamespace(), 1, 3, fake_benchmark,
        )


@pytest.mark.parametrize("candidate", (
    "def kernel(rhs, lhs): return lhs + rhs\n",
    "def kernel(lhs, rhs=2): return lhs + rhs\n",
    "def kernel(lhs, *, rhs): return lhs + rhs\n",
))
def test_production_argument_contract_cannot_be_rewritten(runner, tmp_path, monkeypatch, candidate):
    monkeypatch.setattr(runner, "TASK_DIR", tmp_path)
    monkeypatch.setattr(runner, "BUILD_DIR", tmp_path / "build")
    cfg = {"source_file_path": ["kernel.py"], "target_kernel_functions": ["kernel"]}
    source = tmp_path / "kernel.py"
    source.write_text("def kernel(lhs, rhs): return lhs + rhs\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/source_abi.json").write_text(json.dumps(runner.source_abi(cfg)))
    source.write_text(candidate)

    ok, error = runner.run_compile(cfg)

    assert ok is False
    assert "ABI changed" in error


@pytest.fixture
def colocated_harness(tmp_path):
    source = (
        "import triton\n"
        "import torch\n"
        "from clock_library import elapsed\n"
        "from comparison_library import assert_close\n"
        "import math as tuning_math\n"
        "WARMUP = 10\n"
        "class BenchmarkReport:\n    pass\n"
        "def make_input():\n    return torch.empty(16)\n"
        "@triton.jit\n"
        "def kernel(x):\n    return x\n"
        "def test_performance():\n"
        "    local_only = make_input()\n"
        "    assert_close(kernel(local_only), local_only)\n"
        "    return elapsed(kernel), len(local_only), WARMUP, BenchmarkReport\n"
    )
    entrypoint = tmp_path / "combined.py"
    entrypoint.write_text(source)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "source_file_path": ["combined.py"],
        "target_kernel_functions": ["kernel"],
        "performance_command": ["python3 combined.py"],
    }))
    return entrypoint, source, snapshot_workspace_harness(tmp_path)


@pytest.mark.parametrize("injected", (
    "from payload import test_performance\n",
    "import payload as test_performance\n",
    "from payload import fake_timer as elapsed\n",
    "import payload.nested as torch\n",
    "from payload import fake_compare as assert_close\n",
    "from payload import WARMUP\n",
    "from payload import BenchmarkReport\n",
    "from payload import make_input\n",
    "import payload as len\n",
    "from payload import *\n",
))
def test_colocated_import_cannot_rebind_original_harness(colocated_harness, injected):
    entrypoint, source, snapshot = colocated_harness
    entrypoint.write_text(source + injected)

    with pytest.raises(RuntimeError, match="kernel score is rejected"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("replacement", (
    "from payload import fake_timer as elapsed\n",
    "from payload import elapsed\n",
    "import payload as elapsed\n",
))
def test_colocated_import_source_cannot_be_changed(colocated_harness, replacement):
    entrypoint, source, snapshot = colocated_harness
    entrypoint.write_text(source.replace("from clock_library import elapsed\n", replacement))

    with pytest.raises(RuntimeError, match="kernel score is rejected"):
        verify_workspace_harness(snapshot)


def test_future_import_cannot_change_protected_harness_semantics(colocated_harness):
    entrypoint, source, snapshot = colocated_harness
    entrypoint.write_text("from __future__ import annotations\n" + source)

    with pytest.raises(RuntimeError, match="kernel score is rejected"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("body", (
    "    global elapsed\n    from payload import elapsed\n    return x\n",
    "    def nested():\n        global assert_close\n"
    "        import payload as assert_close\n    nested()\n    return x\n",
))
def test_editable_scope_import_cannot_overwrite_harness_global(colocated_harness, body):
    entrypoint, source, snapshot = colocated_harness
    entrypoint.write_text(source.replace("def kernel(x):\n    return x\n", "def kernel(x):\n" + body))

    with pytest.raises(RuntimeError, match="kernel score is rejected"):
        verify_workspace_harness(snapshot)


def test_harness_import_cannot_be_deferred_to_editable_kernel(colocated_harness):
    entrypoint, source, snapshot = colocated_harness
    source = source.replace("from clock_library import elapsed\n", "")
    source = source.replace("def kernel(x):\n    return x\n",
                            "def kernel(x):\n    global elapsed\n"
                            "    from clock_library import elapsed\n    return x\n")
    entrypoint.write_text(source)

    with pytest.raises(RuntimeError, match="kernel score is rejected"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("name", ("elapsed", "make_input", "test_performance"))
def test_new_triton_helper_cannot_shadow_protected_binding(colocated_harness, name):
    entrypoint, source, snapshot = colocated_harness
    entrypoint.write_text(source + f"@triton.jit\ndef {name}(*args):\n    return 0.000001\n")

    with pytest.raises(RuntimeError, match="kernel score is rejected"):
        verify_workspace_harness(snapshot)


def test_target_decorator_cannot_make_existing_harness_helper_editable(colocated_harness):
    entrypoint, source, snapshot = colocated_harness
    source = source.replace("def make_input():\n    return torch.empty(16)\n",
                            "def make_input():\n    return []\n")
    source = source.replace("@triton.jit\ndef kernel", "@triton.autotune(configs=make_input())\n"
                            "@triton.jit\ndef kernel")
    entrypoint.write_text(source)

    with pytest.raises(RuntimeError, match="kernel score is rejected"):
        verify_workspace_harness(snapshot)


def test_kernel_only_imports_and_local_aliases_remain_editable(colocated_harness):
    entrypoint, source, snapshot = colocated_harness
    source = source.replace("import math as tuning_math\n", "from math import prod\n"
                            "import tuning_module as local_only\n")
    source = source.replace("def kernel(x):\n    return x\n",
                            "def kernel(x):\n    from helpers import tile as elapsed\n"
                            "    return elapsed(x)\n")
    entrypoint.write_text(source)

    verify_workspace_harness(snapshot)


def test_unused_alias_in_shared_import_remains_editable(colocated_harness):
    entrypoint, source, _snapshot = colocated_harness
    source = source.replace("from clock_library import elapsed\n",
                            "from clock_library import elapsed, tuning_only\n")
    entrypoint.write_text(source)
    snapshot = snapshot_workspace_harness(entrypoint.parent)
    entrypoint.write_text(source.replace("elapsed, tuning_only", "elapsed, new_tuning_only"))

    verify_workspace_harness(snapshot)


@pytest.fixture
def trusted_worker_fixture(tmp_path, runner):
    """Run the real runner/worker/monitor in child processes with fake CPU CUDA.

    Only hardware preflight and torch's device objects are simulated. No GPU
    result is inferred from the fixed five-millisecond fake event interval.
    The canonical timing helper and all integrity/orchestration code are real.
    """
    repository = Path(__file__).resolve().parents[1]
    support = repository / "tasks/head_kernels/_support"
    task = tmp_path / "task"
    for relative in ("scripts", "source", "ut", "ut/candidate_overlay"):
        (task / relative).mkdir(parents=True, exist_ok=True)
    for name in ("_trusted_worker.py", "runtime_integrity.py"):
        shutil.copyfile(support / name, task / "scripts" / name)
    shutil.copyfile(repository / "src/tools/perf/aka_benchmark.py", task / "scripts/_aka_benchmark.py")
    (task / "scripts/runtime_preflight.py").write_text(
        "def require_runtime(config): return {'cpu_test_fixture': True}\n")
    (task / "scripts/torch.py").write_text(
        "from types import SimpleNamespace\n"
        "class Event:\n"
        "    def __init__(self, **kwargs): pass\n"
        "    def record(self): pass\n"
        "    def synchronize(self): pass\n"
        "    def elapsed_time(self, other): return 5.0\n"
        "class Stream:\n"
        "    def wait_stream(self, other): pass\n"
        "    def synchronize(self): pass\n"
        "class CUDAGraph:\n"
        "    def capture_begin(self): pass\n"
        "    def capture_end(self): pass\n"
        "    def replay(self): pass\n"
        "class Tensor: pass\n"
        "def synchronize(): pass\n"
        "def current_stream(): return Stream()\n"
        "def is_available(): return True\n"
        "def assert_close(a, b): assert a == b\n"
        "cuda = SimpleNamespace(Event=Event, Stream=Stream, CUDAGraph=CUDAGraph,\n"
        "    synchronize=synchronize, current_stream=current_stream, is_available=is_available)\n"
        "testing = SimpleNamespace(assert_close=assert_close)\n")
    (task / "ut/harness_lib.py").write_text(
        "def correct(out, ref, tol): return (out == ref, 0)\n"
        "def _correct_one(out, ref, tol): return correct(out, ref, tol)\n"
        "def flatten_outputs(out): return [out]\n"
        "def to_device_like(ref, dev): return ref\n"
        "def _torch(): return None\n")
    (task / "ut/reference_io.pt").write_text("3")
    source = "def kernel(x):\n    return x + 1\n"
    config = {"source_file_path": ["source/kernel.py"], "target_kernel_functions": ["kernel"]}
    (task / "source/kernel.py").write_text(source)
    (task / "config.yaml").write_text(yaml.safe_dump(config))
    runner.TASK_DIR = task
    runner.UT_DIR = task / "ut"
    runner.BUILD_DIR = task / "build"
    (task / "scripts/source_abi.json").write_text(json.dumps(runner.source_abi(config)))
    loader = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('candidate_kernel', {str(task / 'source/kernel.py')!r})\n"
        "candidate = importlib.util.module_from_spec(spec)\n"
        "sys.modules['candidate_kernel'] = candidate\n"
        "spec.loader.exec_module(candidate)\n")
    (task / "ut/candidate_overlay/sitecustomize.py").write_text(loader)
    (task / "ut/unittest.py").write_text(
        "import json, sys\n"
        "if 'candidate_kernel' not in sys.modules:\n"
        + "".join("    " + line + "\n" for line in loader.splitlines())
        + "candidate = sys.modules['candidate_kernel']\n"
        "value = candidate.kernel(2)\n"
        "guard = sys.modules['runtime_integrity'].ACTIVE_GUARD\n"
        "ok, error = guard.compare(value, 3, 0.01)\n"
        "assert ok, 'wrong candidate output'\n"
        "samples = sys.modules['_aka_benchmark'].benchmark_cuda_event_samples(lambda: None, repetition=2)\n"
        "print('CPU_WORKER_PROBE=' + json.dumps({'output': value, 'fake_device_ms': samples}))\n")

    def execute(body, *, overlay=False):
        (task / "source/kernel.py").write_text(source + body)
        compiled, error = runner.run_compile(config)
        assert compiled, error
        runner.overlays = lambda: (None, str(task / "ut/candidate_overlay") if overlay else None)
        ok, error = runner.run_correctness(config, 10)
        report = json.loads((task / "build/correctness_report.json").read_text())
        return ok, error, report

    return task, execute


WORKER_ATTACKS = (
    "timer_class", "timer_method", "comparator", "comparator_code", "benchmark_function",
    "protected_import", "oracle_read", "oracle_write", "module_replace", "profile_disable",
    "guard_class", "guard_state", "captured_comparator", "generated_oracle_reader",
)


def _worker_attack(name, task):
    oracle = repr(str(task / "ut/reference_io.pt"))
    bodies = {
        "timer_class": "import torch\nclass FakeEvent: pass\ntorch.cuda.Event = FakeEvent\n",
        "timer_method": "import torch\ntorch.cuda.Event.elapsed_time = lambda *_: 0.000001\n",
        "comparator": "import sys\nsys.modules['harness_lib'].correct = lambda *_: (True, 0)\n",
        "comparator_code": "import sys\ndef fake(out, ref, tol): return (True, 0)\n"
            "sys.modules['harness_lib'].correct.__code__ = fake.__code__\n",
        "benchmark_function": "import sys\nsys.modules['_aka_benchmark']._event_elapsed_ms = lambda *_: 0.000001\n",
        "protected_import": "import harness_lib\n",
        "oracle_read": f"open({oracle}).read()\n",
        "oracle_write": f"open({oracle}, 'w').write('999')\n",
        "module_replace": "import sys, types\nsys.modules['_aka_benchmark'] = types.ModuleType('_aka_benchmark')\n",
        "profile_disable": "import sys\nsys.setprofile(None)\n",
        "guard_class": "import sys\n"
            "sys.modules['runtime_integrity'].RuntimeIntegrity.check = lambda self: None\n"
            "sys.modules['_aka_benchmark']._event_elapsed_ms = lambda *_: 0.000001\n",
        "guard_state": "import sys\n"
            "guard = sys.modules['runtime_integrity'].ACTIVE_GUARD\n"
            "guard.bindings = []\nguard.harness_fingerprints = {}\nguard.modules = {}\n"
            "sys.modules['_aka_benchmark']._event_elapsed_ms = lambda *_: 0.000001\n",
        "captured_comparator": "import sys\n"
            "sys.modules['runtime_integrity'].ACTIVE_GUARD._reference_correct = lambda *_: (True, 0)\n"
            "def wrong(x): return -999\nkernel.__code__ = wrong.__code__\n",
        "generated_oracle_reader":
            f"exec(compile(\"def kernel(x):\\n    return int(open({oracle}).read())\\n\", '<generated_kernel>', 'exec'), globals())\n",
    }
    return bodies[name]


@pytest.mark.parametrize("overlay", (False, True), ids=("direct", "overlay"))
@pytest.mark.parametrize("attack", WORKER_ATTACKS)
def test_actual_trusted_worker_rejects_known_tampering(trusted_worker_fixture, overlay, attack):
    task, execute = trusted_worker_fixture

    ok, error, report = execute(_worker_attack(attack, task), overlay=overlay)

    assert not ok, f"trusted worker accepted {attack}: {report}"
    assert "IntegrityError" in error, error


@pytest.mark.parametrize("overlay", (False, True), ids=("direct", "overlay"))
def test_actual_trusted_worker_accepts_legitimate_source(trusted_worker_fixture, overlay):
    _task, execute = trusted_worker_fixture

    ok, error, report = execute("", overlay=overlay)

    assert ok, error
    assert any('"fake_device_ms": [5.0, 5.0]' in line for line in report["stdout_tail"])


@pytest.mark.parametrize("overlay", (False, True), ids=("direct", "overlay"))
def test_actual_worker_temporary_class_patch_cannot_falsify_bound_event(
    trusted_worker_fixture, overlay,
):
    _task, execute = trusted_worker_fixture
    body = (
        "import torch, sys\n"
        "def transient(x):\n"
        "    original = torch.cuda.Event.elapsed_time\n"
        "    torch.cuda.Event.elapsed_time = float.__add__\n"
        "    observed = sys.modules['_aka_benchmark']._event_elapsed_ms(torch.cuda.Event(), torch.cuda.Event())\n"
        "    torch.cuda.Event.elapsed_time = original\n"
        "    return x + 1 if observed == 5.0 else -999\n"
        "kernel.__code__ = transient.__code__\n"
    )

    ok, error, report = execute(body, overlay=overlay)

    # A stricter monitor may reject the mutation. If it permits a fully
    # restored method, the bound timer must retain the original interval.
    if not ok:
        assert "IntegrityError" in error, error
    else:
        assert any('"fake_device_ms": [5.0, 5.0]' in line for line in report["stdout_tail"])


def test_actual_performance_runner_rejects_fresh_forgery_and_early_exit(
    trusted_worker_fixture, runner,
):
    """A fresh JSON file and exit zero must not replace completed trusted work."""
    task, _execute = trusted_worker_fixture
    runner.overlays = lambda: (None, None)
    raw = _raw_report()
    for row in raw["test_cases"]:
        row["execution_time_ms"] = 0.000001
    source = task / "source/kernel.py"
    source.write_text(source.read_text() +
        "import sys, os, json\n"
        "if '--phase' in sys.argv and sys.argv[sys.argv.index('--phase') + 1] == 'measure':\n"
        "    destination = sys.argv[sys.argv.index('--out') + 1]\n"
        f"    with open(destination, 'w') as handle: json.dump({raw!r}, handle)\n"
        "    os._exit(0)\n")
    # This small benchmark script supplies fixed CPU reference cases. The
    # actual parent, trusted worker, runtime monitor and process exits are used.
    (task / "scripts/_bench.py").write_text(
        "import argparse, importlib.util, json, sys\nfrom pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "for name in ('ut', 'out', 'reference', 'warmup', 'iters', 'phase'): parser.add_argument('--' + name)\n"
        "args = parser.parse_args()\n"
        "reference = Path(args.reference)\n"
        "if args.phase == 'reference':\n"
        "    reference.write_text('trusted CPU fixture reference')\n"
        f"    reference.with_suffix('.cases.json').write_text(json.dumps({raw['expected_case_ids']!r}))\n"
        "else:\n"
        f"    spec = importlib.util.spec_from_file_location('candidate_kernel', {str(source)!r})\n"
        "    candidate = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(candidate)\n"
        "    raise RuntimeError('this fixture never completed a measurement')\n")

    rows = runner.run_performance({}, 10)

    assert rows == [], f"parent accepted candidate-written measurements: {rows}"
