"""CPU tests for literal-only native choices and checked graph replay."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster"
SCRIPTS = TASK / "scripts"
sys.path.insert(0, str(SCRIPTS))
import _routing_contract as contract
import _dense_runtime as runtime
import _native_route_bench as bench


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


META = json.loads((TASK / "ut/meta.json").read_text())
POLICY = (TASK / "source/routing.py").read_bytes()


@pytest.fixture
def workspace(tmp_path):
    target = tmp_path / "task"
    shutil.copytree(TASK, target, symlinks=True, ignore=shutil.ignore_patterns("build", "reports", "__pycache__"))
    return target


@pytest.mark.parametrize("attack", ["module_call", "function_call", "import", "lambda", "decorator", "signature",
                                     "missing", "extra", "duplicate", "unsupported", "dict_comprehension"])
def test_policy_rejects_host_execution_and_incomplete_routes(attack):
    text = POLICY.decode()
    if attack == "module_call": text += "\nraise RuntimeError('host executed')\n"
    elif attack == "function_call": text = text.replace('"recorded"', "__import__('os').getcwd()", 1)
    elif attack == "import": text = "import os\n" + text
    elif attack == "lambda": text = text.replace('"recorded"', "(lambda: 'recorded')()", 1)
    elif attack == "decorator": text = text.replace("def select_routes", "@callback\ndef select_routes")
    elif attack == "signature": text = text.replace("select_routes()", "select_routes(x=callback())")
    elif attack in ("missing", "extra", "duplicate", "unsupported"):
        mapping = contract.parse_policy(POLICY, META)
        if attack == "missing": mapping.pop(next(iter(mapping)))
        elif attack == "extra": mapping["not-a-case"] = "recorded"
        elif attack == "unsupported": mapping[next(iter(mapping))] = "no_op"
        text = "def select_routes():\n    return " + repr(mapping) + "\n"
        if attack == "duplicate": text = text.replace("{", "{" + repr(next(iter(mapping))) + ": 'recorded',", 1)
    else: text = "def select_routes():\n    return {key: value for key, value in records}\n"
    with pytest.raises((ValueError, SyntaxError)):
        contract.parse_policy(text.encode(), META)


def test_comments_and_mapping_order_do_not_create_source_gain(workspace):
    before = contract.source_context(workspace)
    values = dict(reversed(list(before["routes"].items())))
    (workspace / "source/routing.py").write_text("# formatting only\ndef select_routes():\n    return " + repr(values) + "\n")
    after = contract.source_context(workspace)
    assert before["source_file_sha256"] != after["source_file_sha256"]
    assert before["effective_source_sha256"] == after["effective_source_sha256"]


def test_recorded_torch_alias_has_identical_executed_configs_and_source_identity(workspace):
    before = contract.source_context(workspace)
    routes = dict(before["routes"], decode_m64_n32_k8192="torch")
    (workspace / "source/routing.py").write_text("def select_routes():\n    return " + repr(routes) + "\n")
    after = contract.source_context(workspace)
    assert before["source_file_sha256"] != after["source_file_sha256"]
    assert before["routes"] != after["routes"]
    actual_configs = [{row["sig"]: runtime.route_config(row, context["routes"][row["sig"]])
                       for row in contract.specs(META)} for context in (before, after)]
    assert len(actual_configs[0]) == 5 and actual_configs[0] == actual_configs[1]
    assert before["resolved_native_configs"] == after["resolved_native_configs"] == actual_configs[0]
    assert actual_configs[0]["decode_m64_n32_k8192"] == {
        "libtype": "torch", "solidx": 0, "splitK": 0, "kernelName": "native"}
    assert before["effective_source_sha256"] == after["effective_source_sha256"]


def test_changed_native_configuration_changes_effective_source_identity(workspace):
    before = contract.source_context(workspace)
    routes = dict(before["routes"], decode_m64_n32_k8192="triton")
    (workspace / "source/routing.py").write_text("def select_routes():\n    return " + repr(routes) + "\n")
    after = contract.source_context(workspace)
    assert before["resolved_native_configs"] != after["resolved_native_configs"]
    assert before["effective_source_sha256"] != after["effective_source_sha256"]


def test_policy_fifo_is_rejected_without_opening_it(workspace):
    path = workspace / "source/routing.py"
    path.unlink(); os.mkfifo(path)
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; sys.path.insert(0,sys.argv[1]); import _routing_contract as c; c.source_context(sys.argv[2])",
        str(SCRIPTS), str(workspace)], capture_output=True, text=True, timeout=3)
    assert result.returncode != 0 and "regular file" in result.stderr


@pytest.mark.parametrize("relative", ["source/tuned_gemm_candidate.py", "ut/baseline_ref/tuned_gemm.py.orig",
                                     "ut/meta.json", "ut/selection_validation.json", "scripts/_dense_runtime.py"])
def test_frozen_host_native_and_historical_inputs_are_protected(workspace, relative):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
    snapshot = snapshot_workspace_harness(workspace, task_root=TASK)
    path = workspace / relative
    path.write_bytes(path.read_bytes() + b"\n# modified\n")
    with pytest.raises(RuntimeError): verify_workspace_harness(snapshot)


def test_only_literal_route_body_is_an_editable_input(workspace):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
    snapshot = snapshot_workspace_harness(workspace, task_root=TASK)
    path = workspace / "source/routing.py"
    path.write_text(path.read_text().replace('"recorded"', '"torch"', 1))
    verify_workspace_harness(snapshot)
    assert "torch" in contract.source_context(workspace)["routes"].values()


def payload_fixture(workspace):
    request = contract.make_request(workspace, "performance", "a" * 48)
    rows = []
    names = runtime.BACKENDS
    for spec in META["workload"]["cases"]:
        route = request["source"]["routes"][spec["sig"]]
        cfg = runtime.route_config(spec, route)
        values = [0.02 + i * 0.00001 for i in range(100)]
        rows.append({"sig": spec["sig"] + "|" + spec["regime"], "params": contract.case_params(META, spec),
            "selected_route": route, "samples_ms": values, "mean_ms": math.fsum(values) / 100,
            "median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values),
            "benchmark_method": "cuda_graph", "benchmark_warmup": 10, "benchmark_samples": 100,
            "benchmark_effective_repeats": 1, "validation": {"checked_invocations": 114, "fresh_input_sets": 116,
                "expected_device": "cpu", "inputs_unchanged": True, "reference_after_candidate_snapshot": True,
                "tolerance": float(spec.get("tol", META["tol"])), "max_rel_err": 0.0,
                "outputs_poisoned_before_replay": True, "measured_graph_validated": True,
                "native_binding": {"native_source_sha256": request["source"]["native_source_sha256"],
                    "backend": cfg["libtype"], "function": names[cfg["libtype"]], "native_config": cfg,
                    "editable_python_executed": False, "python_invocations": 12}}})
    return request, {"status": "ok", "protocol": contract.PROTOCOL, "request": request,
        "request_sha256": contract.fingerprint(request), "timer": "cuda_graph", "warmup": 10, "iters": 100, "cases": rows}


@pytest.mark.parametrize("attack", ["missing_case", "extra_case", "duplicate", "wrong_shape", "bad_mean", "short_samples",
                                     "stale_request", "event_fallback", "unchecked", "unpoisoned", "wrong_backend", "wrong_config", "boolean_count"])
def test_complete_report_and_actual_dispatch_binding_reject_corruption(workspace, attack):
    runner = load("dense_report_validation", SCRIPTS / "task_runner.py")
    request, raw = payload_fixture(workspace)
    if attack == "missing_case": raw["cases"].pop()
    elif attack == "extra_case": raw["cases"].append(deepcopy(raw["cases"][0]))
    elif attack == "duplicate": raw["cases"][1] = deepcopy(raw["cases"][0])
    elif attack == "wrong_shape": raw["cases"][0]["params"]["m"] = 1
    elif attack == "bad_mean": raw["cases"][0]["mean_ms"] = 0.00001
    elif attack == "short_samples": raw["cases"][0]["samples_ms"].pop()
    elif attack == "stale_request": raw["request"] = dict(request, run_id="b" * 48)
    elif attack == "event_fallback": raw["cases"][0]["benchmark_method"] = "cuda_event"
    elif attack == "unchecked": raw["cases"][0]["validation"]["checked_invocations"] = 113
    elif attack == "unpoisoned": raw["cases"][0]["validation"]["outputs_poisoned_before_replay"] = False
    elif attack == "wrong_backend": raw["cases"][0]["validation"]["native_binding"]["backend"] = "torch"
    elif attack == "wrong_config": raw["cases"][0]["validation"]["native_binding"]["native_config"]["solidx"] = 0
    else: raw["cases"][0]["benchmark_effective_repeats"] = True
    with pytest.raises((ValueError, KeyError)):
        runner._validate_dense_payload(contract, raw, META, request)


def test_outlier_and_order_are_retained_for_authoritative_quality(workspace):
    runner = load("dense_outlier_validation", SCRIPTS / "task_runner.py")
    request, raw = payload_fixture(workspace)
    row = raw["cases"][0]; row["samples_ms"][27] = 68.0
    values = row["samples_ms"]
    row.update(mean_ms=math.fsum(values)/100, median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values))
    rows = runner._validate_dense_payload(contract, raw, META, request)
    assert rows[0]["samples_ms"][27] == 68.0 and len(rows) == 5


@pytest.fixture
def cpu_state(tmp_path):
    native_path = tmp_path / "native.py"
    native_path.write_text("""import torch
def gemm_a16w16(*args, **kwargs):
    raise AssertionError('cached public op must not execute')
def torch_gemm(A,B,solidx,bias,otype,scale_a,scale_b,scale_c,bpreshuffle,config=None):
    return torch.nn.functional.linear(A,B)
solMap={'torch':torch_gemm}
""")
    native = load("fake_dense_image", native_path)
    capturing = {"value": False}
    proxy = SimpleNamespace(**{name: getattr(torch, name) for name in
        ("bfloat16", "uint8", "Generator", "empty_strided", "randn", "nn", "equal", "isfinite", "empty_like")},
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing["value"]))
    spec = {"sig": "tiny", "regime": "decode", "m": 2, "n": 4, "k": 3,
            "a_stride": [3,1], "b_stride": [3,1], "expected_backend": "torch", "expected_solidx": 0,
            "expected_split_k": 0, "expected_kernel": "native"}
    context = {"routes": {"tiny": "torch"}, "native_source_sha256": contract.file_sha(native_path)}
    meta = {"native_routing": {"kind": "dense"}, "tol": 0.02}
    case = runtime.DenseCase(proxy, meta, spec, context, 123, device="cpu", native=native)
    def compare(actual, expected, tol):
        atol = tol * expected.float().square().mean().sqrt().clamp_min(1e-6)
        delta = (actual.float() - expected.float()).abs()
        return bool((delta <= atol + tol * expected.float().abs()).all()), float((delta / (expected.float().abs() + atol)).max())
    return runtime.ReplayState(case, compare), capturing


def graph_model(capturing, *, stale=False, wrong=False, mutate=False, wrong_binding=False):
    """CPU model of canonical preparation/capture/replay ordering, not GPU evidence."""
    def benchmark(run, warmup, repetition, prepare_fn, timed_run):
        state = run.__self__
        def capture():
            prepare_fn(); capturing["value"] = True
            output = run(); capturing["value"] = False
            cached = output.clone()
            def replay():
                if mutate: state.case.args["A"].add_(1)
                value = cached if stale else torch.nn.functional.linear(state.case.args["A"], state.case.args["B"])
                output.copy_(value)
                if wrong: output.reshape(-1)[0] += 1024
            return output, replay
        for _ in range(warmup): prepare_fn(); run()
        _, estimate = capture()
        for _ in range(2): prepare_fn(); estimate()
        output, replay = capture()
        prepare_fn(); replay()
        for _ in range(repetition): prepare_fn(); replay()
        def rerun():
            prepare_fn(); replay()
            return output.clone() if wrong_binding else output
        timed_run._bind(rerun, output)
        return [0.02 + i*0.00001 for i in range(repetition)], {
            "benchmark_method":"cuda_graph", "benchmark_warmup":warmup,
            "benchmark_samples":repetition, "benchmark_effective_repeats":1}
    return benchmark


def test_every_graph_invocation_gets_fresh_inputs_and_CPU_validation(cpu_state):
    state, capturing = cpu_state
    values, _ = bench.checked_graph(state, benchmark=graph_model(capturing))
    assert len(values) == 100 and state.checked == 114 and state.case.generated == 116
    assert state.case.binding()["function"] == "torch_gemm"
    assert state.case.calls == 12 and state.case.binding()["editable_python_executed"] is False


@pytest.mark.parametrize("attack", ["stale", "wrong", "mutate", "wrong_binding"])
def test_graph_stale_output_mutation_and_wrong_graph_are_rejected(cpu_state, attack):
    state, capturing = cpu_state
    with pytest.raises((ValueError, RuntimeError)):
        bench.checked_graph(state, benchmark=graph_model(capturing, **{attack: True}))


def test_native_binding_cannot_be_redirected(cpu_state):
    state, _ = cpu_state
    state.case.native.solMap["torch"] = lambda *args: None
    with pytest.raises(ValueError, match="binding changed"): state.case.invoke()


def test_correctness_preserves_eager_independence_and_checks_graph(cpu_state, monkeypatch):
    state, capturing = cpu_state
    exemplar = state.case
    meta = dict(exemplar.meta, random_draws=3)
    monkeypatch.setattr(bench, "DenseCase", lambda torch_arg, meta_arg, spec, context, seed:
        runtime.DenseCase(exemplar.torch, meta_arg, spec, context, seed, device="cpu", native=exemplar.native))
    original = bench.checked_graph
    monkeypatch.setattr(bench, "checked_graph", lambda state_arg, **kwargs:
        original(state_arg, **kwargs, benchmark=graph_model(capturing)))
    row = bench.correctness_case(exemplar.torch, meta, exemplar.spec, exemplar.context, state.compare, 123)
    assert row["validation"]["checked_invocations"] == 14
    assert row["validation"]["prior_outputs_unchanged"] and row["validation"]["independent_output_storage"]
    assert row["validation"]["negative_controls"] == {"no_op":"rejected", "wrong_output":"rejected"}


def test_source_policy_compile_does_not_execute_callbacks(workspace):
    sentinel = workspace / "callback-ran"
    (workspace / "source/routing.py").write_text(f"def select_routes():\n    return __import__('pathlib').Path({str(sentinel)!r}).touch()\n")
    result = subprocess.run([sys.executable, "-B", str(workspace / "scripts/task_runner.py"), "compile"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode != 0 and not sentinel.exists()
    assert json.loads((workspace / "build/compile_report.json").read_text())["status"] == "fail"


def test_canonical_helper_is_materialized_for_isolated_worker(workspace):
    from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
    materialize_perf_helpers_in_workspace(workspace)
    helper = workspace / "scripts/_aka_benchmark.py"
    assert helper.read_bytes() == (ROOT / "src/tools/perf/aka_benchmark.py").read_bytes()
    wrapper = (workspace / "scripts/_bench.py").read_text()
    assert "sys.path.insert" in wrapper and "_native_route_bench" in wrapper


def test_ordinary_runner_publishes_only_complete_request_bound_rows(workspace, monkeypatch):
    runner = load("dense_runner_complete_case", SCRIPTS / "task_runner.py")
    runner.TASK_DIR, runner.UT_DIR, runner.BUILD_DIR = str(workspace), str(workspace / "ut"), str(workspace / "build")
    runner.CONFIG = str(workspace / "config.yaml")
    (workspace / "build").mkdir()
    commands = []
    class Worker:
        pid = 111
        def __init__(self, command, **kwargs):
            commands.append((command, kwargs))
            request = json.loads(Path(command[command.index("--request") + 1]).read_text())
            _, raw = payload_fixture(workspace)
            raw["request"] = request; raw["request_sha256"] = contract.fingerprint(request)
            Path(command[command.index("--out") + 1]).write_text(json.dumps(raw))
        def wait(self, timeout): return 0
    monkeypatch.setattr(runner.subprocess, "Popen", Worker)
    assert runner._run_routing("performance", 10)
    report = json.loads((workspace / "build/performance_report.json").read_text())
    assert report["status"] == "ok" and len(report["test_cases"]) == 5
    assert all(len(row["samples_ms"]) == 100 for row in report["test_cases"])
    assert commands[0][0][1:3] == ["-I", "-B"] and commands[0][1]["env"]["PYTHONPATH"] == ""
    raw = json.loads((workspace / "build/_bench_raw.json").read_text())
    assert report["test_cases"] == runner._benchmark_cases(raw)
    assert len(list((workspace / "build/native-routing-runs").glob("*/raw.json"))) == 1
