"""CPU adversarial coverage for the SG520 quant task's trusted boundary."""

import contextlib
import copy
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest


TASK = Path(__file__).resolve().parents[1] / "experimental/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8"


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, TASK / relative)
    result = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, "path", [str(TASK / "ut"), *sys.path]):
        spec.loader.exec_module(result)
    return result


GUARD = module("quant_source_guard_test", "ut/source_guard.py")
ORACLE = module("quant_oracle_test", "ut/oracle.py")
NATIVE = module("quant_native_test", "ut/native.py")
CONTRACT = module("quant_contract_test", "ut/contract.py")
WORKER = module("quant_worker_test", "scripts/worker.py")
BENCHMARK = module("quant_benchmark_test", "scripts/benchmark.py")
ORIGINAL = (TASK / "source/quant_kernels.cu").read_text()
MANIFEST, CASES = ORACLE.load_cases(TASK / "cases.json")


def with_body(body):
    start, end = GUARD.body_boundary(ORIGINAL)
    return ORIGINAL[:start] + body + ORIGINAL[end:]


def test_stock_and_device_body_edits_are_accepted():
    GUARD.validate_source(ORIGINAL, ORIGINAL)
    GUARD.validate_source(with_body(' if (threadIdx.x == 0) { out[0] = {}; } '), ORIGINAL)
    GUARD.validate_source(with_body(' auto value = [] __device__ () { return 1; };\n'
                                    ' // braces } and " in comments\n'
                                    ' const char* text = R"tag( } { " )tag";\n'), ORIGINAL)


@pytest.mark.parametrize("injection", [
    '\nextern "C" __attribute__((destructor)) void forge() {}\n',
    '\n#include "forged.h"\n',
    '\n#define dynamic_per_group_scaled_quant_kernel bogus\n',
])
def test_host_suffix_and_prefix_changes_are_rejected(injection):
    for source in (ORIGINAL + injection, injection + ORIGINAL):
        with pytest.raises(ValueError, match="only the target GPU"):
            GUARD.validate_source(source, ORIGINAL)


@pytest.mark.parametrize("body", [
    '} void forged_host_function() {} void decoy() {',
    '%> void forged_host_function() <% %> void decoy() <%',
    '#include "forged.h"\n',
    '%:include "forged.h"\n',
    '#define forged }\n',
    '_Pragma("GCC poison target")',
    'struct Local { __attribute__((constructor)) static void forged() {} };',
    'struct Local { [[gnu::constructor]] static void forged() {} };',
    'auto bad = [] __host__ () {};',
    '// hide a brace \\\n }',
    '// hide a brace \\ \n }',
    '// hide a brace ??/\n }',
    '/* unterminated',
    'const char* malformed = R"raw( unterminated;',
    '{',
])
def test_gpu_body_cannot_escape_into_host_or_preprocessor(body):
    with pytest.raises(ValueError):
        GUARD.validate_source(with_body(body), ORIGINAL)


def test_frozen_signature_launcher_and_conditional_cannot_change():
    for before, after in (("__launch_bounds__(block_size)", "__launch_bounds__(1)"),
                          ("dim3 const block(dynGroupQuantBlockSize)", "dim3 const block(1)"),
                          ("#if defined(__gfx942__)", "#if 0")):
        with pytest.raises(ValueError):
            GUARD.validate_source(ORIGINAL.replace(before, after, 1), ORIGINAL)


@pytest.mark.parametrize("mutation", [
    lambda m: m["cases"].pop(),
    lambda m: m["cases"].append(copy.deepcopy(m["cases"][0])),
    lambda m: m["cases"].reverse(),
    lambda m: m["cases"][0].update(shape=[128, 1536]),
    lambda m: m["cases"][0].update(trace_call_count=729),
    lambda m: m["cases"][0].update(group_size=64),
    lambda m: m["cases"][0].update(transpose_scale=1),
    lambda m: m["cases"][0].pop("num_rows"),
    lambda m: m["cases"][0].update(scale_allocated_stride=[1, 8192]),
    lambda m: m["cases"][0]["trace_call_counts_per_rank"].pop("7"),
    lambda m: m.update(counts_complete_for_sampled_window=False),
    lambda m: m.update(weight_sum_per_rank=1),
])
def test_workload_manifest_rejects_reduced_or_changed_contract(tmp_path, mutation):
    value = copy.deepcopy(MANIFEST)
    mutation(value)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        ORACLE.load_cases(path)


def test_invoke_passes_exact_frozen_abi():
    fake_torch = SimpleNamespace(float8_e4m3fn=object(), float32=object())
    function = mock.Mock(return_value="output")
    x = object()
    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        assert ORACLE.invoke(function, x) == "output"
    function.assert_called_once_with(x, scale=None, quant_dtype=fake_torch.float8_e4m3fn,
                                    group_size=128, transpose_scale=True, num_rows=None,
                                    num_rows_factor=1, scale_type=fake_torch.float32)


def result_fixture(root, *, mode="performance", leg="candidate_native", pid=12345):
    request = {"run_id": "fresh-nonce", "mode": mode, "leg": leg,
               "package_sha256": "b" * 64, "source_tree_sha256": "a" * 64,
               "challenge_seed": 42}
    name = "module_quant_aka_candidate_" + "a" * 20 + "_fresh"
    extension = root / "build/aiter_jit" / name / (name + ".so")
    if leg == "production_native":
        extension = root.parent / "pinned-runtime.so"
    extension.parent.mkdir(parents=True, exist_ok=True)
    extension.write_bytes(b"fresh compiled device code")
    proof = {"leg": leg, "candidate_module": name, "extension_path": str(extension),
             "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
             "source_tree_sha256": request["source_tree_sha256"],
             "fresh_compilation": True, "production_namespace_rebound": False}
    rows = []
    for case in CASES if mode != "compile" else []:
        rows.append({"case_id": case["case_id"], "shape": case["shape"],
                     "trace_call_count": case["trace_call_count"], "correct": True,
                     "input_immutable": True, "benchmark_method": "cuda_graph",
                     "fresh_input_replays": 100, "poisoned_output_replays": 100,
                     "oracle_checks": 101, "warmup_iterations": 10,
                     "seeds": [0, 1], "negative_controls": True,
                     "timings": CONTRACT.sample_statistics([1.0] * 100)})
    return request, {**request, "status": "ok", "pid": pid, "native_build": proof, "results": rows}


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(run_id="previous-run"),
    lambda p: p.update(pid=os.getpid()),
    lambda p: p.update(package_sha256="old-source"),
    lambda p: p.update(source_tree_sha256="old-kernel"),
    lambda p: p.update(challenge_seed=0),
    lambda p: p["results"].pop(),
    lambda p: p["results"].append(p["results"][0]),
    lambda p: p["results"][0].update(trace_call_count=1),
    lambda p: p["results"][0].update(input_immutable=False),
    lambda p: p["results"][0].update(fresh_input_replays=0),
    lambda p: p["results"][0].update(poisoned_output_replays=0),
    lambda p: p["results"][0].update(oracle_checks=1),
    lambda p: p["results"][0]["timings"].update(mean_ms=0.00001),
    lambda p: p["results"][0]["timings"]["samples_ms"].pop(),
    lambda p: p["results"][0]["timings"]["samples_ms"].__setitem__(0, float("nan")),
    lambda p: p["results"][0]["timings"]["samples_ms"].__setitem__(0, False),
    lambda p: p["native_build"].update(fresh_compilation=False),
    lambda p: p["native_build"].update(extension_sha256="forged"),
    lambda p: p["native_build"].update(source_tree_sha256="uncompiled"),
    lambda p: p["native_build"].update(production_namespace_rebound=True),
])
def test_stale_forged_or_partial_worker_results_fail_closed(tmp_path, mutation):
    request, payload = result_fixture(tmp_path)
    with mock.patch.object(CONTRACT, "ROOT", tmp_path):
        CONTRACT.validate_worker(payload, request, 12345, CASES)
        mutation(payload)
        with pytest.raises(ValueError):
            CONTRACT.validate_worker(payload, request, 12345, CASES)


def test_both_correctness_seeds_are_mandatory(tmp_path):
    request, payload = result_fixture(tmp_path, mode="correctness")
    payload["results"][0]["seeds"] = [0]
    with mock.patch.object(CONTRACT, "ROOT", tmp_path), pytest.raises(ValueError, match="both seeds"):
        CONTRACT.validate_worker(payload, request, 12345, CASES)


def test_score_has_all_three_cases_and_weights_are_diagnostic(tmp_path):
    _, payload = result_fixture(tmp_path)
    for row, timing in zip(payload["results"], [1.0, 10.0, 100.0]):
        row["timings"] = CONTRACT.sample_statistics([timing] * 100)
    results = {leg: copy.deepcopy(payload) for leg in CONTRACT.LEGS}
    report = CONTRACT.performance_report(MANIFEST, CASES, results, {})
    assert len(report["paired_cases"]) == 3
    assert report["weight_sum_per_rank"] == 2184
    score_cases = report["test_cases"]
    assert len(score_cases) == 3
    assert [case["test_case_id"] for case in score_cases] == [case["case_id"] for case in CASES]
    assert [case["shape"] for case in score_cases] == [case["shape"] for case in CASES]
    assert [case["execution_time_ms"] for case in score_cases] == [1.0, 10.0, 100.0]
    assert [case["params"]["trace_call_count"] for case in score_cases] == [728, 488, 968]
    assert all(case["metadata"] == {"benchmark_method": "cuda_graph"} for case in score_cases)
    assert report["weighted_mean_ms"] == {
        leg: (728 + 488 * 10 + 968 * 100) / 2184 for leg in CONTRACT.LEGS
    }


def test_failed_benchmark_clears_stale_scoreable_report(tmp_path):
    output = tmp_path / "build/performance_report.json"
    output.parent.mkdir()
    output.write_text('{"status":"ok","test_cases":[{"execution_time_ms":0.001}]}')
    with (mock.patch.object(BENCHMARK, "ROOT", tmp_path),
          mock.patch.object(BENCHMARK, "run_workers", side_effect=RuntimeError("worker failed")),
          pytest.raises(RuntimeError, match="worker failed")):
        BENCHMARK.main()
    assert not output.exists()


class Buffer:
    def __init__(self, value, log, name, device="cuda"):
        self.value, self.log, self.name = value, log, name
        self.device = device

    def detach(self):
        return self

    def to(self, *, device, copy):
        assert copy is True
        self.log.append(("copy_to", self.name, device))
        return Buffer(self.value, self.log, self.name, device=device)

    def copy_(self, other):
        self.log.append("copy_input")
        self.value = other.value

    def view(self, dtype):
        return self

    def fill_(self, value):
        self.log.append("poison_" + self.name)
        self.value = value


@pytest.mark.parametrize("kind", ["fresh", "no_op", "cached", "partial", "mutate_input", "mutate_input_and_fresh"])
def test_replay_requires_current_input_and_both_outputs(kind):
    log = []
    x, fresh = Buffer(1, log, "input"), Buffer(7, log, "fresh")
    result = (Buffer(2, log, "quant"), Buffer(3, log, "scale"))

    def replay():
        log.append("replay")
        if kind == "no_op":
            return
        value = 1 if kind == "cached" else x.value
        result[0].value = value * 2
        if kind != "partial":
            result[1].value = value * 3
        if kind in ("mutate_input", "mutate_input_and_fresh"):
            x.value = -1
        if kind == "mutate_input_and_fresh":
            fresh.value = -1

    def reference(input_):
        log.append("reference")
        assert input_.value == 7
        assert ("copy_to", "quant", "cpu") in log
        assert ("copy_to", "scale", "cpu") in log
        assert ("copy_to", "input", "cpu") in log
        return (Buffer(input_.value * 2, log, "expected_quant"),
                Buffer(input_.value * 3, log, "expected_scale"))

    def compare(actual, expected, input_):
        log.append("compare")
        assert actual[0].device == actual[1].device == input_.device == "cpu"
        assert (actual[0].value, actual[1].value) == (expected[0].value, expected[1].value)

    def event(*, enable_timing):
        assert enable_timing
        return SimpleNamespace(record=lambda: log.append("event"), elapsed_time=lambda end: 0.25)

    torch = SimpleNamespace(uint8="uint8", equal=lambda a, b: a.value == b.value,
                            cuda=SimpleNamespace(Event=event, synchronize=lambda: log.append("sync")))
    with (mock.patch.object(WORKER, "compare", compare),
          mock.patch.object(WORKER, "reference", reference),
          mock.patch.object(WORKER, "validate_output_metadata", side_effect=lambda actual, input_: log.append("metadata"))):
        if kind == "fresh":
            assert WORKER.replay_with_fresh_input(SimpleNamespace(replay=replay), result, x,
                                                 fresh, torch) == 0.25
            assert log[:10] == [("copy_to", "fresh", "cpu"), "copy_input", "poison_quant",
                                "poison_scale", "event", "replay", "event", "sync", "metadata",
                                ("copy_to", "quant", "cpu")]
            assert log.index("reference") > log.index(("copy_to", "input", "cpu"))
            assert log.index("compare") > log.index("reference")
        else:
            with pytest.raises(AssertionError):
                WORKER.replay_with_fresh_input(SimpleNamespace(replay=replay), result, x,
                                              fresh, torch)


def test_validation_owns_cpu_snapshots_before_reference_can_change_gpu_storage():
    log = []
    x = Buffer(7, log, "input")
    before_cpu = WORKER.cpu_snapshot(x)
    result = (Buffer(14, log, "quant"), Buffer(21, log, "scale"))

    def reference(input_):
        assert input_.value == 7
        log.append("reference")
        # CPU snapshots must survive later changes to every GPU allocation.
        x.value, result[0].value, result[1].value = 99, 99, 99
        return (Buffer(14, log, "expected_quant"), Buffer(21, log, "expected_scale"))

    def compare(actual, expected, input_):
        assert input_.device == "cpu" and input_.value == 7
        assert [value.device for value in actual] == ["cpu", "cpu"]
        assert [value.value for value in actual] == [value.value for value in expected] == [14, 21]

    with (mock.patch.object(WORKER, "reference", reference),
          mock.patch.object(WORKER, "compare", compare),
          mock.patch.object(WORKER, "validate_output_metadata", side_effect=lambda actual, input_: log.append("metadata"))):
        WORKER.validate_completed(result, x, before_cpu, SimpleNamespace(equal=lambda a, b: a.value == b.value))
    assert before_cpu.value == 7
    assert log.index("metadata") < log.index(("copy_to", "quant", "cpu")) < log.index("reference")
    assert log.index(("copy_to", "scale", "cpu")) < log.index("reference")
    assert log.index(("copy_to", "input", "cpu"), 1) < log.index("reference")


def test_invalid_gpu_metadata_is_rejected_before_output_copy_or_oracle():
    with (mock.patch.object(WORKER, "validate_output_metadata", side_effect=AssertionError("alias")),
          mock.patch.object(WORKER, "cpu_snapshot") as snapshot,
          mock.patch.object(WORKER, "reference") as reference,
          pytest.raises(AssertionError, match="alias")):
        WORKER.validate_completed(object(), object(), object(), object())
    snapshot.assert_not_called()
    reference.assert_not_called()


def test_correctness_saves_host_truth_before_each_seed_and_syncs_before_validation():
    log = []
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: log.append("sync")))

    def invoke(function, x):
        log.append("invoke")
        assert log[-2] == ("copy_to", "input", "cpu")
        return "actual"

    def validate(actual, x, before_cpu, torch, *, negative_controls):
        assert log[-1] == "sync"
        assert before_cpu.device == "cpu" and before_cpu is not x
        assert before_cpu.value == x.value
        assert negative_controls is True
        log.append("validate")

    with (mock.patch.dict(sys.modules, {"torch": torch}),
          mock.patch.object(WORKER, "generate", side_effect=lambda case, seed, device: Buffer(seed, log, "input")),
          mock.patch.object(WORKER, "invoke", invoke),
          mock.patch.object(WORKER, "validate_completed", validate),
          mock.patch.object(WORKER, "reference") as reference):
        assert WORKER.correctness(object(), CASES[0]) == {"seeds": [0, 1], "negative_controls": True}
    reference.assert_not_called()
    assert log.count("invoke") == log.count("validate") == 2


def test_candidate_always_compiles_before_loading_a_unique_extension(tmp_path):
    calls = []
    staged = tmp_path / "staged"
    staged.mkdir()
    identity = "a" * 64
    built = {}

    def build_module(md_name, srcs, extra_include):
        calls.append(("build", md_name, srcs))
        path = tmp_path / "build/aiter_jit" / md_name / (md_name + ".so")
        path.write_bytes(b"compiled " + md_name.encode())
        built[md_name] = SimpleNamespace(__file__=str(path))

    def get_module(name):
        calls.append(("load", name))
        assert name in built, "a cache must never be loaded before compilation"
        return built[name]

    core = SimpleNamespace(build_module=build_module, get_module=get_module,
                           get_args_of_build=lambda name: {"extra_include": [], "third_party": []},
                           _pybind_develop_hooks=lambda: (lambda x: x, type(None), lambda x: 0, lambda: 0))
    wrapper = SimpleNamespace(per_group_quant_hip=lambda x: x)
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda value: None))
    with (mock.patch.object(NATIVE, "ROOT", tmp_path),
          mock.patch.object(NATIVE, "_stage_native_sources", return_value=(identity, ["current.cu"], staged)),
          mock.patch.object(NATIVE, "source_identity", return_value=identity),
          mock.patch.object(NATIVE, "_candidate_build_scope", return_value=contextlib.nullcontext()),
          mock.patch.object(NATIVE.importlib.util, "spec_from_file_location", return_value=spec),
          mock.patch.object(NATIVE.importlib.util, "module_from_spec", return_value=wrapper),
          mock.patch.dict(sys.modules, {"aiter": SimpleNamespace(), "aiter.jit": SimpleNamespace(core=core)})):
        NATIVE.candidate()
        NATIVE.candidate()
    assert [call[0] for call in calls] == ["build", "load", "build", "load"]
    assert calls[0][1] != calls[2][1]
    assert calls[0][2] == calls[2][2] == ["current.cu"]


def test_coordinator_uses_two_isolated_workers_and_current_requests(tmp_path):
    commands = []

    def popen(command, **kwargs):
        pid = 40000 + len(commands)
        commands.append(command)
        request = json.loads(Path(command[-2]).read_text())
        _, payload = result_fixture(tmp_path, leg=request["leg"], pid=pid)
        payload.update(request)
        Path(command[-1]).write_text(json.dumps(payload))
        assert kwargs["stdout"] is not None and kwargs["stderr"] == -2
        return SimpleNamespace(pid=pid, wait=lambda timeout: 0)

    with (mock.patch.object(CONTRACT, "ROOT", tmp_path),
          mock.patch.object(CONTRACT, "load_cases", return_value=(MANIFEST, CASES)),
          mock.patch.object(CONTRACT, "source_identity", return_value="a" * 64),
          mock.patch.object(CONTRACT, "package_identity", return_value="b" * 64),
          mock.patch.object(CONTRACT.subprocess, "Popen", side_effect=popen)):
        _, _, results, identity = CONTRACT.run_workers("performance")
    assert len(commands) == 2 and all(command[1] == "-I" for command in commands)
    assert {result["pid"] for result in results.values()} == {40000, 40001}
    assert all(result["run_id"] == identity["run_id"] for result in results.values())
    assert results["candidate_native"]["challenge_seed"] == results["production_native"]["challenge_seed"]
