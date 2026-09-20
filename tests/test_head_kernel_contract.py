"""CPU regressions for imported task identity, protected inputs and replay gates."""
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.preprocessing import setup_workspace
from src.testcases import analyze_benchmark_method_consistency, parse_test_cases_from_json

from head_kernel_test_utils import task_directory

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "tasks/head_kernels/_support"


def load(name):
    spec = importlib.util.spec_from_file_location(f"_test_hk_{name}", SUPPORT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "task"
    for directory in ("source", "ut/kernel_src", "scripts"):
        (root / directory).mkdir(parents=True)
    config = {"task_type": "triton2triton", "source_file_path": ["source/kernel.py"],
              "target_kernel_functions": ["kernel"],
              "compile_command": ["python3 scripts/task_runner.py compile"],
              "correctness_command": ["python3 scripts/task_runner.py correctness"],
              "performance_command": ["python3 scripts/task_runner.py performance"]}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "source/kernel.py").write_text("def kernel(x, scale=1):\n    return x * scale\n")
    (root / "ut/kernel_src/kernel.py").symlink_to("../../source/kernel.py")
    for name in ("task_runner", "_bench"):
        (root / f"scripts/{name}.py").write_bytes((SUPPORT / f"{name}.py").read_bytes())
    (root / "ut/meta.json").write_text("{}")
    (root / "ut/cases.py").write_text("CASES = [32, 64]\n")
    (root / "ut/reference_io.pt").write_bytes(b"immutable oracle")
    runner = load("task_runner")
    runner.TASK_DIR = root
    (root / "scripts/source_abi.json").write_text(json.dumps(runner.source_abi(config)))
    (root / "scripts/artifacts.json").write_text("[]")
    return root


def workspace(task, tmp_path):
    return setup_workspace(str(task / "config.yaml"), tmp_path / "run", "test",
                           logging.getLogger(__name__), task_name="head_kernels/example")


def test_workspace_preserves_editable_alias_and_materializes_helper(task, tmp_path):
    copied = workspace(task, tmp_path)
    alias = copied / "ut/kernel_src/kernel.py"
    assert alias.is_symlink()
    snapshot = snapshot_workspace_harness(copied, task_root=task)
    (copied / "source/kernel.py").write_text("def kernel(x, scale=1):\n    return scale * x\n")
    assert "return scale * x" in alias.read_text()
    verify_workspace_harness(snapshot)
    helper = copied / "scripts/_aka_benchmark.py"
    assert helper.read_bytes() == (ROOT / "src/tools/perf/aka_benchmark.py").read_bytes()


@pytest.mark.parametrize("kind", ["escape", "absolute", "dangling"])
def test_workspace_rejects_nonportable_alias(task, tmp_path, kind):
    outside = tmp_path / "outside.py"
    outside.write_text("outside")
    target = {"escape": "../outside.py", "absolute": str(task / "source/kernel.py"),
              "dangling": "missing.py"}[kind]
    (task / "invalid.py").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        workspace(task, tmp_path)


@pytest.mark.parametrize("relative", ["config.yaml", "scripts/task_runner.py", "scripts/_bench.py",
                                      "scripts/source_abi.json", "scripts/artifacts.json",
                                      "ut/cases.py", "ut/meta.json", "ut/reference_io.pt"])
def test_protected_harness_cases_oracle_and_abi_cannot_be_changed(task, tmp_path, relative):
    copied = workspace(task, tmp_path)
    snapshot = snapshot_workspace_harness(copied, task_root=task)
    (copied / relative).write_text("weakened contract")
    with pytest.raises(RuntimeError):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("definition", ["def kernel(renamed, scale=1): return renamed * scale\n",
                                        "def kernel(x, scale=2): return x * scale\n",
                                        "def renamed_kernel(x, scale=1): return x * scale\n",
                                        "def container():\n    def kernel(x, scale=1): return x * scale\n"])
def test_compile_rejects_target_abi_changes(task, definition):
    runner = load("task_runner")
    runner.TASK_DIR, runner.BUILD_DIR = task, task / "build"
    cfg = yaml.safe_load((task / "config.yaml").read_text())
    assert runner.run_compile(cfg)[0]
    (task / "source/kernel.py").write_text(definition)
    assert not runner.run_compile(cfg)[0]


def test_fixture_hash_gate_runs_without_importing_torch(task):
    runner = load("task_runner")
    runner.TASK_DIR, runner.UT_DIR = task, task / "ut"
    data = (task / "ut/reference_io.pt").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    (task / "ut/meta.json").write_text(json.dumps({"reference_io_sha256": digest}))
    (task / "scripts/artifacts.json").write_text(json.dumps([
        {"filename": "reference_io.pt", "sha256": digest, "size_bytes": len(data)}]))
    runner.verify_fixtures()
    (task / "ut/reference_io.pt").write_bytes(b"x" * len(data))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        runner.verify_fixtures()


def test_long_case_identities_remain_distinct_and_complete():
    bench = load("_bench")
    rows = [{"sig": "same-prefix" * 10 + str(i), "regime": "decode"} for i in range(35)]
    ids = bench.validate_cases(rows)
    assert len(set(ids)) == 35
    assert ids[-1] != ids[-2]
    with pytest.raises(RuntimeError, match="duplicate"):
        bench.validate_cases(rows + rows[:1])


def report_case():
    return {"test_case_id": "full-case", "execution_time_ms": 1.0,
            "benchmark_method": "cuda_graph", "benchmark_samples": 100,
            "benchmark_method_consistent": True,
            "benchmark_replay_probe": "input_change_and_restore",
            "benchmark_state_restore": "all_input_storages_before_each_replay",
            "benchmark_output_validation": "exact_timed_graph_replay"}


@pytest.mark.parametrize("change", ["missing", "duplicate", "method", "host", "nan", "samples"])
def test_incomplete_or_unscoreable_reports_fail(change):
    runner = load("task_runner")
    row = report_case()
    report = {"status": "ok", "expected_case_ids": [row["test_case_id"]], "test_cases": [row]}
    if change == "missing":
        report["test_cases"] = []
    elif change == "duplicate":
        report["test_cases"] *= 2
    elif change == "method":
        row["benchmark_method"] = "cuda_event_fallback"
    elif change == "host":
        row["host_time_ms"] = row.pop("execution_time_ms")
    elif change == "nan":
        row["execution_time_ms"] = float("nan")
    else:
        row["benchmark_samples"] = 1
    with pytest.raises(RuntimeError):
        runner.validate_performance_report(report)


def test_canonical_report_is_consumed_by_arena(tmp_path):
    runner = load("task_runner")
    row = report_case()
    report = {"status": "ok", "expected_case_ids": [row["test_case_id"]], "test_cases": [row]}
    assert runner.validate_performance_report(report) == [row]
    path = tmp_path / "performance_report.json"
    path.write_text(json.dumps(report))
    cases = parse_test_cases_from_json(path)
    assert analyze_benchmark_method_consistency(cases, cases) == (True, [])


@pytest.mark.parametrize("worker_fails", [False, True])
def test_stale_results_are_removed_and_worker_failure_never_falls_back(task, monkeypatch, worker_fails):
    runner = load("task_runner")
    runner.TASK_DIR, runner.BUILD_DIR = task, task / "build"
    runner.UT_DIR = task / "ut"
    runner.BUILD_DIR.mkdir()
    for name in ("performance_report.json", "_bench_raw.json", "_benchmark_reference.pt"):
        (runner.BUILD_DIR / name).write_text(json.dumps({"status": "ok", "test_cases": [report_case()]}))
    monkeypatch.setattr(runner, "run_correctness", lambda *_: (True, None))
    monkeypatch.setattr(runner, "overlays", lambda: (None, None))
    phases = []

    def worker(command, *_, **kwargs):
        phases.append(command[-1])
        if not worker_fails and command[-1] == "reference":
            (runner.BUILD_DIR / "_benchmark_reference.cases.json").write_text('["full-case"]')
            (runner.BUILD_DIR / "_benchmark_reference.pt").write_bytes(b"reference fixture")
            paths = [command[index + 1] for index, item in enumerate(command) if item == "--attest-file"]
            receipt = {"status": "complete", "nonce": command[command.index("--nonce") + 1],
                       "script": command[command.index("--script") + 1],
                       "outputs": {str(Path(path).resolve()): runner.file_digest(path) for path in paths}}
            Path(command[command.index("--completion") + 1]).write_text(json.dumps(receipt))
        return subprocess.CompletedProcess(command, 1 if worker_fails else 0, "", "failure")

    monkeypatch.setattr(runner, "run_process", worker)
    assert runner.run_performance({}, 30) == []
    assert phases == (["reference"] if worker_fails else ["reference", "measure"])
    result = json.loads((runner.BUILD_DIR / "performance_report.json").read_text())
    assert result["status"] == "fail" and result["test_cases"] == []
    assert not (runner.BUILD_DIR / "_benchmark_reference.pt").exists()


def test_candidate_cannot_shrink_parent_baseline_case_declaration():
    runner = load("task_runner")
    row = report_case()
    raw = {"status": "ok", "expected_case_ids": ["full-case"], "test_cases": [row]}
    with pytest.raises(RuntimeError, match="baseline worker"):
        runner.validate_performance_report(raw, ["full-case", "other-captured-case"])


@pytest.mark.parametrize("single", [False, True])
def test_reference_workers_use_live_overlay_without_candidate_registry(single):
    bench = load("_bench")
    row = {"sig": "captured", "regime": "decode", "args": {"positional": ()}}
    current = lambda args: "baseline overlay callable"

    def candidate_only_baseline(args):
        raise AssertionError("candidate registry must not be imported in the reference worker")

    module = SimpleNamespace(call=current, baseline_call=candidate_only_baseline)
    if single:
        module.load_live_case = lambda: row
    else:
        module.load_live_cases = lambda: {"case": row}
    rows, call = bench.selected_cases(module, None, {}, None, True)
    assert call(rows[0]["args"]) == "baseline overlay callable"


def test_timeout_kills_the_worker_process_group(monkeypatch):
    runner = load("task_runner")
    count = 0

    def communicate(timeout=None):
        nonlocal count
        count += 1
        if count == 1:
            raise subprocess.TimeoutExpired(["worker"], timeout)
        return "", ""

    process = SimpleNamespace(pid=12345, communicate=communicate)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: process)
    signals = []
    monkeypatch.setattr(runner.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_process(["worker"], 1, {})
    assert signals == [(12345, runner.signal.SIGKILL)]


def test_state_restore_preserves_aliases_strides_and_dispatch_attributes():
    torch = pytest.importorskip("torch")
    bench = load("_bench")
    storage = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    q = storage[:, ::2]
    other = storage[1:, 1:]
    q.is_shuffled = True
    saved = storage.clone()
    state = bench.InputState({"q": q, "alias": other}, torch, q)
    assert len(state.storages) == 1
    storage.fill_(-999)
    q.is_shuffled = False
    state.restore()
    assert torch.equal(storage, saved)
    assert q.stride() == (6, 2) and q.is_shuffled is True
    assert q.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()
    state.probe_enabled = True
    state.restore()
    assert torch.equal(q, saved[:, ::2] * -0.75 + 0.125)
    state.probe_enabled = False
    state.restore()
    assert torch.equal(storage, saved)


@pytest.mark.parametrize("corruption", [None, "memoized", "truncated", "fallback", "few_samples"])
def test_exact_graph_replay_checks_changed_input_and_full_output(corruption):
    torch = pytest.importorskip("torch")
    bench = load("_bench")
    q = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    row = {"sig": "test-case", "regime": "decode", "args": {"q": q}}
    expected = {"base": q.square(), "probe": (q * -0.75 + 0.125).square()}
    history = []

    def call(values):
        out = values["q"].square()
        return out[:1] if corruption == "truncated" else out

    def benchmark(fn, *, prepare_fn, timed_run, **kwargs):
        prepare_fn()
        captured = fn().clone()

        def replay():
            prepare_fn()
            history.append(q.clone())
            return captured if corruption == "memoized" else fn()

        timed_run._bind(replay, captured)
        return ([1.0] * (1 if corruption == "few_samples" else 3),
                {"benchmark_method": "cuda_event_fallback" if corruption == "fallback" else "cuda_graph"})

    h = SimpleNamespace(to_device_like=lambda value, _: value,
                        correct=lambda out, ref, tol: (out.shape == ref.shape and torch.allclose(out, ref), 1))
    args = (row, call, expected, SimpleNamespace(), h, {"tol": 0.02}, torch, 1, 3, benchmark)
    if corruption:
        with pytest.raises(RuntimeError):
            bench.measure_case(*args)
    else:
        result = bench.measure_case(*args)
        assert result["benchmark_method"] == "cuda_graph"
        assert result["benchmark_output_validation"] == "exact_timed_graph_replay"
        assert len(history) == 3
        assert torch.equal(history[0], history[2])
        assert not torch.equal(history[0], history[1])


def test_all_imported_copies_use_the_same_protected_runner():
    configs = list((ROOT / "tasks/head_kernels").rglob("config.yaml"))
    assert configs
    for config in configs:
        for filename in ("task_runner.py", "_bench.py", "runtime_integrity.py", "_trusted_worker.py"):
            assert (config.parent / "scripts" / filename).read_bytes() == (SUPPORT / filename).read_bytes()


def integrity_fixture(tmp_path, monkeypatch):
    import sys
    import types
    torch = pytest.importorskip("torch")
    benchmark_path = ROOT / "src/tools/perf/aka_benchmark.py"
    spec = importlib.util.spec_from_file_location("_aka_benchmark", benchmark_path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    monkeypatch.setitem(sys.modules, "_aka_benchmark", benchmark)
    harness = types.ModuleType("harness_lib")
    exec("def correct(out, ref, tol): return (out == ref, 0)\n"
         "def _correct_one(out, ref, tol): return correct(out, ref, tol)\n"
         "def flatten_outputs(out): return [out]\n"
         "def to_device_like(ref, dev): return ref\n"
         "def _torch(): return None\n", vars(harness))
    monkeypatch.setitem(sys.modules, "harness_lib", harness)
    module = load("runtime_integrity")
    task = tmp_path / "task"
    (task / "source").mkdir(parents=True)
    (task / "ut").mkdir()
    (task / "scripts").mkdir()
    (task / "ut/reference_io.pt").write_bytes(b"oracle")
    (task / "scripts/harness.py").write_text("trusted")
    guard = module.RuntimeIntegrity(task, torch, benchmark, harness)
    return task, guard, module


@pytest.mark.parametrize("attack", ["timer", "comparator", "code", "benchmark", "import", "oracle",
                                   "write", "module", "disable_monitor"])
def test_source_level_timer_comparator_and_oracle_attacks_are_rejected(tmp_path, monkeypatch, attack):
    import builtins
    import sys
    task, guard, module = integrity_fixture(tmp_path, monkeypatch)
    bodies = {
        "timer": "import torch\ntorch.cuda.Event.elapsed_time = lambda *a: 0.000001\n",
        "comparator": "import sys\nsys.modules['harness_lib'].correct = lambda *a: (True, 0)\n",
        "code": "import sys\ndef fake(out, ref, tol): return (True, 0)\n"
                "sys.modules['harness_lib'].correct.__code__ = fake.__code__\n",
        "benchmark": "import sys\nsys.modules['_aka_benchmark']._event_elapsed_ms = lambda *a: 0.000001\n",
        "import": "import harness_lib\n",
        "oracle": f"open({str(task / 'ut/reference_io.pt')!r}, 'rb').read()\n",
        "write": f"open({str(task / 'scripts/harness.py')!r}, 'w').write('weakened')\n",
        "module": "import sys, types\nsys.modules['torch'] = types.ModuleType('torch')\n",
        "disable_monitor": "import sys\nsys.setprofile(None)\n",
    }
    comparators = dict(vars(sys.modules["harness_lib"]))
    comparator_codes = {name: value.__code__ for name, value in comparators.items()
                        if hasattr(value, "__code__")}
    guard.install()
    try:
        with pytest.raises(module.IntegrityError):
            exec(compile(bodies[attack], str(task / "source/kernel.py"), "exec"), {})
            guard.check()
    finally:
        # An integrity failure terminates a real worker. Restore globals only
        # because these negative fixtures share the pytest interpreter.
        guard.active = False
        guard._setprofile(None)
        builtins.__import__ = guard._import
        for name, original in guard.modules.items():
            sys.modules[name] = original
        for owner, name, original, _ in guard.bindings:
            setattr(owner, name, original)
        vars(sys.modules["harness_lib"]).clear()
        vars(sys.modules["harness_lib"]).update(comparators)
        for name, code in comparator_codes.items():
            getattr(sys.modules["harness_lib"], name).__code__ = code
    assert (task / "scripts/harness.py").read_text() == "trusted"


@pytest.mark.parametrize("generated", [False, True])
def test_runtime_attestation_allows_real_source_computation(tmp_path, monkeypatch, generated):
    task, guard, _ = integrity_fixture(tmp_path, monkeypatch)
    scope = {}
    guard.install()
    try:
        implementation = "def kernel(x): return x + 1\n"
        if generated:
            implementation = f"exec(compile({implementation!r}, '<generated_kernel>', 'exec'), globals())\n"
        exec(compile("import torch\n" + implementation +
                     "result = kernel(torch.tensor([1, 2, 3]))\n",
                     str(task / "source/kernel.py"), "exec"), scope)
        guard.check()
        assert scope["result"].tolist() == [2, 3, 4]
    finally:
        guard.close()


def test_event_primitive_uses_the_pre_candidate_method_binding(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    class InstrumentedEvent:
        def elapsed_time(self, other):
            return 5.0

    monkeypatch.setattr(torch.cuda, "Event", InstrumentedEvent)
    _, guard, module = integrity_fixture(tmp_path, monkeypatch)
    original = InstrumentedEvent.elapsed_time
    guard.install()
    try:
        InstrumentedEvent.elapsed_time = lambda *_: 0.000001
        # CPU instrumentation only: the bound timing primitive retains 5.0
        # even before the integrity check rejects the altered method.
        benchmark = guard.modules["_aka_benchmark"]
        assert benchmark._event_elapsed_ms(InstrumentedEvent(), InstrumentedEvent()) == 5.0
        with pytest.raises(module.IntegrityError, match="elapsed_time"):
            guard.check()
    finally:
        InstrumentedEvent.elapsed_time = original
        guard.close()


@pytest.mark.parametrize("kind", ["attention", "flydsl_moe_stage1", "flydsl_moe_stage2"])
def test_kimi_direct_adapters_do_not_require_generic_overlays_or_package_reexports(tmp_path, monkeypatch, kind):
    runner = load("task_runner")
    runner.UT_DIR = tmp_path / "ut"
    runner.UT_DIR.mkdir()
    if kind == "attention":
        metadata = {"target_callable": "sglang.kernels.ops.attention.decode_attention:_decode_grouped_att_m_fwd"}
        paths = ("bindings.py", "baseline_ref/decode_attention.py.orig", "kernel_src/geak_mla_stage1.py")
    else:
        metadata = {"entry_attr": kind}
        paths = ("flydsl_package.py", "dependency_manifest.json", "baseline_src/flydsl/moe_kernels.py",
                 "kernel_src/flydsl/moe_kernels.py")
    (runner.UT_DIR / "meta.json").write_text(json.dumps(metadata))
    for name in paths:
        path = runner.UT_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if name.endswith(".json") else "# protected adapter fixture\n")
    monkeypatch.setattr(runner, "load_harness", lambda: pytest.fail("obsolete generic overlay loader used"))
    assert runner.overlays() == (None, None)
    if kind != "attention":
        assert not (runner.UT_DIR / "kernel_src/flydsl/__init__.py").exists()
    (runner.UT_DIR / paths[-1]).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        runner.overlays()


def test_kimi_flydsl_cannot_alias_its_frozen_source_tree(tmp_path):
    runner = load("task_runner")
    runner.UT_DIR = tmp_path
    (tmp_path / "meta.json").write_text('{"entry_attr": "flydsl_moe_stage1"}')
    for name in ("flydsl_package.py", "dependency_manifest.json"):
        (tmp_path / name).write_text("{}")
    baseline = tmp_path / "baseline_src/flydsl"
    baseline.mkdir(parents=True)
    (baseline / "moe_kernels.py").write_text("def kernel(): return 1\n")
    (tmp_path / "kernel_src").mkdir()
    (tmp_path / "kernel_src/flydsl").symlink_to("../baseline_src/flydsl", target_is_directory=True)
    with pytest.raises(RuntimeError, match="same source tree"):
        runner.overlays()


def test_kimi_attention_performance_uses_the_protected_resolved_pair(tmp_path):
    import sys
    torch = pytest.importorskip("torch")
    bench = load("_bench")
    bindings_path = task_directory("kimi-k3__fwd_grouped_kernel_stage1") / "ut/bindings.py"
    bindings = bench.load_module("_test_common_kimi_bindings", bindings_path)
    (tmp_path / "baseline_ref").mkdir()
    (tmp_path / "kernel_src").mkdir()
    (tmp_path / "baseline_ref/decode_attention.py.orig").write_text(
        "def _decode_grouped_att_m_fwd(args): return args['value'] + 10\n")
    (tmp_path / "kernel_src/geak_mla_stage1.py").write_text(
        "def make_launcher(base):\n"
        "    def candidate(args): return args['value'] + 20\n"
        "    return candidate\n")
    try:
        baseline, candidate = bindings.resolve_pair(str(tmp_path))
        module = SimpleNamespace(BASELINE_FN=baseline, CANDIDATE_FN=candidate,
                                 baseline_call=baseline, current_call=candidate, DEV="cpu",
                                 _online_buckets=lambda: [("captured-one", 1, 8192, "decode"),
                                                         ("captured-two", 64, 8192, "decode")],
                                 _synth=lambda batch, ctx, rng: {"value": batch})
        base_rows, base_call = bench.selected_cases(module, None, {}, torch, True)
        candidate_rows, candidate_call = bench.selected_cases(module, None, {}, torch, False)
        assert bench.validate_cases(base_rows) == bench.validate_cases(candidate_rows)
        assert [base_call(row["args"]) for row in base_rows] == [11, 74]
        assert [candidate_call(row["args"]) for row in candidate_rows] == [21, 84]
        # Replacing the candidate callable leaves the saved baseline untouched.
        module.CANDIDATE_FN = module.current_call = lambda args: -999
        _, changed = bench.selected_cases(module, None, {}, torch, False)
        assert changed(candidate_rows[0]["args"]) == -999
        assert base_call(base_rows[0]["args"]) == 11
        module.CANDIDATE_FN = baseline
        with pytest.raises(RuntimeError, match="independent"):
            bench.selected_cases(module, None, {}, torch, False)
    finally:
        for name in ("_aka_kimi_mla_frozen_reference", "_aka_kimi_mla_candidate", "_test_common_kimi_bindings"):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("stage", ["flydsl_moe_stage1", "flydsl_moe_stage2"])
def test_kimi_moe_performance_keeps_independent_functions_and_original_output_boundary(stage):
    torch = pytest.importorskip("torch")
    bench = load("_bench")
    baseline = lambda output: output.fill_(2)
    candidate = lambda output: output.fill_(3)
    module = SimpleNamespace(BASELINE_FN=baseline, CANDIDATE_FN=candidate, REGIME="decode",
                             DEV="cpu", TOPK=2, INTER_DIM=3, MODEL_DIM=6,
                             timing_cases=lambda: [{"spec": {"sig": "full-captured-case"}, "regime": "decode", "m": 4}],
                             build_inputs=lambda _: {"token_num": 4, "a": torch.ones(4, 6)},
                             _invoke=lambda fn, args, out, zero=False: fn(out))
    if stage.endswith("stage1"):
        module.MEDIAN_LAUNCHES = 21
    h = SimpleNamespace(compiled_op=lambda fn, regime: fn)
    metadata = {"entry_attr": stage}
    baseline_rows, baseline_call = bench.selected_cases(module, h, metadata, torch, True)
    candidate_rows, candidate_call = bench.selected_cases(module, h, metadata, torch, False)
    assert torch.all(baseline_call(baseline_rows[0]["args"]) == 2)
    output = candidate_rows[0]["args"]["output"]
    assert candidate_call(candidate_rows[0]["args"]).data_ptr() == output.data_ptr()
    assert torch.all(output == 3)
    assert candidate_rows[0]["validation_replays"] == (21 if stage.endswith("stage1") else 1)
    assert output.shape == ((4, 2, 3) if stage.endswith("stage1") else (4, 6))
    module.CANDIDATE_FN = baseline
    with pytest.raises(RuntimeError, match="independent"):
        bench.selected_cases(module, h, metadata, torch, False)


@pytest.mark.parametrize("receipt_kind", ["missing", "wrong_nonce", "changed_output", "valid"])
def test_parent_requires_completed_worker_and_matching_output_digest(tmp_path, monkeypatch, receipt_kind):
    runner = load("task_runner")
    runner.TASK_DIR = tmp_path
    runner.BUILD_DIR = tmp_path / "build"
    script = tmp_path / "scripts/worker.py"
    output = runner.BUILD_DIR / "_bench_raw.json"

    def process(command, *args, **kwargs):
        output.write_text('{"test_cases": []}')
        if receipt_kind != "missing":
            receipt = {"status": "complete", "script": str(script.resolve()),
                       "nonce": command[command.index("--nonce") + 1],
                       "outputs": {str(output.resolve()): runner.file_digest(output)}}
            if receipt_kind == "wrong_nonce":
                receipt["nonce"] = "another invocation"
            Path(command[command.index("--completion") + 1]).write_text(json.dumps(receipt))
            if receipt_kind == "changed_output":
                output.write_text('{"test_cases": ["forged"]}')
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner, "run_process", process)
    result = runner.run_worker(script, [], None, 10, True, attest_files=(output,))
    assert (result.returncode == 0) == (receipt_kind == "valid")
    if receipt_kind != "valid":
        assert "Trusted worker did not finalize" in result.stderr
    assert not list(runner.BUILD_DIR.glob("_worker_completion_*.json"))
