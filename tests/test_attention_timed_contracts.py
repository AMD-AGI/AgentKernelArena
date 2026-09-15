"""Actual task-harness regression controls; CPU tests do not certify GPU runs."""
import hashlib
import importlib.util
import inspect
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
MERGE = ROOT / "tasks/triton2triton/vllm/triton_merge_attn_states"
NONE_DIAG = ROOT / "tasks/triton2triton/vllm/triton_lightning_attn_none_diag"


def load(path):
    cwd = Path.cwd()
    try:
        spec = importlib.util.spec_from_file_location("attention_test_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(cwd)


def cpu_inputs(monkeypatch):
    for name in ("randn", "rand", "arange", "tensor"):
        original = getattr(torch, name)
        def factory(*args, _original=original, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = "cpu"
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, factory)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


def merge_cpu(output, p, pl, s, sl, output_lse=None):
    # Independently use normalized softmax rather than the checker's logaddexp.
    logits = torch.stack((pl.double(), sl.double()))
    logits = torch.where(torch.isposinf(logits), -torch.inf, logits)
    weights = logits.softmax(0).permute(0, 2, 1).unsqueeze(-1)
    output.copy_(p.double() * weights[0] + s.double() * weights[1])
    if output_lse is not None:
        output_lse.copy_(logits.logsumexp(0))


def test_merge_known_answer_lse_and_empty_partition():
    checks = load(MERGE / "_arena_checks.py")
    p = torch.tensor([[[2., 4.]], [[3., 5.]], [[7., 9.]]], dtype=torch.float16)
    s = torch.tensor([[[6., 8.]], [[1., 2.]], [[4., 6.]]], dtype=torch.float16)
    pl = torch.tensor([[0., torch.inf, math.log(2)]])
    sl = torch.tensor([[math.log(3), 0., -torch.inf]])
    expected = torch.tensor([[[5., 7.]], [[1., 2.]], [[7., 9.]]], dtype=torch.float16)
    expected_lse = torch.tensor([[math.log(4), 0., math.log(2)]])
    actual, lse = checks.reference((p, pl, s, sl))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse)
    out, out_lse = torch.empty_like(p), torch.empty_like(pl)
    merge_cpu(out, p, pl, s, sl, out_lse)
    checks.check_output(out, expected)
    checks.check_output(out_lse, expected_lse)


@pytest.mark.parametrize("fault", ["none", "ignored_lse", "wrong_lse", "lse_dtype", "ignored_inf",
                                   "tail", "wrong_output", "wrong_dtype", "nonfinite", "mutate", "return"])
def test_merge_actual_correctness_optional_output_and_sentinels(monkeypatch, fault):
    cpu_inputs(monkeypatch)
    harness = load(MERGE / "scripts/task_runner.py")
    checks = load(MERGE / "_arena_checks.py")
    calls = []
    def candidate(out, p, pl, s, sl, output_lse=None):
        calls.append((tuple(out.shape), output_lse is not None))
        if fault == "ignored_inf" and torch.isposinf(pl).any():
            out.fill_(torch.nan)
        else:
            merge_cpu(out, p, pl, s, sl, None if fault == "ignored_lse" else output_lse)
        if output_lse is not None:
            if fault == "wrong_lse": output_lse.zero_()
            if fault == "lse_dtype": output_lse.data = output_lse.double()
        if fault == "tail" and out.shape[-1] == 17: out[..., -1].zero_()
        if fault == "wrong_output": out.zero_()
        if fault == "wrong_dtype": out.data = out.float()
        if fault == "nonfinite": out.fill_(torch.inf)
        if fault == "mutate": p.zero_()
        if fault == "return": return out
    module = SimpleNamespace(merge_attn_states=candidate)
    harness.load_module = lambda: module
    checks.install(harness)
    ok, reason = harness.run_correctness()
    assert ok is (fault == "none"), reason
    if fault == "none":
        assert [shape for shape, _ in calls if shape[-1] != 17] == harness.TEST_SHAPES
        assert [lse for shape, lse in calls if shape[-1] == 17] == [False, True]
    assert module.merge_attn_states is candidate


@pytest.mark.parametrize("fault", ["none", "wrong_timed", "wrong_replay", "stale", "no_write",
                                   "mutate_timed", "mutate_replay", "raise_replay"])
def test_merge_actual_performance_and_exact_replay(monkeypatch, fault):
    cpu_inputs(monkeypatch)
    harness = load(MERGE / "scripts/task_runner.py")
    checks = load(MERGE / "_arena_checks.py")
    harness._TimedRun = load(ROOT / "src/tools/perf/aka_benchmark.py").TimedRun
    module = SimpleNamespace(merge_attn_states=merge_cpu)
    harness.load_module = lambda: module
    inputs, originals, options = [], [], []
    def benchmark(measured, *, timed_run, **kwargs):
        inner = inspect.getclosurevars(measured).nonlocals["fn"]
        state = inspect.getclosurevars(inner).nonlocals
        values = tuple(state[k] for k in ("prefix_output", "prefix_lse", "suffix_output", "suffix_lse"))
        inputs.append(values)
        originals.append(checks.snapshots(values))
        options.append(kwargs)
        out = measured()
        cache = out.clone()
        if fault == "wrong_timed": out.zero_()
        if fault == "mutate_timed": values[0].zero_()
        def replay():
            if fault == "raise_replay": raise RuntimeError("replay failed")
            if fault == "stale": out.copy_(cache)
            elif fault != "no_write": measured()
            if fault == "wrong_replay": out.zero_()
            if fault == "mutate_replay": values[1].zero_()
            return out
        timed_run._bind(replay, out)
        return 0.125, {"benchmark_method": "cuda_graph"}
    harness._benchmark_cuda_graph_or_events = benchmark
    checks.install(harness)
    rows = harness.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for index, row in enumerate(rows):
        assert row["test_case_id"] == f"perf{index + 1}"
        assert row["execution_time_ms"] == (0.125 if fault == "none" else -1.0)
        if fault == "none": assert row["perturbed_input_replay_checked"]
    for values, pristine in zip(inputs, originals): checks.unchanged(values, pristine)
    assert harness._benchmark_cuda_graph_or_events is benchmark


def test_merge_adapter_installs_checks_and_preserves_original_work(monkeypatch):
    monkeypatch.chdir(ROOT)
    adapter = load(MERGE / "_arena_eval.py")
    harness = adapter.load_harness()
    assert harness.run_correctness.__module__ == harness.run_performance.__module__ == "_merge_attn_checks"
    expected = {
        "scripts/task_runner.py": "4fbde456efdbe836ffba0066a7b93d4c8c185151acb04170d2de43ed07d33e43",
        "source/triton_merge_attn_states.py": "70135d9ee3edbb5351a4245fbf260f923b4ff098718634dcf55071ad3ce57c66",
        "workloads.json": "5438ddd30db1d80b3231903af014939a0501a96ed5d7134a74d865617006ef7c",
        "config.yaml": "e7d6291904f5a87e3b14a561e60a781b7074391c863c3568e74dca4709e43d9b",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((MERGE / relative).read_bytes()).hexdigest() == digest


def none_diag_cpu(q, out, slope, kv, BLOCK=256, CBLOCK=64):
    n = q.shape[2]
    slope = slope.reshape(1, -1, 1, 1).double()
    for block, start in enumerate(range(0, n, BLOCK)):
        end = min(start + BLOCK, n)
        decay = (-slope * torch.arange(end-start, dtype=torch.float64).reshape(1, 1, -1, 1)).exp()
        contribution = q[:, :, start:end].double() @ kv[:, :, block].double()
        out[:, :, start:end].copy_(out[:, :, start:end].double() + contribution * decay)
    return out


def test_none_diag_independent_second_block_known_answer():
    harness = load(NONE_DIAG / "scripts/task_runner.py")
    checks = load(NONE_DIAG / "_arena_checks.py")
    q = torch.ones(1, 1, 5, 2, dtype=torch.float16)
    out = torch.full((1, 1, 5, 2), 0.25, dtype=torch.float16)
    kv = torch.stack((torch.eye(2), 2*torch.eye(2))).reshape(1, 1, 2, 2, 2)
    slope = torch.tensor([math.log(2)]).reshape(1, 1, 1, 1)
    wanted = torch.tensor([1.25, .75, .5, .375, 2.25], dtype=torch.float16).reshape(1, 1, 5, 1).expand_as(out)
    checked = checks.reference(harness, q, out, slope, kv, 4, 2)
    torch.testing.assert_close(checked, wanted, atol=0, rtol=0)
    actual = none_diag_cpu(q, out.clone(), slope, kv, 4, 2)
    torch.testing.assert_close(actual, wanted, atol=0, rtol=0)


@pytest.mark.parametrize("fault", ["none", "wrong", "no_write", "return_copy", "return_none",
                                   "dtype", "nonfinite", "mutate_q", "mutate_kv", "first_block_only", "tail"])
def test_none_diag_actual_correctness_and_in_place_contract(monkeypatch, fault):
    cpu_inputs(monkeypatch)
    harness = load(NONE_DIAG / "scripts/task_runner.py")
    checks = load(NONE_DIAG / "_arena_checks.py")
    calls = []
    def candidate(q, out, s, kv, BLOCK=256, CBLOCK=64):
        calls.append(tuple(q.shape))
        if fault != "no_write": none_diag_cpu(q, out, s, kv, BLOCK, CBLOCK)
        if fault == "wrong": out.zero_()
        if fault == "return_copy": return out.clone()
        if fault == "return_none": return None
        if fault == "dtype": out.data = out.float()
        if fault == "nonfinite": out.fill_(torch.nan)
        if fault == "mutate_q": q.zero_()
        if fault == "mutate_kv": kv.zero_()
        if fault == "first_block_only" and q.shape[2] > BLOCK: out[:, :, BLOCK:].zero_()
        if fault == "tail" and q.shape[2] % CBLOCK: out[:, :, -1].zero_()
        return out
    module = SimpleNamespace(lightning_attn_none_diag_forward=candidate)
    harness.load_module = lambda: module
    checks.install(harness)
    ok, reason = harness.run_correctness()
    assert ok is (fault == "none"), reason
    if fault == "none":
        assert [s for s in calls if s[2] != 273] == [tuple(s[:4]) for s in harness.TEST_SHAPES]
        assert calls.count((1, 2, 273, 32)) == 1
    assert module.lightning_attn_none_diag_forward is candidate


@pytest.mark.parametrize("fault", ["none", "wrong_timed", "wrong_replay", "stale", "no_write",
                                   "skip_reset", "mutate_timed", "mutate_replay", "raise_replay"])
def test_none_diag_real_preparation_and_timed_replay(monkeypatch, fault):
    cpu_inputs(monkeypatch)
    harness = load(NONE_DIAG / "scripts/task_runner.py")
    checks = load(NONE_DIAG / "_arena_checks.py")
    harness._TimedRun = load(ROOT / "src/tools/perf/aka_benchmark.py").TimedRun
    harness.load_module = lambda: SimpleNamespace(lightning_attn_none_diag_forward=none_diag_cpu)
    inputs, originals, outputs, output_originals, options = [], [], [], [], []
    def benchmark(measured, *, timed_run, prepare_fn, **kwargs):
        inner = inspect.getclosurevars(measured).nonlocals["fn"]
        state = inspect.getclosurevars(inner).nonlocals
        diagonal = inspect.getclosurevars(prepare_fn).nonlocals["o"]
        values = (*[state[k] for k in ("q", "s", "kv")], diagonal)
        inputs.append(values); originals.append(checks.snapshots(values)); options.append(kwargs)
        outputs.append(state["output_work"]); output_originals.append(state["output_work"].clone())
        # Two prepared calls detect whether preparation actually resets the
        # in-place accumulator rather than retaining the previous sample.
        prepare_fn(); measured(); prepare_fn(); out = measured()
        cache = out.clone()
        if fault == "wrong_timed": out.zero_()
        if fault == "mutate_timed": values[0].zero_()
        def replay():
            if fault == "raise_replay": raise RuntimeError("replay failed")
            if fault != "skip_reset": prepare_fn()
            if fault == "stale": out.copy_(cache)
            elif fault != "no_write": measured()
            if fault == "wrong_replay": out.zero_()
            if fault == "mutate_replay": values[1].zero_()
            return out
        timed_run._bind(replay, out)
        return .125, {"benchmark_method": "cuda_graph", "benchmark_effective_repeats": 1}
    harness._benchmark_cuda_graph_or_events = benchmark
    checks.install(harness)
    rows = harness.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for index, row in enumerate(rows):
        assert row["test_case_id"] == f"perf{index + 1}"
        assert row["execution_time_ms"] == (.125 if fault == "none" else -1.)
        if fault == "none":
            assert row["prepared_in_place_replay_checked"]
            assert row["benchmark_effective_repeats"] == 1
    for values, pristine in zip(inputs, originals): checks.unchanged(values, pristine)
    checks.unchanged(outputs, output_originals)
    assert harness._benchmark_cuda_graph_or_events is benchmark


def test_none_diag_adapter_and_original_work_preserved(monkeypatch):
    monkeypatch.chdir(ROOT)
    adapter = load(NONE_DIAG / "_arena_eval.py")
    harness = adapter.load_harness()
    assert harness.run_correctness.__module__ == harness.run_performance.__module__ == "_lightning_none_diag_checks"
    expected = {
        "scripts/task_runner.py": "d463f95df34e3bfb88b8a2f70863e698b8df95a329c89b3541cf4b0531a1871e",
        "source/triton_lightning_attn_none_diag.py": "c63666fd8a2595722cd130348a782fa974582f66bf669bd53bc34ddb60b107e9",
        "workloads.json": "408294485155d27795ca3e46df2b786ef31897fd2843636c0707537f3768e26a",
        "config.yaml": "97fd2165ca404fbbe31b101b2556db056b10f87874df31dd45d04f86987427c8",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((NONE_DIAG / relative).read_bytes()).hexdigest() == digest
