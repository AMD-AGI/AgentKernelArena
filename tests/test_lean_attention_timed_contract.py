"""Small CPU process fixtures for the real protected paged-attention harness."""
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/triton2triton/geak_eval/L2/lean_atten_paged"


def load(path):
    spec = importlib.util.spec_from_file_location("lean_test_"+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(monkeypatch, fault="none"):
    # Retain the actual 4 correctness and 7 benchmark configuration identities;
    # only the synthetic CPU tensors are small. This is not GPU qualification.
    names = {"_DTYPE", "CORRECTNESS_CONFIGS", "ALL_CONFIGS"}
    nodes = [node for node in ast.parse((TASK/"kernel.py").read_text()).body
             if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
    ns = {"torch": torch}
    exec(compile(ast.Module(nodes, type_ignores=[]), "protected_tables", "exec"), ns)
    records = []
    def make_case(*cfg):
        batch, _, nq, *_ = cfg
        q = (torch.arange(2*batch*nq*4).reshape(2, batch*nq, 4) % 9 / 16).half()
        k = (torch.arange(2*batch*4*4).reshape(2, batch*4, 4) % 7 / 8).half()
        v = (torch.arange(k.numel()).reshape_as(k) % 11 / 8 - .25).half()
        tables = torch.arange(batch*4).reshape(batch, 4).flip(-1).repeat(2, 1, 1)
        case = dict(q=q, k=k, v=v, kv_block_tables=tables,
                    ref_indices=[[r.clone() for r in head] for head in tables],
                    batch_num_block_n=torch.arange(batch+1, dtype=torch.int32),
                    Mp=torch.zeros(2, 2), Lp=torch.zeros(2, 2), Op=torch.zeros(2, 2, 4),
                    locks=torch.zeros(2, dtype=torch.int32), sm_scale=.5, num_warps=4, waves_per_eu=2)
        records.append((cfg, case))
        return case
    def candidate(**kw):
        q, k, v = (kw[n] for n in ("q", "k", "v"))
        out = torch.empty_like(q)
        nq = q.shape[1] // kw["batch_size"]
        for head in range(q.shape[0]):
            for b in range(kw["batch_size"]):
                indices = kw["kv_block_tables"][head, b]
                queries = q[head, b*nq:(b+1)*nq].double()
                logits = queries @ k[head, indices].double().T * kw["sm_scale"]
                weights = (logits-logits.max(-1, keepdim=True).values).exp()
                weights = (weights/weights.sum(-1, keepdim=True)).to(q.dtype)
                out[head, b*nq:(b+1)*nq] = weights.double() @ v[head, indices].double()
        for name in ("Mp", "Lp", "Op"): kw[name].fill_(1)
        kw["locks"].fill_(1)
        if fault == "wrong": out.zero_()
        if fault == "dtype": out = out.float()
        if fault == "shape": out = out[..., :1]
        if fault == "nonfinite": out.fill_(torch.nan)
        if fault == "mutate_q": q.zero_()
        if fault == "mutate_map": kw["kv_block_tables"].zero_()
        return out
    stub = SimpleNamespace(persistent_lean_attention_paged=candidate, _make_test_case=make_case,
                           _config_tag=lambda *args: str(args), CORRECTNESS_CONFIGS=ns["CORRECTNESS_CONFIGS"],
                           ALL_CONFIGS=ns["ALL_CONFIGS"], HARNESS_CONFIGS=ns["ALL_CONFIGS"],
                           PROFILE_CONFIGS=ns["ALL_CONFIGS"][:5], ATOL=1e-2, RTOL=3e-3)
    monkeypatch.setitem(sys.modules, "kernel", stub)
    monkeypatch.setitem(sys.modules, "_aka_benchmark", load(ROOT/"src/tools/perf/aka_benchmark.py"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    h = load(TASK/"test_kernel_harness.py")
    checks = load(TASK/"_arena_checks.py")
    monkeypatch.setitem(sys.modules, "test_kernel_harness", h)
    monkeypatch.setitem(sys.modules, "_arena_checks", checks)
    return h, checks, records


@pytest.mark.parametrize("fault", ["none", "wrong", "dtype", "shape", "nonfinite", "mutate_q", "mutate_map"])
def test_lean_real_correctness_action_and_all_original_cases(monkeypatch, fault):
    h, checks, records = fixture(monkeypatch, fault)
    actions = load(TASK/"_arena_actions.py")
    adapter = load(TASK/"_arena_eval.py")
    if fault == "none":
        actions.correctness(adapter.require_success)
        assert [r[0] for r in records] == [*h.CORRECTNESS_CONFIGS, *h.ALL_SHAPES]
        assert len(records) == 11
    else:
        with pytest.raises(RuntimeError, match="Correctness did not complete"):
            actions.correctness(adapter.require_success)


@pytest.mark.parametrize("fault", ["none", "wrong_timed", "wrong_replay", "stale", "no_write",
                                   "skip_reset", "stale_scratch", "mutate_timed", "mutate_replay", "raise_replay"])
def test_lean_protocol_captures_checked_replay_and_original_lock_preparation(monkeypatch, fault):
    h, checks, records = fixture(monkeypatch)
    options, saved = [], []
    def benchmark(fn, *, timed_run, prepare_fn, **kwargs):
        case = inspect.getclosurevars(fn).nonlocals["case"]
        tensors = checks.readonly(case)
        scratch = tuple(case[n] for n in ("Mp", "Lp", "Op", "locks"))
        saved.append((tensors, checks.snapshots(tensors), scratch, checks.snapshots(scratch)))
        options.append(kwargs)
        assert prepare_fn.__self__ is case["locks"]
        prepare_fn(); out = fn(); cache = out.clone()
        if fault == "wrong_timed": out.zero_()
        if fault == "mutate_timed": case["q"].zero_()
        def replay():
            if fault == "raise_replay": raise RuntimeError("replay failed")
            if fault != "skip_reset": prepare_fn()
            assert torch.equal(case["locks"], torch.zeros_like(case["locks"])), "lock preparation missing"
            if fault == "stale": out.copy_(cache)
            elif fault == "stale_scratch": out.fill_(case["Mp"][0, 0])
            elif fault != "no_write": out.copy_(fn())
            if fault == "wrong_replay": out.zero_()
            if fault == "mutate_replay": case["kv_block_tables"].zero_()
            return out
        timed_run._bind(replay, out)
        return .125, {"benchmark_method": "cuda_graph", "benchmark_effective_repeats": 1}
    h.benchmark_cuda_graph_or_events = benchmark
    actions = load(TASK/"_arena_actions.py")
    before = h.benchmark_cuda_graph_or_events
    checks.install(h)
    assert h.benchmark_cuda_graph_or_events is before  # idempotent installation
    adapter = load(TASK/"_arena_eval.py")
    monkeypatch.setattr(adapter, "load_actions", lambda: actions)
    result = adapter.evaluate("candidate", "performance")
    assert result["status"] == ("PASS" if fault == "none" else "FAIL")
    assert all(o == dict(warmup=50, repetition=200) for o in options)
    if fault == "none":
        assert len(result["cases"]) == len(options) == len(records) == 7
        assert [r[0] for r in records] == h.ALL_SHAPES
        for row in result["cases"]:
            assert row["execution_time_ms"] == .125
            md = row["metadata"]["device_timing"]
            assert md["prepared_lock_replay_checked"] and md["scratch_reinitialized_checked"]
            assert md["benchmark_effective_repeats"] == 1
    for tensors, original, scratch, scratch_original in saved:
        checks.unchanged(tensors, original)
        checks.unchanged(scratch, scratch_original)


def test_lean_original_reference_known_uniform_attention_and_preserved_sources(monkeypatch):
    h, _, _ = fixture(monkeypatch)
    q, k = torch.zeros(1, 2, 2, dtype=torch.float16), torch.zeros(1, 3, 2, dtype=torch.float16)
    v = torch.tensor([[[2., 4.], [99., 99.], [6., 8.]]], dtype=torch.float16)
    actual = h.torch_op(q, k, v, [[torch.tensor([2, 0])]], 2, .5)
    torch.testing.assert_close(actual, torch.tensor([[[4., 6.], [4., 6.]]], dtype=torch.float16))
    expected = {
        "test_kernel_harness.py": "e847c22256ee10c3d3e36d2b3c63441890bd81df1bcc4eae2bb7b45935f239ed",
        "kernel.py": "c85fe968399d2ac487dce6e5aaec5d267ee68a645a2abdb1725fbb5e824fa366",
        "workloads.json": "3ebce4a1cc51cefbdac825ee19a0cd4ab279f367c4e4098c5db415cca949d09e",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((TASK/relative).read_bytes()).hexdigest() == digest
