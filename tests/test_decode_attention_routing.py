"""CPU routing/collector evidence; actual Triton/FlyDSL GPU validation is separate."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import types

import pytest
import torch

from src.tools.perf.aka_benchmark import TimedRun

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/triton2flydsl/sglang/decode_attention"


def module(path):
    spec = importlib.util.spec_from_file_location("decode_routing_checks", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def harness(monkeypatch):
    checks = module(TASK / "scripts/replay_checks.py")
    ns = {name: getattr(checks, name) for name in (
        "require_tensor_contract", "require_unchanged", "verify_timed_run")}
    ns.update(TimedRun=TimedRun, _retry_oom=lambda fn: fn())
    path = TASK / "test_kernel_harness.py"
    tree = ast.parse(path.read_text())
    functions = {"make_inputs", "reference", "_shape_of", "_compare_decode_output",
                 "_reroute_kv_indices_", "_check_decode_routing", "run_correctness", "run_performance"}
    constants = {"TEST_SHAPES", "MAX_KV_SPLITS", "WARMUP_ITERATIONS", "BENCHMARK_ITERATIONS"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions
             or isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in constants]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield ns
    finally:
        torch.set_num_threads(previous_threads)


def initial_host_with_cpu_stages(behavior="correct", phase=None):
    """Actual initial host dispatcher/wrappers plus independent CPU stage doubles.

    The two GPU stages are modeled in FP64 with split softmax partials/LSE and
    an LSE-weighted combine. This tests the public routing ABI and GQA mapping;
    it does not claim to execute the original Triton kernels on CPU.
    """
    phase = phase if phase is not None else {"name": "correctness"}
    branches = []

    def stage1(q, k, v, partials, lse, indptr, indices, splits, maximum,
               scale, logit_cap, temperature, **kwargs):
        assert logit_cap == 0 and temperature == -1
        use_indices = indices
        if behavior == "ignore_indices" or behavior == "ignore_replay" and phase["name"] == "replay":
            use_indices = torch.arange(indices.numel(), dtype=indices.dtype)
        group = q.shape[1] // v.shape[1]
        partials.zero_()
        lse.fill_(float("-inf"))
        for batch in range(q.shape[0]):
            start, end = int(indptr[batch]), int(indptr[batch + 1])
            length = end - start
            # Original kernels align each split's length to MIN_BLOCK_KV=32.
            width = math.ceil(math.ceil(length / int(splits[batch])) / 32) * 32
            for split in range(int(splits[batch])):
                lo, hi = start + split * width, min(start + (split + 1) * width, end)
                if lo >= hi:
                    continue
                selected = use_indices[lo:hi].long()
                for head in range(q.shape[1]):
                    kv_head = head // group
                    scores = k[selected, kv_head].double().mv(q[batch, head].double()) * scale
                    largest = scores.max()
                    weights = (scores - largest).exp()
                    partials[batch, head, split] = weights @ v[selected, kv_head].double() / weights.sum()
                    lse[batch, head, split] = largest + weights.sum().log()

    def combine(partials, lse, q, out, scale, v, indptr, splits, maximum, sinks, **kwargs):
        assert sinks is None
        for batch in range(q.shape[0]):
            for head in range(q.shape[1]):
                values = lse[batch, head, :int(splits[batch])].double()
                valid = values.isfinite()
                weights = (values[valid] - values[valid].max()).exp()
                out[batch, head] = scale * (weights @ partials[batch, head, :int(splits[batch])][valid].double()) / weights.sum()
        if behavior == "mutate_replay" and phase["name"] == "replay":
            q.add_(1)
        if behavior == "raise_replay" and phase["name"] == "replay":
            q.add_(1)
            raise RuntimeError("injected replay failure")
        if behavior == "wrong_measured" and phase["name"] == "measured":
            out.add_(30)

    def normal(*args, **kwargs):
        branches.append("normal")
        stage1(*args, **kwargs)

    def grouped(*args, **kwargs):
        branches.append("grouped")
        stage1(*args, **kwargs)

    source = TASK / "decode_attention.py"
    names = {"decode_attention_fwd", "decode_attention_fwd_normal", "decode_attention_fwd_grouped"}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"_decode_att_m_fwd": normal, "_decode_grouped_att_m_fwd": grouped,
          "_decode_softmax_reducev_fwd": combine}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    return types.SimpleNamespace(decode_attention_fwd=ns["decode_attention_fwd"], branches=branches)


@pytest.mark.parametrize("case", range(6))
def test_every_original_case_has_oracle_sensitive_routing_including_b1(harness, case):
    cfg = harness["TEST_SHAPES"][case]
    torch.manual_seed(42 + case)
    tensors = harness["make_inputs"](cfg, "cpu")
    q, k, v, o, indptr, indices, partials, lse, splits = tensors
    pointer = indices.data_ptr()
    original = indices.clone()
    expected_identity = harness["reference"](q, k, v, indices, cfg)
    harness["_reroute_kv_indices_"](indices)
    assert indices.data_ptr() == pointer and indices.shape == original.shape
    assert indices.min() >= 0 and indices.max() < k.shape[0]
    assert len(indices.unique()) < len(indices) and not torch.equal(indices, original)
    expected_routed = harness["reference"](q, k, v, indices, cfg)
    with pytest.raises(AssertionError, match="Numerical mismatch"):
        harness["_compare_decode_output"](expected_identity.bfloat16(), expected_routed)
    # q/v replay perturbation must not erase sensitivity to ignored indices.
    q.neg_(); v.neg_()
    expected_replay = harness["reference"](q, k, v, indices, cfg)
    ignored_replay = harness["reference"](q, k, v, original, cfg)
    with pytest.raises(AssertionError, match="Numerical mismatch"):
        harness["_compare_decode_output"](ignored_replay.bfloat16(), expected_replay)


@pytest.mark.parametrize("heads,kv_heads", [(2, 2), (4, 2), (4, 1)])
def test_known_answer_and_actual_initial_host_routing(harness, heads, kv_heads):
    cfg = dict(seqs=[4], head=heads, kv_head=kv_heads, Lk=4, Lv=2)
    tensors = harness["make_inputs"](cfg, "cpu")
    q, k, v, out, _, indices, *_ = tensors
    q.zero_(); k.zero_()
    for slot in range(4):
        for head in range(kv_heads):
            v[slot, head] = torch.tensor([slot * 4 + head, slot * 4 + head + 2])
    # New routing selects [3, 1, 3, 1]; its mean differs from [0, 1, 2, 3].
    expected = torch.tensor([[[8 + h // (heads // kv_heads), 10 + h // (heads // kv_heads)]
                              for h in range(heads)]], dtype=torch.float32)
    harness["_reroute_kv_indices_"](indices)
    assert indices.tolist() == [3, 1, 3, 1]
    assert torch.equal(harness["reference"](q, k, v, indices, cfg), expected)
    baseline = initial_host_with_cpu_stages()
    baseline.decode_attention_fwd(*tensors, 8, .5, 1., 1.)
    harness["_compare_decode_output"](out, expected)
    assert baseline.branches == ["normal" if heads == kv_heads else "grouped"]
    bad = initial_host_with_cpu_stages("ignore_indices")
    bad.decode_attention_fwd(*tensors, 8, .5, 1., 1.)
    with pytest.raises(AssertionError, match="Numerical mismatch"):
        harness["_compare_decode_output"](out, expected)


@pytest.mark.parametrize("behavior", ["correct", "ignore_indices"])
@pytest.mark.parametrize("heads,kv_heads", [(2, 2), (4, 2), (4, 1)])
def test_actual_correctness_runs_candidate_routing_control(harness, behavior, heads, kv_heads):
    cfg = dict(seqs=[65], head=heads, kv_head=kv_heads, Lk=4, Lv=4)
    torch.manual_seed(4)
    tensors = harness["make_inputs"](cfg, "cpu")
    readonly = tuple(tensors[i] for i in (0, 1, 2, 4, 5, 8))
    originals = tuple(x.clone() for x in readonly)
    candidate = initial_host_with_cpu_stages(behavior)
    harness.update(TEST_SHAPES=[cfg], make_inputs=lambda *args: tensors, load_module=lambda: candidate)
    ok, error, details = harness["run_correctness"]()
    assert ok is (behavior == "correct"), error
    assert len(details) == 1
    if ok:
        assert details[0]["kv_routing_control"]["status"] == "PASS"
    else:
        assert "Numerical mismatch" in error
    harness["require_unchanged"](readonly, originals)


def test_all_six_original_shapes_accept_initial_host_with_independent_cpu_stages(harness):
    make_inputs = harness["make_inputs"]
    candidate = initial_host_with_cpu_stages()
    harness.update(make_inputs=lambda cfg, device: make_inputs(cfg, "cpu"),
                   load_module=lambda: candidate)
    ok, error, details = harness["run_correctness"]()
    assert ok, error
    assert len(details) == 6
    assert all(row["kv_routing_control"]["status"] == "PASS" for row in details)
    assert candidate.branches.count("normal") == 4
    assert candidate.branches.count("grouped") == 8


@pytest.mark.parametrize("failure", ["candidate_mutation", "candidate_exception", "reference_exception"])
def test_routing_correctness_restores_inputs_on_failure(harness, failure):
    cfg = dict(seqs=[4], head=2, kv_head=1, Lk=4, Lv=2)
    tensors = harness["make_inputs"](cfg, "cpu")
    readonly = tuple(tensors[i] for i in (0, 1, 2, 4, 5, 8))
    originals = tuple(x.clone() for x in readonly)
    baseline = initial_host_with_cpu_stages()

    def candidate(*args):
        baseline.decode_attention_fwd(*args)
        tensors[0].add_(1)
        if failure == "candidate_exception":
            raise RuntimeError("injected candidate exception")

    if failure == "reference_exception":
        def broken_reference(*args):
            raise RuntimeError("injected reference exception")
        harness["reference"] = broken_reference
    with pytest.raises((AssertionError, RuntimeError), match="read-only input|injected .* exception"):
        harness["_check_decode_routing"](types.SimpleNamespace(decode_attention_fwd=candidate),
                                          tensors, cfg, .5)
    harness["require_unchanged"](readonly, originals)


@pytest.mark.parametrize("behavior", ["correct", "ignore_indices", "ignore_replay", "mutate_replay", "raise_replay", "wrong_measured"])
@pytest.mark.parametrize("method", ["captured_graph", "eager_event"])
def test_actual_performance_uses_same_collector_and_restores_inputs(harness, behavior, method):
    cfg = dict(seqs=[65], head=4, kv_head=2, Lk=4, Lv=4)
    torch.manual_seed(3)
    tensors = harness["make_inputs"](cfg, "cpu")
    readonly = tuple(tensors[i] for i in (0, 1, 2, 4, 5, 8))
    originals = tuple(x.clone() for x in readonly)
    original_pointers = tuple(x.data_ptr() for x in tensors)
    phase = {"name": "setup"}
    candidate = initial_host_with_cpu_stages(behavior, phase)
    harness.update(TEST_SHAPES=[cfg], make_inputs=lambda *args: tensors, load_module=lambda: candidate)
    observations = []

    def benchmark(fn, *, warmup, repetition, timed_run):
        assert (warmup, repetition) == (0, 100)
        # Original retry + ten explicit warmup invocations used identity routing.
        assert len(candidate.branches) == 11
        harness["require_unchanged"](readonly, originals)
        assert tuple(x.data_ptr() for x in tensors) == original_pointers
        phase["name"] = "measured"
        measured = fn()
        def replay():
            phase["name"] = "replay"
            assert tuple(x.data_ptr() for x in tensors) == original_pointers
            assert not torch.equal(tensors[5], originals[4])
            assert bool(torch.isnan(measured).all())
            observations.append(tensors[5].clone())
            return fn()
        timed_run._bind(replay, measured)
        return .1, {"benchmark_method": "cuda_graph" if method == "captured_graph" else "cuda_event_fallback",
                    "benchmark_timed_run_kind": method}

    harness["benchmark_cuda_graph_or_events"] = benchmark
    rows = harness["run_performance"]()
    assert len(rows) == 1
    if behavior == "correct":
        assert rows[0]["execution_time_ms"] == .1
        assert rows[0]["replay_correctness"] == "PASS"
        assert rows[0]["replay_kv_routing"] == "repeated_noncontiguous_kv"
    else:
        assert rows[0]["execution_time_ms"] < 0
    assert len(observations) == (0 if behavior == "wrong_measured" else 1)
    harness["require_unchanged"](readonly, originals)


def test_measured_readonly_failure_also_restores_all_inputs(harness):
    value = torch.tensor([1.])
    timed = TimedRun(); timed._bind(lambda: torch.tensor([1.]), torch.tensor([1.]))
    value.add_(1)
    with pytest.raises(AssertionError, match="read-only input"):
        harness["verify_timed_run"](timed, inputs=(value,), originals=(torch.tensor([1.]),),
                                    expected=torch.tensor([1.]), perturb=lambda: None,
                                    reference=lambda: torch.tensor([1.]), compare=lambda a, b: None)
    assert value.item() == 1


def test_original_six_case_manifest_initial_source_and_numeric_gate_unchanged(harness):
    # Exact base 360febda bytes, separate from migration AST fingerprints.
    expected = {
        "decode_attention.py": "a23e5d2ae1ac258762b1c6c905e14be1f18f12f14d0f007fa71aab37e5fe0296",
        "config.yaml": "fe1212e1709687d57a05a1dfb92f82cc537fc9adb94cb4048f195ab8533ff7c2",
        "cases.json": "317e42740234bf339963e72efc86b13c8123429c93b36c7a0889b5a54b0304c2",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((TASK / name).read_bytes()).hexdigest() == digest
    functions = {node.name: node for node in ast.parse((TASK / "test_kernel_harness.py").read_text()).body
                 if isinstance(node, ast.FunctionDef)}
    for name, digest in {
        "make_inputs": "ff972c7d9407f064832cb3723d992e01698229636b0e3a240e21c14f4ecac8d0",
        "reference": "53ed31ba17e871c851920e07c2bbc979756119be1d98a1c1a297c869a0443e07",
        "_shape_of": "8e39dc68c2adf8bc2e271904456679a6252f0156f78dbfacb3e455dd44478c60",
        "_compare_decode_output": "6cd549af2b28b8af1cdbde87ab8bdff1bf776fd840bbed26e7373d7ffa6917d8",
    }.items():
        assert hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest() == digest
    cases = json.loads((TASK / "cases.json").read_text())["cases"]
    assert len(cases) == 6
    for i, case in enumerate(cases):
        assert case["test_case_id"] == f"case_{i:04d}"
        assert case["params"] == dict(harness["TEST_SHAPES"][i], seed=42+i)
        assert case["checks"] == ["correctness", "performance"]
    assert (harness["WARMUP_ITERATIONS"], harness["BENCHMARK_ITERATIONS"], harness["MAX_KV_SPLITS"]) == (10, 100, 8)


def test_original_correctness_gate_and_complete_timing_body_preserved():
    """Remove exactly the new unscored checks, then compare the original AST."""
    def dump(node):
        return ast.dump(node, include_attributes=False)

    added_statements = {dump(ast.parse(source).body[0]) for source in (
        "routing = _check_decode_routing(mod, (q, k_buf, v_buf, o, kvp, kvi, al, alse, nks), cfg, sm_scale) if passed else None",
        'bench_meta["replay_kv_routing"] = "repeated_noncontiguous_kv"',
    )}
    added_perturb = ast.parse("lambda: (q.neg_(), v_buf.neg_(), _reroute_kv_indices_(kvi))", mode="eval").body
    original_perturb = ast.parse("lambda: (q.neg_(), v_buf.neg_())", mode="eval").body

    class RemoveRoutingOnly(ast.NodeTransformer):
        def visit_Assign(self, node):
            return None if dump(node) in added_statements else self.generic_visit(node)

        def visit_Dict(self, node):
            pairs = [(k, v) for k, v in zip(node.keys, node.values)
                     if not (isinstance(k, ast.Constant) and k.value == "kv_routing_control"
                             and isinstance(v, ast.Name) and v.id == "routing")]
            node.keys, node.values = [k for k, v in pairs], [v for k, v in pairs]
            return self.generic_visit(node)

        def visit_Lambda(self, node):
            return original_perturb if dump(node) == dump(added_perturb) else self.generic_visit(node)

    originals = {
        "run_correctness": "46bbfe2e7cb8de3fae5670bd69f908e99bbf08be6765ec8bac52fef19d03141b",
        "run_performance": "1e09f2498cb8355f2e39a606fbe56465a4ba8bc6d6a21c72c8cd8e365970f25b",
    }
    for node in ast.parse((TASK / "test_kernel_harness.py").read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in originals:
            assert hashlib.sha256(dump(RemoveRoutingOnly().visit(node)).encode()).hexdigest() == originals[node.name]
