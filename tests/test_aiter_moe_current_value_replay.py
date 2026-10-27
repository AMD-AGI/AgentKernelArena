"""CPU controls for authenticated current-value replay; no GPU qualification."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/aiter_workloads/deepseek_v4_pro_tp8_ep1_c64_8k1k_fused_moe"
MUTABLE = ("hidden_states", "topk_weight", "topk_ids")
POSITIONAL = ("hidden_states", "w1", "w2", "topk_weight", "topk_ids")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    torch = pytest.importorskip("torch", reason="Replay storage tests require CPU PyTorch")
    monkeypatch.setattr(sys, "path", list(sys.path))
    reference = SimpleNamespace(validate_contract=lambda _: None)
    monkeypatch.setitem(sys.modules, "reference", reference)
    loader = _load("_moe_current_value_cases", TASK / "scripts/cases.py")
    evaluator = _load("_moe_current_value_evaluator", TASK / "scripts/evaluate.py")
    return SimpleNamespace(torch=torch, loader=loader, evaluator=evaluator, reference=reference)


def _public(torch, payload):
    return dict(zip(POSITIONAL, payload["args"])) | {
        name: value for name, value in payload["kwargs"].items()
        if isinstance(value, torch.Tensor)
    }


def _whole(torch, tensor):
    return torch.empty(0, dtype=torch.uint8).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes(),), (1,))


def _payload(torch, state):
    # Nonzero offset, holes between rows, and bytes outside every public view.
    hidden_backing = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8) + state * 8
    hidden = hidden_backing[1:3, 2:6]
    weight_backing = torch.full((56,), 17 + state, dtype=torch.uint8)
    id_backing = torch.full((56,), 17 + state, dtype=torch.uint8)
    weights = torch.empty(0, dtype=torch.float32).set_(weight_backing.untyped_storage(), 1, (2, 2), (6, 1))
    ids = torch.empty(0, dtype=torch.int32).set_(id_backing.untyped_storage(), 3, (2, 2), (6, 1))
    weights.copy_(torch.tensor([[0.1 + state * 0.2, 0.2], [0.3, 0.4]]))
    ids.copy_(torch.tensor([[state % 3, (state + 1) % 3], [(state + 1) % 3, (state + 2) % 3]]))
    w1, w2 = torch.zeros((3, 4, 2), dtype=torch.uint8), torch.zeros((3, 4, 1), dtype=torch.uint8)
    w1.is_shuffled = w2.is_shuffled = True
    return {
        "args": (hidden, w1, w2, weights, ids),
        "kwargs": {"w1_scale": torch.ones(4, dtype=torch.uint8),
                   "w2_scale": torch.ones(4, dtype=torch.uint8),
                   "bias1": torch.arange(4, dtype=torch.float32), "doweight_stage1": False},
        "reference": {"w1": w1.clone(), "w2": w2.clone()},
    }


def _share_routing_storage(payload):
    weights, ids = payload["args"][3:5]
    old_ids = ids.clone()
    ids.set_(weights.untyped_storage(), ids.storage_offset(), ids.shape, ids.stride())
    ids.copy_(old_ids)


def _metadata(torch, payload):
    groups, metadata = {}, {}
    for name, value in sorted(_public(torch, payload).items()):
        identity = (value.device, value.untyped_storage()._cdata)
        group = groups.setdefault(identity, len(groups))
        metadata[name] = {
            "shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
            "storage_offset": value.storage_offset(), "storage_nbytes": value.untyped_storage().nbytes(),
            "storage_group": group,
        }
    return metadata


@pytest.fixture
def replay_case(tmp_path, modules):
    torch = modules.torch
    payload = _payload(torch, 0)
    next_public = _public(torch, _payload(torch, 1))
    replacements = {name: next_public[name] for name in MUTABLE}
    path = tmp_path / "next.pt"

    def bind(tensors):
        torch.save(tensors, path)
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    case = {
        "test_case_id": "cpu_storage_control", "checks": ["correctness", "performance"],
        "shape": [2, 4, 2, 3, 2], "dtype": "bfloat16",
        "params": {"benchmark_method": "cuda_graph", "tensors": _metadata(torch, payload)},
        "inputs": {"current_value_replay": bind(replacements)},
    }
    return SimpleNamespace(root=tmp_path, path=path, payload=payload, replacements=replacements,
                           case=case, bind=bind)


def test_replay_copies_complete_storage_and_preserves_public_objects(modules, replay_case):
    torch, loader, fixture = modules.torch, modules.loader, replay_case
    public = _public(torch, fixture.payload)
    original_args, original_kwargs = fixture.payload["args"], dict(fixture.payload["kwargs"])
    identities = {name: (id(value), value.untyped_storage()._cdata, value.data_ptr())
                  for name, value in public.items()}
    original_metadata = _metadata(torch, fixture.payload)
    protected = loader.snapshot_inputs({"args": original_args[1:3], "kwargs": original_kwargs})
    before = {name: _whole(torch, public[name]).clone() for name in MUTABLE}
    replacements = loader.load_current_value_replay(fixture.root, fixture.case, fixture.payload)

    loader.copy_current_value_replay(fixture.payload, replacements)

    assert fixture.payload["args"] is original_args
    assert all(fixture.payload["kwargs"][name] is value for name, value in original_kwargs.items())
    assert _metadata(torch, fixture.payload) == original_metadata
    for name, value in public.items():
        assert (id(value), value.untyped_storage()._cdata, value.data_ptr()) == identities[name]
    for name in MUTABLE:
        assert not torch.equal(before[name], _whole(torch, public[name]))
        assert torch.equal(_whole(torch, public[name]), _whole(torch, replacements[name]))
    loader.check_inputs_unchanged(protected)


def test_raw_copy_preserves_mixed_dtype_storage_aliases(modules):
    # torch.save may reject aliases across dtypes; exercise the raw copier
    # without claiming that a serialized carrier supports that arrangement.
    torch, loader = modules.torch, modules.loader
    payload, replacement_payload = _payload(torch, 0), _payload(torch, 1)
    _share_routing_storage(payload)
    _share_routing_storage(replacement_payload)
    public, replacement_public = _public(torch, payload), _public(torch, replacement_payload)
    replacements = {name: replacement_public[name] for name in MUTABLE}
    metadata = _metadata(torch, payload)
    storage_id = public["topk_weight"].untyped_storage()._cdata
    loader.copy_current_value_replay(payload, replacements)
    assert _metadata(torch, payload) == metadata
    assert public["topk_weight"].untyped_storage()._cdata == storage_id
    assert public["topk_ids"].untyped_storage()._cdata == storage_id
    for name in MUTABLE:
        assert torch.equal(_whole(torch, public[name]), _whole(torch, replacements[name]))


@pytest.mark.parametrize("fault", ["hash", "missing", "extra", "metadata", "aliases", "nonfinite", "expert_id"])
def test_replay_rejects_unauthenticated_or_incompatible_state(modules, replay_case, fault):
    torch, fixture = modules.torch, replay_case
    replacements = dict(fixture.replacements)
    if fault == "missing":
        replacements.pop("topk_ids")
    elif fault == "extra":
        replacements["w1"] = fixture.payload["args"][1]
    elif fault == "metadata":
        replacements["hidden_states"] = replacements["hidden_states"].contiguous()
    elif fault == "aliases":
        _share_routing_storage(fixture.payload)
        fixture.case["params"]["tensors"] = _metadata(torch, fixture.payload)
    elif fault == "nonfinite":
        replacements["hidden_states"][0, 0] = float("nan")
    elif fault == "expert_id":
        replacements["topk_ids"][0, 0] = 3
    fixture.case["inputs"]["current_value_replay"] = fixture.bind(replacements)
    if fault == "hash":
        fixture.path.write_bytes(b"changed after binding")
    before = modules.loader.snapshot_inputs(fixture.payload)
    with pytest.raises(ValueError):
        modules.loader.load_current_value_replay(fixture.root, fixture.case, fixture.payload)
    modules.loader.check_inputs_unchanged(before)


@pytest.mark.parametrize("fault", ["metadata", "alias_split", "alias_merge", "weight_alias"])
def test_copy_preflights_every_storage_before_writing(modules, replay_case, fault):
    torch, loader, fixture = modules.torch, modules.loader, replay_case
    replacements = dict(fixture.replacements)
    if fault == "metadata":
        # This is the last mutable operand, after the changed hidden-state source.
        replacements["topk_ids"] = replacements["topk_ids"].contiguous()
    elif fault == "alias_split":
        _share_routing_storage(fixture.payload)
    elif fault == "alias_merge":
        _share_routing_storage({"args": (None, None, None, replacements["topk_weight"], replacements["topk_ids"])})
    else:
        fixture.payload["kwargs"]["bias1"] = fixture.payload["args"][3].view_as(fixture.payload["args"][3])
    before = loader.snapshot_inputs(fixture.payload)
    with pytest.raises(ValueError):
        loader.copy_current_value_replay(fixture.payload, replacements)
    loader.check_inputs_unchanged(before)


def _evaluate_fixture(monkeypatch, modules, fixture, *, fault=None):
    torch, loader, evaluator = modules.torch, modules.loader, modules.evaluator
    timed_class = _load("_moe_cpu_timed_run", ROOT / "src/tools/perf/aka_benchmark.py").TimedRun
    calls, replays, checked = [], [], []
    native_verifications = []
    native_identity = {"test_stub": "baseline_native_identity"}

    def verify_baseline_native():
        native_verifications.append(None)
        return native_identity

    def compute(hidden, _w1, _w2, routing_weight, routing_ids, **_kwargs):
        return (hidden.float() + routing_weight.sum(dim=1, keepdim=True)
                + routing_ids.sum(dim=1, keepdim=True)).to(torch.bfloat16)

    def reference(payload, _contract):
        return compute(*payload["args"], **payload["kwargs"])

    def compare(actual, expected, _contract):
        return {"accepted": torch.equal(actual, expected)}

    def check(actual, expected, contract):
        result = compare(actual, expected, contract)
        assert result["accepted"], "stale or incorrect current-value output"
        checked.append(expected.clone())
        return result

    def candidate(*args, **kwargs):
        calls.append(tuple(id(value) for value in args))
        return compute(*args, **kwargs)

    def benchmark(launch, *, timed_run, use_cuda_graph, **_policy):
        method = fixture.case["params"]["benchmark_method"]
        assert use_cuda_graph is (method == "cuda_graph")
        output = launch()
        timed_run.after_sample(output)
        captured_args, captured_kwargs = fixture.payload["args"], dict(fixture.payload["kwargs"])

        def rerun():
            nonlocal output
            replays.append(tuple(id(value) for value in captured_args))
            if fault != "stale":
                if use_cuda_graph:
                    output.copy_(compute(*captured_args, **captured_kwargs))
                else:
                    output = launch()
            if fault == "replay_slack" and len(replays) == 2:
                _whole(torch, captured_args[0])[0] ^= 1
            if fault == "replay_weight" and len(replays) == 2:
                _whole(torch, captured_args[1])[-1] ^= 1
            return output

        timed_run._bind(rerun, output)
        if fault == "sample_slack":
            _whole(torch, captured_args[0])[0] ^= 1
        if fault == "sample_weight":
            _whole(torch, captured_args[1])[-1] ^= 1
        return [0.125, 0.25, 0.375], {
            "benchmark_method": method,
            "benchmark_timed_run_kind": "captured_graph" if use_cuda_graph else "eager_callable",
        }

    monkeypatch.setattr(evaluator, "ROOT", fixture.root)
    monkeypatch.setattr(evaluator, "candidate_callable", lambda: candidate)
    monkeypatch.setattr(loader, "load_case", lambda *_args, **_kwargs: fixture.payload)
    monkeypatch.setitem(sys.modules, "cases", loader)
    monkeypatch.setitem(sys.modules, "runtime", SimpleNamespace(
        verify_runtime=lambda: {}, verify_baseline_native=verify_baseline_native))
    monkeypatch.setitem(sys.modules, "reference", SimpleNamespace(
        reference=reference, compare_output=compare, check_output=check,
        check_reference_controls=lambda *_args: {}))
    monkeypatch.setitem(sys.modules, "_aka_benchmark", SimpleNamespace(
        TimedRun=timed_class, benchmark_cuda_graph_or_events_samples=benchmark))
    aiter, fused_moe = ModuleType("aiter"), ModuleType("aiter.fused_moe")
    aiter.__path__ = []
    fused_moe.fused_moe = candidate
    aiter.fused_moe = fused_moe
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fused_moe)
    data = {"cases": [fixture.case], "numerical_contract": {},
            "timing": {"warmup": 1, "repetition": 3, "target_ms": 1.0, "max_graph_repeats": 8}}
    return SimpleNamespace(data=data, calls=calls, replays=replays, checked=checked,
                           native_verifications=native_verifications, native_identity=native_identity)


@pytest.mark.parametrize("role", ["baseline", "candidate"])
@pytest.mark.parametrize("method", ["cuda_graph", "cuda_event_fallback"])
def test_performance_uses_same_timed_unit_with_new_values(monkeypatch, modules, replay_case, role, method):
    replay_case.case["params"]["benchmark_method"] = method
    state = _evaluate_fixture(monkeypatch, modules, replay_case)
    rows, metadata = modules.evaluator.run_action(role, "performance", state.data)
    assert rows[0]["status"] == "PASS"
    assert rows[0]["execution_time_ms"] == 0.25
    assert len(state.calls) == (1 if method == "cuda_graph" else 3)
    assert len(state.replays) == 2  # Original-state replay, then changed-state replay.
    assert state.replays == state.calls[:1] * 2
    assert not modules.torch.equal(state.checked[0], state.checked[-1])
    assert "current_value_replay" in rows[0]["metadata"]
    assert len(state.native_verifications) == (1 if role == "baseline" else 0)
    if role == "baseline":
        assert metadata["baseline_native_identity"] is state.native_identity
    else:
        assert "baseline_native_identity" not in metadata


@pytest.mark.parametrize("fault", ["stale", "sample_slack", "replay_slack", "sample_weight", "replay_weight"])
def test_performance_rejects_stale_output_and_mutation_in_either_state(monkeypatch, modules, replay_case, fault):
    state = _evaluate_fixture(monkeypatch, modules, replay_case, fault=fault)
    with pytest.raises(AssertionError, match="stale|backing storage changed"):
        modules.evaluator.run_action("candidate", "performance", state.data)
    assert len(state.calls) == 1
    assert len(state.replays) == (1 if fault in {"sample_slack", "sample_weight"} else 2)


def test_correctness_reuses_original_public_input_objects(monkeypatch, modules, replay_case):
    state = _evaluate_fixture(monkeypatch, modules, replay_case)
    rows, _ = modules.evaluator.run_action("candidate", "correctness", state.data)
    assert rows[0]["status"] == "PASS"
    assert len(state.calls) == 2 and state.calls[0] == state.calls[1]
    assert not modules.torch.equal(state.checked[0], state.checked[-1])


@pytest.mark.parametrize("distinguishable", [False, True])
def test_validate_task_requires_oracle_separation_before_candidate_execution(
        monkeypatch, modules, replay_case, distinguishable):
    if not distinguishable:
        # Distinct routing values can still compute the same answer. Merely
        # observing changed bytes cannot establish stale-output sensitivity.
        second = _public(modules.torch, _payload(modules.torch, 0))
        second["topk_ids"].copy_(second["topk_ids"].flip(1))
        second["topk_weight"].copy_(second["topk_weight"].flip(1))
        replay_case.case["inputs"]["current_value_replay"] = replay_case.bind(
            {name: second[name] for name in MUTABLE})
    state = _evaluate_fixture(monkeypatch, modules, replay_case)
    source = replay_case.root / "source"
    source.mkdir()
    (source / "kernel.py").write_text("def fused_moe(*args, **kwargs):\n    return delegated(*args, **kwargs)\n")
    if distinguishable:
        rows, metadata = modules.evaluator.run_action("task", "validate-task", state.data)
        assert rows[0]["status"] == "PASS"
        control = metadata["reference_controls"][replay_case.case["test_case_id"]]["current_value_replay"]
        assert control["accepted"] is False
    else:
        with pytest.raises(ValueError, match="cannot distinguish stale output"):
            modules.evaluator.run_action("task", "validate-task", state.data)
    assert state.calls == state.replays == []
