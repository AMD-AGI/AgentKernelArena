"""CPU replay controls for the mHC carrier; no native workload qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/aiter_workloads/deepseek_v4_pro_tp8_ep1_c64_8k1k_mhc"
TENSORS = ("layer_input", "residual_in", "post_layer_mix", "comb_res_mix",
           "fn", "hc_scale", "hc_base", "norm_weight")
OUTPUTS = ("post_mix", "comb_mix", "layer_input_out", "next_residual")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    torch = pytest.importorskip("torch", reason="mHC replay controls require CPU PyTorch")
    monkeypatch.setattr(sys, "path", list(sys.path))
    return SimpleNamespace(
        torch=torch,
        cases=_load("_mhc_replay_cases", TASK / "scripts/cases.py"),
        reference=_load("_mhc_replay_reference", TASK / "scripts/reference.py"),
        evaluator=_load("_mhc_replay_evaluator", TASK / "scripts/evaluate.py"),
        contract=json.loads((TASK / "workloads.json").read_text())["numerical_contract"],
    )


def _public(payload):
    return dict(zip(TENSORS, payload["args"] + (payload["kwargs"]["norm_weight"],)))


def _whole(torch, tensor):
    return torch.empty(0, dtype=torch.uint8).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes(),), (1,))


def _view(torch, backing, offset, shape, stride):
    return torch.empty(0, dtype=backing.dtype).set_(backing.untyped_storage(), offset, shape, stride)


def _payload(torch, contract, state):
    # Two dynamic alias groups, with offsets, holes and unused backing bytes.
    activation = torch.arange(80, dtype=torch.bfloat16) + state * 64
    mixes = torch.arange(96, dtype=torch.float32) + state * 64
    # Three prepared tensors share one immutable backing storage.
    prepared = torch.arange(512, dtype=torch.float32) / 64
    norm = torch.arange(12, dtype=torch.bfloat16) + 2
    tensors = {
        "layer_input": _view(torch, activation, 1, (2, 4), (32, 1)),
        "residual_in": _view(torch, activation, 6, (2, 4, 4), (32, 5, 1)),
        "post_layer_mix": _view(torch, mixes, 1, (2, 4), (40, 1)),
        "comb_res_mix": _view(torch, mixes, 8, (2, 4, 4), (40, 5, 1)),
        "fn": _view(torch, prepared, 4, (24, 16), (18, 1)),
        "hc_scale": _view(torch, prepared, 440, (3,), (1,)),
        "hc_base": _view(torch, prepared, 450, (24,), (1,)),
        "norm_weight": norm[2:6],
    }
    return {"args": tuple(tensors[name] for name in TENSORS[:7]),
            "kwargs": dict(contract["call_kwargs"], norm_weight=tensors["norm_weight"]),
            "captured": True}


def _metadata(payload):
    groups, result = {}, {}
    for name, value in _public(payload).items():
        identity = (value.device, value.untyped_storage()._cdata)
        result[name] = {
            "shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
            "storage_offset": value.storage_offset(), "storage_nbytes": value.untyped_storage().nbytes(),
            "storage_group": groups.setdefault(identity, len(groups)),
        }
    return result


@pytest.fixture
def replay_case(tmp_path, modules):
    torch = modules.torch
    payload = _payload(torch, modules.contract, 0)
    second = _public(_payload(torch, modules.contract, 1))

    def bind(name, tensors):
        path = tmp_path / name
        torch.save(tensors, path)
        return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    case = {
        "test_case_id": "cpu_mhc_control", "shape": [2, 4, 4], "dtype": "bfloat16",
        "checks": ["correctness", "performance"],
        "params": {"m": 2, "benchmark_method": "cuda_graph", "tensors": _metadata(payload)},
        "inputs": dict(bind("first.pt", _public(payload)), current_value_replay=bind("second.pt", second)),
    }
    return SimpleNamespace(root=tmp_path, payload=payload, second=second, case=case, bind=bind)


def test_copy_preserves_weights_aliases_and_all_public_objects(modules, replay_case):
    torch, cases, fixture = modules.torch, modules.cases, replay_case
    public = _public(fixture.payload)
    identities = {name: (id(value), value.untyped_storage()._cdata, value.data_ptr())
                  for name, value in public.items()}
    metadata = _metadata(fixture.payload)
    weights_before = cases.snapshot_inputs({"args": fixture.payload["args"][4:7],
                                            "kwargs": {"norm_weight": fixture.payload["kwargs"]["norm_weight"]}})
    second = cases.load_current_value_replay(fixture.root, fixture.case, fixture.payload)
    cases.copy_current_value_replay(fixture.payload, second)
    assert _metadata(fixture.payload) == metadata
    for name, value in _public(fixture.payload).items():
        assert (id(value), value.untyped_storage()._cdata, value.data_ptr()) == identities[name]
        assert torch.equal(_whole(torch, value), _whole(torch, second[name]))
    assert public["layer_input"].untyped_storage()._cdata == public["residual_in"].untyped_storage()._cdata
    assert public["fn"].untyped_storage()._cdata == public["hc_base"].untyped_storage()._cdata
    cases.check_inputs_unchanged(weights_before)


@pytest.mark.parametrize("fault", ["weight_value", "weight_slack", "last_metadata",
                                   "dynamic_alias_split", "prepared_alias_split", "dynamic_weight_alias"])
def test_copy_rejects_incompatible_state_before_any_write(modules, replay_case, fault):
    torch, cases, fixture = modules.torch, modules.cases, replay_case
    second = dict(fixture.second)
    if fault == "weight_value":
        second["norm_weight"][0] += 1
    elif fault == "weight_slack":
        visible = second["norm_weight"].clone()
        _whole(torch, second["norm_weight"])[0] ^= 1
        assert torch.equal(visible, second["norm_weight"])
    elif fault == "last_metadata":
        second["norm_weight"] = second["norm_weight"].clone()
    elif fault in {"dynamic_alias_split", "prepared_alias_split"}:
        name = "residual_in" if fault == "dynamic_alias_split" else "hc_base"
        old = second[name]
        backing = _whole(torch, old).clone()
        second[name] = torch.empty(0, dtype=old.dtype).set_(
            backing.untyped_storage(), old.storage_offset(), old.shape, old.stride())
    else:
        dynamic = fixture.payload["args"][0]
        fixture.payload["kwargs"]["norm_weight"] = _view(torch, dynamic, 2, (4,), (1,))
        second["norm_weight"] = _view(torch, second["layer_input"], 2, (4,), (1,))
    before = cases.snapshot_inputs(fixture.payload)
    with pytest.raises(ValueError):
        cases.copy_current_value_replay(fixture.payload, second)
    cases.check_inputs_unchanged(before)


@pytest.mark.parametrize("fault", ["missing", "hash", "prepared_bytes"])
def test_captured_replay_requires_authentication_and_original_weights(modules, replay_case, fault):
    fixture = replay_case
    if fault == "missing":
        fixture.case["inputs"].pop("current_value_replay")
    elif fault == "hash":
        (fixture.root / "second.pt").write_bytes(b"changed after binding")
    else:
        _whole(modules.torch, fixture.second["norm_weight"])[0] ^= 1
        fixture.case["inputs"]["current_value_replay"] = fixture.bind("second.pt", fixture.second)
    before = modules.cases.snapshot_inputs(fixture.payload)
    with pytest.raises(ValueError):
        modules.cases.load_current_value_replay(fixture.root, fixture.case, fixture.payload)
    modules.cases.check_inputs_unchanged(before)


@pytest.mark.parametrize("fault", [None, "missing_replay", "receipt_mismatch"])
def test_enrollment_binds_the_mandatory_replay_carrier(monkeypatch, modules, replay_case, fault):
    case = replay_case.case
    if fault == "missing_replay":
        case["inputs"].pop("current_value_replay")
    receipt = json.loads(json.dumps({"exported_cases": [dict(case, count=1)]}))
    if fault == "receipt_mismatch":
        receipt["exported_cases"][0]["inputs"]["current_value_replay"]["sha256"] = "0" * 64
    body = json.dumps(receipt).encode()
    (replay_case.root / "capture.json").write_bytes(body)
    provenance = {
        "status": "measured", "enrolled_for_scoring": True,
        "case_counts": [{"test_case_id": case["test_case_id"], "count": 1}],
        "capture": {"reference": "capture.json", "sha256": hashlib.sha256(body).hexdigest()},
    }
    monkeypatch.setattr(modules.evaluator, "ROOT", replay_case.root)
    if fault is None:
        modules.evaluator.check_enrollment(provenance, {case["test_case_id"]: 1}, [case])
    else:
        with pytest.raises(ValueError, match="current_value_replay|input binding"):
            modules.evaluator.check_enrollment(provenance, {case["test_case_id"]: 1}, [case])


def _outputs(torch):
    return (torch.ones((2, 4, 1), dtype=torch.float32), torch.ones((2, 4, 4), dtype=torch.float32),
            torch.ones((2, 4), dtype=torch.bfloat16), torch.ones((2, 4, 4), dtype=torch.bfloat16))


@pytest.mark.parametrize("changed", range(4))
def test_one_output_beyond_tolerance_suffices_to_separate_references(modules, changed):
    first = _outputs(modules.torch)
    second = tuple(value.clone() for value in first)
    second[changed].add_(1)
    assert modules.reference.check_separated(first, second, modules.contract) == {
        "old_oracle_rejected": True,
        "per_output_moved": {name: index == changed for index, name in enumerate(OUTPUTS)},
    }


@pytest.mark.parametrize("delta", [0.0, 0.005])
def test_oracle_pair_inside_existing_tolerance_cannot_qualify_replay(modules, delta):
    first = _outputs(modules.torch)
    second = tuple(value + delta for value in first)
    with pytest.raises(ValueError, match="cannot distinguish stale output"):
        modules.reference.check_separated(first, second, modules.contract)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("fault", ["short", "nonfinite", "noncontiguous", "geometry"])
def test_invalid_reference_cannot_be_accepted_as_separation(modules, side, fault):
    torch = modules.torch
    pair = [list(_outputs(torch)), list(_outputs(torch))]
    for value in pair[1]:
        value.add_(1)
    if fault == "short":
        pair[side].pop()
    elif fault == "nonfinite":
        pair[side][0][0, 0, 0] = float("nan")
    elif fault == "noncontiguous":
        pair[side][1] = pair[side][1].transpose(1, 2)
    else:
        pair[side][0] = torch.ones((2, 4, 2))
    with pytest.raises(ValueError):
        modules.reference.check_separated(tuple(pair[0]), tuple(pair[1]), modules.contract)


def _evaluate_fixture(monkeypatch, modules, fixture, *, stale_output=None, mutation=None):
    torch, evaluator = modules.torch, modules.evaluator
    timed_class = _load("_mhc_cpu_timed_run", ROOT / "src/tools/perf/aka_benchmark.py").TimedRun
    calls, replays = [], []

    def compute(layer_input, residual_in, post_layer_mix, comb_res_mix, *_prepared, **_kwargs):
        # Tiny independent outputs exercise the real four-output comparator;
        # this fixture does not claim native mHC numerical correctness.
        return (post_layer_mix.unsqueeze(-1).contiguous(), comb_res_mix.contiguous(),
                layer_input.contiguous(), residual_in.contiguous())

    def reference(payload, _contract):
        return compute(*payload["args"], **payload["kwargs"])

    def candidate(*args, **kwargs):
        calls.append(tuple(id(value) for value in args))
        return compute(*args, **kwargs)

    def benchmark(launch, *, timed_run, use_cuda_graph, **_kwargs):
        assert use_cuda_graph
        output = launch()
        timed_run.after_sample(output)
        captured_args, captured_kwargs = fixture.payload["args"], dict(fixture.payload["kwargs"])

        def rerun():
            replays.append(tuple(id(value) for value in captured_args))
            current = compute(*captured_args, **captured_kwargs)
            for index, value in enumerate(output):
                if index != stale_output:
                    value.copy_(current[index])
            if mutation == "replay_weight" and len(replays) == 2:
                _whole(torch, captured_args[4])[0] ^= 1
            return output

        timed_run._bind(rerun, output)
        if mutation == "sample_weight":
            _whole(torch, captured_args[4])[0] ^= 1
        return [0.1, 0.2, 0.3], {"benchmark_method": "cuda_graph", "benchmark_timed_run_kind": "captured_graph"}

    monkeypatch.setattr(evaluator, "ROOT", fixture.root)
    monkeypatch.setattr(evaluator, "candidate_callable", lambda: candidate)
    monkeypatch.setattr(modules.cases, "load_case", lambda *_args, **_kwargs: fixture.payload)
    monkeypatch.setitem(sys.modules, "cases", modules.cases)
    monkeypatch.setitem(sys.modules, "reference", SimpleNamespace(
        reference=reference, check_output=modules.reference.check_output,
        check_reference_controls=lambda *_args: {}, check_separated=modules.reference.check_separated))
    monkeypatch.setitem(sys.modules, "runtime", SimpleNamespace(
        verify_runtime=lambda: {}, verify_baseline_native=lambda: {}))
    monkeypatch.setitem(sys.modules, "_aka_benchmark", SimpleNamespace(
        TimedRun=timed_class, benchmark_cuda_graph_or_events_samples=benchmark))
    aiter, ops, mhc = (ModuleType(name) for name in ("aiter", "aiter.ops", "aiter.ops.mhc"))
    aiter.__path__, ops.__path__ = [], []
    aiter.ops, ops.mhc, mhc.mhc_fused_post_pre = ops, mhc, candidate
    for module in (aiter, ops, mhc):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    data = {"cases": [fixture.case], "numerical_contract": modules.contract,
            "timing": {"warmup": 1, "repetition": 3, "target_ms": 1.0, "max_graph_repeats": 8}}
    return SimpleNamespace(data=data, calls=calls, replays=replays)


@pytest.mark.parametrize("role", ["baseline", "candidate"])
def test_all_four_outputs_use_the_same_timed_graph(monkeypatch, modules, replay_case, role):
    state = _evaluate_fixture(monkeypatch, modules, replay_case)
    rows, _ = modules.evaluator.run_action(role, "performance", state.data)
    assert rows[0]["status"] == "PASS"
    assert rows[0]["execution_time_ms"] == 0.2
    assert set(rows[0]["metadata"]["current_value_replay"]) == set(OUTPUTS)
    assert len(state.calls) == 1
    assert state.replays == state.calls * 2


@pytest.mark.parametrize("stale_output", range(4))
def test_each_stale_graph_output_is_rejected(monkeypatch, modules, replay_case, stale_output):
    state = _evaluate_fixture(monkeypatch, modules, replay_case, stale_output=stale_output)
    with pytest.raises(AssertionError, match=OUTPUTS[stale_output]):
        modules.evaluator.run_action("candidate", "performance", state.data)
    assert len(state.calls) == 1 and len(state.replays) == 2


@pytest.mark.parametrize("mutation", ["sample_weight", "replay_weight"])
def test_weight_bytes_are_checked_before_and_after_state_replacement(monkeypatch, modules, replay_case, mutation):
    state = _evaluate_fixture(monkeypatch, modules, replay_case, mutation=mutation)
    with pytest.raises(AssertionError, match="backing storage changed"):
        modules.evaluator.run_action("candidate", "performance", state.data)
    assert len(state.calls) == 1
    assert len(state.replays) == (1 if mutation == "sample_weight" else 2)


def test_captured_correctness_reuses_original_public_objects(monkeypatch, modules, replay_case):
    state = _evaluate_fixture(monkeypatch, modules, replay_case)
    rows, _ = modules.evaluator.run_action("candidate", "correctness", state.data)
    assert rows[0]["status"] == "PASS"
    assert set(rows[0]["metadata"]["current_value_replay"]) == set(OUTPUTS)
    assert len(state.calls) == 2 and state.calls[0] == state.calls[1]
    assert state.replays == []


@pytest.mark.parametrize("separates", [False, True])
def test_captured_task_qualification_checks_separation_without_candidate_execution(
        monkeypatch, modules, replay_case, separates):
    if not separates:
        replay_case.case["inputs"]["current_value_replay"] = replay_case.bind(
            "second.pt", _public(replay_case.payload))
    state = _evaluate_fixture(monkeypatch, modules, replay_case)
    source = replay_case.root / "source"
    source.mkdir()
    (source / "kernel.py").write_text(
        "def mhc_fused_post_pre(*args, **kwargs):\n    return delegated(*args, **kwargs)\n")
    if separates:
        rows, metadata = modules.evaluator.run_action("task", "validate-task", state.data)
        assert rows[0]["status"] == "PASS"
        controls = metadata["reference_controls"][replay_case.case["test_case_id"]]
        assert controls["current_value_replay"]["old_oracle_rejected"] is True
        assert controls["current_value_replay"]["per_output_moved"] == dict.fromkeys(OUTPUTS, True)
    else:
        with pytest.raises(ValueError, match="cannot distinguish stale output"):
            modules.evaluator.run_action("task", "validate-task", state.data)
    assert state.calls == state.replays == []


@pytest.mark.parametrize("action", ["correctness", "performance"])
def test_synthetic_preflight_remains_unscored(monkeypatch, modules, replay_case, action):
    replay_case.case.pop("inputs")
    replay_case.payload["captured"] = False
    state = _evaluate_fixture(monkeypatch, modules, replay_case)
    if action == "performance":
        with pytest.raises(ValueError, match="Synthetic inputs are never timed"):
            modules.evaluator.run_action("candidate", action, state.data)
        assert state.calls == []
    else:
        rows, _ = modules.evaluator.run_action("candidate", action, state.data)
        assert rows[0]["status"] == "PASS"
        assert "execution_time_ms" not in rows[0]
        assert len(state.calls) == 1
    assert state.replays == []
