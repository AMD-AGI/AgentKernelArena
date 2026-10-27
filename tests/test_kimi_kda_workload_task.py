"""CPU checks of the Kimi task's state oracle and scored-sample checks."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest

torch = pytest.importorskip("torch")
TASK = Path(__file__).resolve().parents[1] / "tasks/aiter_workloads/kimi_k3_tp8_c64_8k1k_kda_decode"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(TASK / "scripts"))
    loaded = {}
    for name in ("cases", "kda_ref", "reference", "evaluate"):
        spec = importlib.util.spec_from_file_location(name, TASK / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def small_case(padding=0):
    return {"shape": [2, 1, 2, 4], "params": {"state_padding_floats": padding}}


def oracle_call(reference, payload):
    """A CPU known-correct implementation of the public observable boundary."""
    output, state = reference.expected(payload)
    kw = payload["kwargs"]
    kw["out"].copy_(output)
    kw["initial_state"][kw["ssm_state_indices"].long()] = state
    return kw["out"], kw["initial_state"]


def test_manifest_is_checkpoint_derived(modules):
    data = modules["evaluate"].read_contract()
    assert [case["shape"] for case in data["cases"]] == [[64, 1, 12, 128]] * 2
    assert data["workload"]["ep"] is None
    assert data["workload"]["dcp"] is None
    modules["evaluate"].check_candidate_source()


def test_reference_known_scalar_transition(modules):
    # Decay 4 -> 2; residual 3 - 2*2 = -1; update 2 + 2*.25*(-1) = 1.5.
    shape = (1, 1, 1, 1)
    out, state = modules["kda_ref"].chunk_kda_ref(
        q=torch.ones(shape), k=torch.full(shape, 2.0), v=torch.full(shape, 3.0),
        g=torch.full(shape, 0.5).log(), beta=torch.full((1, 1, 1), 0.25),
        initial_state=torch.full(shape, 4.0), output_final_state=True,
    )
    torch.testing.assert_close(out, torch.full(shape, 1.5), rtol=0, atol=1e-7)
    torch.testing.assert_close(state, torch.full(shape, 1.5), rtol=0, atol=1e-7)


@pytest.mark.parametrize("padding", [0, 4])
def test_oracle_accepts_recurrence_and_consecutive_state(modules, padding):
    cases, ref = modules["cases"], modules["reference"]
    payload = cases.make_case(small_case(padding), device="cpu")
    for seed in (42, 71):
        cases.refresh_operands(payload, seed)
        before, expected = cases.snapshot(payload), ref.expected(payload)
        actual = oracle_call(ref, payload)
        assert ref.check_result(payload, actual, expected, before)["untouched_state_bytes_equal"]


@pytest.mark.parametrize("corruption", ["output", "active_state", "unused_slot", "padding", "input", "alias"])
def test_oracle_rejects_wrong_result_or_state(modules, corruption):
    cases, ref = modules["cases"], modules["reference"]
    payload = cases.make_case(small_case(4), device="cpu")
    before, expected = cases.snapshot(payload), ref.expected(payload)
    actual = oracle_call(ref, payload)
    kw = payload["kwargs"]
    if corruption == "output":
        kw["out"].zero_()
    elif corruption == "active_state":
        kw["initial_state"][kw["ssm_state_indices"].long()] = 0
    elif corruption == "unused_slot":
        slot = int(payload["unused"].nonzero()[0].item())
        payload["raw"].view(payload["slots"], payload["stride"])[slot, 0] = 0
    elif corruption == "padding":
        payload["raw"][payload["plane"]] = 0
    elif corruption == "input":
        kw["q"].add_(1)
    else:
        actual = (kw["out"].clone(), kw["initial_state"])
    with pytest.raises(AssertionError):
        ref.check_result(payload, actual, expected, before)


def test_prepare_restores_whole_pool_and_poison_output(modules):
    cases = modules["cases"]
    payload = cases.make_case(small_case(4), device="cpu")
    payload["raw"].zero_()
    payload["kwargs"]["out"].zero_()
    cases.prepare(payload)
    assert torch.equal(payload["raw"], payload["pristine"])
    assert torch.isnan(payload["kwargs"]["out"]).all()


@pytest.mark.parametrize("wrong_timed_state", [False, True])
def test_measure_checks_every_sample_before_reset(modules, monkeypatch, wrong_timed_state):
    cases, ref, evaluator = (modules[name] for name in ("cases", "reference", "evaluate"))
    original_make = cases.make_case
    holder = {}

    def make_cpu(case, seed):
        holder["payload"] = original_make(case, seed, device="cpu")
        return holder["payload"]

    monkeypatch.setattr(cases, "make_case", make_cpu)
    observations = []

    class TimedRun:
        after_sample = None

        def rerun(self):
            self.prepare()
            return self.fn()

    def fake_benchmark(fn, *, warmup, repetition, target_ms, max_graph_repeats,
                       use_cuda_graph, prepare_fn, timed_run):
        assert use_cuda_graph and max_graph_repeats == 1 and prepare_fn is not None
        timed_run.fn, timed_run.prepare = fn, prepare_fn
        for _ in range(repetition):
            prepare_fn()
            payload = holder["payload"]
            assert torch.equal(payload["raw"], payload["pristine"])
            result = fn()
            if wrong_timed_state:
                slot = int(payload["unused"].nonzero()[0].item())
                payload["raw"].view(payload["slots"], payload["stride"])[slot, 0] = 0
            timed_run.after_sample(result)
            observations.append(True)
        return [1.0] * repetition, {"benchmark_method": "cuda_graph", "benchmark_effective_repeats": 1}

    monkeypatch.setitem(sys.modules, "_aka_benchmark", types.SimpleNamespace(
        TimedRun=TimedRun, benchmark_cuda_graph_or_events_samples=fake_benchmark))

    def call(**kwargs):
        return oracle_call(ref, holder["payload"])

    timing = {"warmup": 1, "repetition": 3, "target_ms": 1.0}
    if wrong_timed_state:
        with pytest.raises(AssertionError, match="unselected state"):
            evaluator.measure_case(call, small_case(4), timing, 0.005)
    else:
        _, _, metrics = evaluator.measure_case(call, small_case(4), timing, 0.005)
        assert len(observations) == metrics["checked_timed_samples"] == 3
        assert metrics["fresh_value_graph_replay"]["read_only_bytes_equal"]
