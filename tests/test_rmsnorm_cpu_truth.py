"""Numerical checks for evaluator-owned BF16 inputs and the CPU-only oracle."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).parents[1]
TASK = ROOT / "tasks/headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


inputs = load("rmsnorm_cpu_inputs_test", TASK / "scripts/_rmsnorm_inputs.py")
cases = load("rmsnorm_cpu_cases_test", TASK / "ut/cases.py")
harness = load("rmsnorm_cpu_harness_test", TASK / "ut/harness_lib.py")
META = json.loads((TASK / "ut/meta.json").read_text())


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(min(prior, 4))
    yield
    torch.set_num_threads(prior)


def make_args(spec):
    result = {name: torch.empty_strided(spec[name + "_shape"], spec[name + "_stride"],
                                       dtype=torch.bfloat16, device="cpu") for name in inputs.INPUTS}
    result.update(eps=spec["eps"], verify_inputs=False)
    return result


def independent_reference(values, eps):
    # Different precision and operation order, evaluated over the full native
    # hidden width. This detects premature BF16 rounding and missing eps.
    summed = values["x"].double() + values["residual"].double()
    denominator = (torch.linalg.vector_norm(summed, dim=-1, keepdim=True).square()
                   / summed.shape[-1] + eps).sqrt()
    normed = summed / denominator * (values["weight"].double() + 1)
    return normed.to(torch.bfloat16), summed.to(torch.bfloat16)


@pytest.mark.parametrize("spec", META["workload"]["cases"], ids=lambda row: row["sig"])
def test_real_live_shapes_have_fresh_legal_BF16_values_and_CPU_truth(spec):
    source = inputs.CPUInputs(torch, make_args(spec), 31000)
    first, first_expected = source.next()
    second, expected = source.next()
    assert source.generated == 2
    for name in inputs.INPUTS:
        assert first[name].device.type == second[name].device.type == "cpu"
        assert tuple(second[name].shape) == tuple(spec[name + "_shape"])
        assert tuple(second[name].stride()) == tuple(spec[name + "_stride"])
        assert second[name].dtype == torch.bfloat16
        low, high = inputs.DOMAINS[name]
        assert bool(((second[name] >= low) & (second[name] <= high)).all())
        assert not torch.equal(first[name], second[name])
    assert all(value.device.type == "cpu" and value.dtype == torch.bfloat16 for value in expected)
    assert not torch.equal(first_expected[0], expected[0])
    assert not torch.equal(first_expected[1], expected[1])
    selected = {name: value[[0, -1]] if name != "weight" else value for name, value in second.items()}
    reference = independent_reference(selected, spec["eps"])
    assert harness.correct(tuple(value[[0, -1]] for value in expected), reference, 0.02)[0]


@pytest.mark.parametrize("kind", ("ordinary", *inputs.CHALLENGES))
def test_zero_cancellation_and_epsilon_sensitive_challenges_match_independent_truth(kind):
    source = inputs.CPUInputs(torch, make_args(META["workload"]["cases"][1]), 77)
    values, expected = source.next(kind)
    summed = values["x"].float() + values["residual"].float()
    assert harness.correct(expected, independent_reference(values, 1e-6), 0.02)[0]
    if kind == "zero_residual":
        assert values["residual"].count_nonzero() == 0
    elif kind == "zero_sum":
        assert summed.count_nonzero() == 0
        assert all(value.count_nonzero() == 0 for value in expected)
    elif kind == "near_cancellation":
        assert summed.count_nonzero() > 0 and summed.abs().max() <= 2.0**-9
    elif kind == "small_amplitude":
        assert summed.square().mean() < 1e-6 / 4
        without_eps = summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True))
        without_eps *= 1 + values["weight"].float()
        assert not harness.correct(without_eps.to(torch.bfloat16), expected[0], 0.02)[0]
    following, _ = source.next(kind)
    assert not torch.equal(values["weight"], following["weight"])


def test_cache_first_output_candidate_rejected_with_real_CPU_BF16_inputs(monkeypatch):
    bench = load("rmsnorm_cpu_replay_state_test", TASK / "scripts/_bench.py")
    proxy = types.SimpleNamespace(**{name: getattr(torch, name) for name in
                                    ("Generator", "empty_strided", "empty_like", "float32", "rsqrt")},
                                  cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: False))
    cached = []

    def memoizing_candidate(args):
        if not cached:
            cached.extend(independent_reference(args, args["eps"]))
        # Fresh public allocations pass the alias check. The private cache is
        # unaffected when ReplayState poisons the preceding public outputs.
        return tuple(value.clone() for value in cached)

    adapter = types.SimpleNamespace(call=memoizing_candidate, _validate_outputs=cases._validate_outputs)
    state = bench.ReplayState(proxy, adapter, harness, make_args(META["workload"]["cases"][1]), 0.02, 91)
    state.prepare()
    state.run()
    state.validate()
    assert state.checked_invocations == 1
    previous = {name: value.clone() for name, value in state.cpu_inputs.items()}
    state.prepare()
    assert all(not torch.equal(previous[name], state.cpu_inputs[name]) for name in inputs.INPUTS)
    assert all(torch.isnan(value).all() for value in state.outputs)
    state.run()
    with pytest.raises(RuntimeError, match="failed tolerance"):
        state.validate()


def test_CPU_oracle_does_not_round_the_residual_sum_before_normalizing():
    x = torch.tensor([[1.0, 1.0, 0.125, -0.125]], dtype=torch.bfloat16)
    residual = torch.tensor([[0.00390625, -0.00390625, 0.00048828125, -0.00048828125]], dtype=torch.bfloat16)
    weight = torch.tensor([0.0625, -0.0625, 0.125, -0.125], dtype=torch.bfloat16)
    values = {"x": x, "residual": residual, "weight": weight}
    actual = inputs.cpu_reference(torch, values, 1e-6)
    reference = independent_reference(values, 1e-6)
    assert all(torch.equal(a, b) for a, b in zip(actual, reference))
    assert not torch.equal(x.float() + residual.float(), actual[1].float())
