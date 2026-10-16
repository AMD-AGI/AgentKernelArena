"""Finite semantic controls must execute operators and reject wrong behavior."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("torch2flydsl/smoothquant_kernel", "triton2flydsl/sglang/fused_gdn_gating",
         "triton2flydsl/aiter/fused_clamp_act_mul")


def load(path):
    spec = importlib.util.spec_from_file_location("semantic_" + str(abs(hash(str(path)))), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(task):
    base = ROOT / "tasks" / task
    controls = load(base / "scripts/semantic_controls.py")
    replay = load(base / "scripts/replay_checks.py")
    refs = load(base / "scripts/reference_support.py").references
    if task == TASKS[0]:
        h = refs(["_compare_quant_outputs", "_checked_quant_pair", "_compare"])
        model = load(base / "model.py").Model()
        operator = lambda x, s: model(x, s)
        compare = lambda a, e, inp: h._compare_quant_outputs(a, e, inp, None)
    elif task == TASKS[1]:
        h = refs(["reference_gating", "_compare_router_output", "_checked_router_output"])
        operator = lambda A_log, a, b, dt_bias, beta=1., threshold=20.: h.reference_gating(dict(A_log=A_log, a=a, b=b, dt_bias=dt_bias), beta, threshold)
        compare = h._compare_router_output
    else:
        h = refs(["_checked_elementwise_output"])
        def operator(inp, out=None, swiglu_limit=0., activation="silu", weights=None):
            gate, up = inp.float().chunk(2, -1)
            if swiglu_limit > 0:
                gate, up = gate.clamp(max=swiglu_limit), up.clamp(-swiglu_limit, swiglu_limit)
            if activation == "silu":
                result = F.silu(gate) * up
            elif activation == "gelu":
                result = F.gelu(gate, approximate="none") * up
            else:
                result = F.gelu(gate, approximate="tanh") * up
            if weights is not None:
                result *= weights
            result = result.to(inp.dtype)
            if out is not None:
                out.copy_(result)
                return out
            return result
        def compare(a, e, inp):
            h._checked_elementwise_output(a, inp)
            replay.allclose_output(a, e, atol=.01, rtol=.01)
    return base, controls, replay, h, operator, compare


@pytest.mark.parametrize("task", TASKS)
def test_controls_execute_three_draws_of_same_allocations_per_identity(task):
    _, controls, replay, _, operator, compare = environment(task)
    calls = []
    def observed(*args, **kwargs):
        tensors = tuple(value for value in args if isinstance(value, torch.Tensor))
        calls.append((tuple(value.data_ptr() for value in tensors), tuple(value.clone() for value in tensors)))
        return operator(*args, **kwargs)
    records = controls.run_controls(observed, compare, replay.require_unchanged, device="cpu")
    assert len(calls) == len(controls.CONTROL_CASES) * 3
    assert len(records) == len(controls.CONTROL_CASES)
    for offset in range(0, len(calls), 3):
        a, b, c = calls[offset:offset + 3]
        assert a[0] == b[0] == c[0]
        assert all(torch.equal(x, y) for x, y in zip(a[1], c[1]))
        assert any(not torch.equal(x, y) for x, y in zip(a[1], b[1]))
    assert all(r["actual_operator_draws"] == 3 and r["timed"] is False for r in records)


@pytest.mark.parametrize("task", TASKS)
def test_read_only_mutation_is_rejected(task):
    _, controls, replay, _, operator, compare = environment(task)
    def mutate(*args, **kwargs):
        out = operator(*args, **kwargs)
        args[0].add_(1)
        return out
    with pytest.raises(AssertionError, match="read-only"):
        controls.run_controls(mutate, compare, replay.require_unchanged, device="cpu")


@pytest.mark.parametrize("task", TASKS)
def test_reusing_first_result_for_changed_input_is_rejected(task):
    _, controls, replay, _, operator, compare = environment(task)
    cached = None
    def reuse(*args, **kwargs):
        nonlocal cached
        if cached is None:
            cached = operator(*args, **kwargs)
        if kwargs.get("out") is not None:
            kwargs["out"].copy_(cached)
            return kwargs["out"]
        return cached
    with pytest.raises(AssertionError, match="mismatch"):
        controls.run_controls(reuse, compare, replay.require_unchanged, device="cpu")


def test_zero_row_scale_and_codes_are_independent_analytic_answers():
    _, controls, replay, h, operator, compare = environment(TASKS[0])
    inp, expected = controls.make_draw("mixed_zero_rows", device="cpu")
    assert expected[1][0, 0] == 1 and expected[0][0].count_nonzero() == 0
    assert expected[0][1, :2].tolist() == [63, 127]
    assert expected[0][2, :2].tolist() == [-63, -127]
    def omit_zero_rule(x, s):
        hidden = x.float() * s
        scale = hidden.abs().amax(-1, keepdim=True) / 127.
        return (hidden / scale).clamp(-128, 127).to(torch.int8), scale
    x = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
    s = torch.tensor([2., 3.])
    assert h._compare(omit_zero_rule(x, s), operator(x, s))[0]
    with pytest.raises(AssertionError, match="mismatch"):
        controls.run_controls(omit_zero_rule, compare, replay.require_unchanged, device="cpu")


def test_softplus_explicit_threshold_separates_stable_nonlinear_at_original_gate():
    _, controls, replay, h, operator, compare = environment(TASKS[1])
    def omit_threshold(A_log, a, b, dt_bias, beta=1., threshold=20.):
        x = a.float() + dt_bias
        # Independently stable formula, deliberately ignoring threshold.
        bx = x * beta
        g = -A_log.exp() * (bx.clamp_min(0) + torch.log1p(torch.exp(-bx.abs()))) / beta
        return g[None], b.float().sigmoid().to(b.dtype).float()[None]
    torch.manual_seed(7)
    args = (torch.randn(4), torch.randn(2, 4).bfloat16(), torch.randn(2, 4).bfloat16(), torch.randn(4))
    h._compare_router_output(omit_threshold(*args), operator(*args), args[1].dtype)
    default, kwargs, expected = controls.make_draw("default_linear", device="cpu")
    compare(omit_threshold(*default, **kwargs), expected, default[1].dtype)
    explicit, kwargs, expected = controls.make_draw("explicit_threshold", device="cpu")
    error = (omit_threshold(*explicit, **kwargs)[0] - expected[0]).abs().max().item()
    assert error > .06  # > 0.001 + 0.001 * |expected|, with no gate change.
    with pytest.raises(AssertionError, match="mismatch"):
        controls.run_controls(omit_threshold, compare, replay.require_unchanged, device="cpu")


def test_both_gelu_modes_reject_silu_only_and_enforce_owned_output():
    _, controls, replay, _, operator, compare = environment(TASKS[2])
    def only_silu(*args, **kwargs):
        kwargs["activation"] = "silu"
        return operator(*args, **kwargs)
    for case in controls.CONTROL_CASES:
        inp, weights = controls.make_draw(case["params"], device="cpu")
        p = case["params"]
        expected = controls.reference(inp, p["limit"], weights, p["activation"])
        with pytest.raises(AssertionError, match="mismatch"):
            compare(only_silu(inp, swiglu_limit=p["limit"], activation=p["activation"], weights=weights), expected, inp)
    def ignore_out(*args, **kwargs):
        kwargs["out"] = None
        return operator(*args, **kwargs)
    with pytest.raises(AssertionError, match="caller-owned"):
        controls.run_controls(ignore_out, compare, replay.require_unchanged, device="cpu")


@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh"])
def test_gelu_independent_scalar_known_answer(activation):
    _, controls, _, _, _, _ = environment(TASKS[2])
    x = torch.tensor([[-1., 1., 2., -2.]])
    def act(v):
        if activation == "gelu":
            return .5 * v * (1 + math.erf(v / math.sqrt(2)))
        return .5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) * (v + .044715 * v ** 3)))
    expected = torch.tensor([[2 * act(-1.), -2 * act(1.)]])
    torch.testing.assert_close(controls.reference(x, 0., None, activation), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("task", TASKS)
def test_manifest_controls_are_correctness_only_and_task_role_rejects_missing_controls(task, tmp_path):
    base, controls, _, _, _, _ = environment(task)
    manifest = json.loads((base / "cases.json").read_text())["cases"]
    assert manifest[-len(controls.CONTROL_CASES):] == controls.CONTROL_CASES
    assert all(c["checks"] == ["correctness"] for c in controls.CONTROL_CASES)
    actions = load(base / "scripts/task_actions.py")
    # Exercise the actual additive role gate with its task-local imports, while
    # preserving all module bindings after the fixture.
    import sys
    from unittest.mock import patch
    scripts = SimpleNamespace(semantic_controls=controls)
    h = SimpleNamespace(__file__=str(base / "test_kernel_harness.py"))
    if task == TASKS[0]:
        h.SHAPES = actions.EXPECTED_CASES
    elif task == TASKS[1]:
        h.TEST_SHAPES = actions.EXPECTED_SHAPES
    else:
        h._cases = lambda: actions.EXPECTED_SHAPES
    with patch.dict(sys.modules, {"scripts": scripts, "scripts.semantic_controls": controls}):
        actions.select_role(h, "candidate", False)
        copied = tmp_path / "cases.json"
        copied.write_text(json.dumps({"cases": manifest[:-1]}))
        h.__file__ = str(tmp_path / "test_kernel_harness.py")
        with pytest.raises(ValueError, match="Semantic controls"):
            actions.select_role(h, "candidate", False)


def _smoothquant_correctness_roles(monkeypatch, candidate=None, provided=False):
    """Execute the real correctness function on CPU with finite native-op data.

    The runtime fixture deliberately uses a different zero-row convention.
    Only input placement and CUDA housekeeping are replaced; the protected
    correctness, quantizer checks, comparator, Model and control bodies execute.
    This is a role-wiring regression, not native runtime validation.
    """
    import sys
    base, controls, replay, _, _, _ = environment(TASKS[0])
    real = load(base / "model.py")
    calls = {"model": 0, "aiter": 0, "candidate": 0, "controls": []}

    class CPUModel(real.Model):
        def to(self, *args, **kwargs):
            return self

        def forward(self, *args):
            calls["model"] += 1
            return super().forward(*args)

    def native_fixture(x, s):
        calls["aiter"] += 1
        hidden = x.float() * s.float()
        scale = hidden.abs().amax(-1, keepdim=True) / 127.
        codes = (hidden / torch.where(scale == 0, 1., scale)).clamp(-128, 127).to(torch.int8)
        codes = torch.where(scale == 0, torch.full_like(codes, -128), codes)
        return codes, scale

    def candidate_fixture(x, s):
        calls["candidate"] += 1
        return (candidate or real.Model())(x, s)

    def cpu_controls(operator, compare, unchanged):
        calls["controls"].append(operator)
        return controls.run_controls(operator, compare, unchanged, device="cpu")

    source = (base / "test_kernel_harness.py").read_text()
    tree = ast.parse(source)
    constants = {}
    functions = []
    # Loading the harness normally imports the materialized GPU timer. Compile
    # only its real correctness dependencies; no replacement comparison body.
    names = {"run_correctness", "_checked_quant_pair", "_compare_quant_outputs", "_compare", "_retry"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            functions.append(node)
    model_module = SimpleNamespace(Model=CPUModel, get_init_inputs=real.get_init_inputs)
    kernel_module = SimpleNamespace(flydsl_smoothquant=candidate_fixture)
    ns = dict(constants, torch=torch, _KERNEL_DIR=str(base), KERNEL_FILE="kernel.py",
              _load_module=lambda directory, filename, alias: model_module if filename == "model.py" else None if provided else kernel_module,
              _make_inputs=lambda shape: (torch.tensor([[1., 2., -1., -2.]], dtype=torch.bfloat16), torch.tensor([1., 2., 1., 2.])),
              _aiter_op=native_fixture, require_unchanged=replay.require_unchanged)
    exec(compile(ast.Module(functions, type_ignores=[]), str(base / "test_kernel_harness.py"), "exec"), ns)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    control_module = SimpleNamespace(run_controls=cpu_controls)
    monkeypatch.setitem(sys.modules, "scripts", SimpleNamespace(semantic_controls=control_module))
    monkeypatch.setitem(sys.modules, "scripts.semantic_controls", control_module)
    return lambda: ns["run_correctness"](verbose=False), calls


@pytest.mark.parametrize("provided", [True, False])
def test_smoothquant_zero_controls_use_canonical_model_and_candidate_not_secondary_runtime(monkeypatch, provided):
    run, calls = _smoothquant_correctness_roles(monkeypatch, provided=provided)
    assert run() is True
    # All six original runtime cross-checks remain. Zero controls call the
    # canonical Model six times, and the candidate independently when present.
    assert calls["aiter"] == 6
    assert calls["model"] == 12
    assert calls["candidate"] == (0 if provided else 12)
    assert len(calls["controls"]) == (1 if provided else 2)


@pytest.mark.parametrize("zero_codes,zero_scale", [(-128, 0.), (0, 0.), (-128, 1.)])
def test_smoothquant_candidate_runtime_zero_convention_is_rejected_at_original_gate(monkeypatch, zero_codes, zero_scale):
    base = ROOT / "tasks" / TASKS[0]
    model = load(base / "model.py").Model()
    def wrong(x, s):
        codes, scale = model(x, s)
        zeros = (x.float() * s).abs().amax(-1, keepdim=True) == 0
        return (torch.where(zeros, torch.full_like(codes, zero_codes), codes),
                torch.where(zeros, torch.full_like(scale, zero_scale), scale))
    run, calls = _smoothquant_correctness_roles(monkeypatch, candidate=wrong)
    with pytest.raises(AssertionError, match="Numerical mismatch"):
        run()
    assert calls["aiter"] == 6
    assert calls["candidate"] == 7  # Original six pass; first zero control fails.
