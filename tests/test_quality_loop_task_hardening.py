"""CPU regression checks for false passes fixed by the task-quality campaigns."""

import importlib.util
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner(monkeypatch):
    # Task runners change cwd/sys.path at import; restore both after each test.
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    helper = load_file(ROOT / "src/tools/perf/aka_benchmark.py", "_aka_benchmark")
    monkeypatch.setitem(sys.modules, "_aka_benchmark", helper)

    def load(task):
        path = ROOT / 'tasks' / task / 'scripts/task_runner.py'
        monkeypatch.syspath_prepend(str(path.parent))
        monkeypatch.syspath_prepend(str(path.parent.parent))
        for key in list(sys.modules):
            if key == 'scripts' or key.startswith('scripts.') or key in {'_contract_checks', '_numerical_contract'}:
                monkeypatch.delitem(sys.modules, key, raising=False)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 'cuda': node.value = 'cpu'
        module = __import__('types').ModuleType('task_runner_test')
        module.__file__ = str(path)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(tree, str(path), 'exec'), module.__dict__)
        return module

    return load


@pytest.mark.parametrize("bad_output", ["empty", "missing", "extra", "list", "none", "shape", "wrong"])
def test_kda_rejects_incomplete_or_invalid_outputs(runner, monkeypatch, bad_output):
    task = runner("triton2triton/vllm/triton_kda_dot_kkt_intra")
    refs = (torch.ones(2), torch.ones(2) * 2)
    outputs = {
        "empty": (), "missing": refs[:1], "extra": refs + refs[:1],
        "list": list(refs), "none": (refs[0], None),
        "shape": (refs[0], torch.ones(1)), "wrong": (refs[0], torch.zeros(2)),
    }
    monkeypatch.setattr(task, "SEEDS", [42])
    monkeypatch.setattr(task, "load_module", lambda: SimpleNamespace(
        kda_dot_kkt_intra=lambda *args, **kwargs: outputs[bad_output]
    ))
    monkeypatch.setattr(task, "gen_inputs", lambda *args: ((0,), {}))
    monkeypatch.setattr(task, "reference", lambda *args: refs)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    ok, error = task.run_correctness()
    assert not ok
    assert error
    outputs[bad_output] = refs
    assert task.run_correctness() == (True, None)


@pytest.mark.parametrize("replacement", [float("nan"), float("inf"), 0.0])
def test_logit_bias_rejects_corrupted_masked_logits(runner, replacement):
    check = load_file(ROOT / 'tasks/triton2triton/vllm/triton_logit_bias/_arena_replay.py', 'masked_check').compare
    reference = torch.tensor([1.0, -float('inf'), float('inf')])
    output = reference.clone(); output[1] = replacement
    with pytest.raises(AssertionError): check(output, reference, atol=.01, rtol=.01)
    check(reference.clone(), reference, atol=.01, rtol=.01)
    output = reference.clone(); output[0] = 2.0
    with pytest.raises(AssertionError): check(output, reference, atol=.01, rtol=.01)


def test_moe_rejects_omitted_writes_even_with_small_reference(runner):
    task = runner("triton2triton/vllm/triton_fused_moe_gptq_awq")
    inputs, options = task.control_inputs('int4_explicit', 'cpu')
    oracle, check = task.numerical_contract(options)
    reference = oracle(inputs)
    check(reference.clone(), reference)
    for wrong in (torch.zeros_like(reference), reference * .5):
        with pytest.raises(Exception, match='mismatch|bound|accuracy'):
            check(wrong, reference)


@pytest.mark.parametrize("seed", [0, 17])
def test_assign_score_backward_reference_accumulates_duplicate_neighbors(runner, seed):
    task = runner("hip2hip/others/assign_score_withk")
    torch.manual_seed(seed)
    scores = torch.randn(2, 3, 3, 2, dtype=torch.float64, requires_grad=True)
    points = torch.randn(2, 4, 2, 3, dtype=torch.float64, requires_grad=True)
    centers = torch.randn_like(points, requires_grad=True)
    indices = torch.tensor([[[0, 1, 1], [0, 0, 2], [2, 1, 2]],
                            [[1, 1, 1], [3, 2, 3], [0, 3, 0]]])
    # Use scalar indexing plus autograd, independent of the new gather/scatter reference.
    output = torch.stack([
        torch.stack([
            torch.stack([
                (scores[b, n, k, :, None]
                 * (points[b, indices[b, n, k]] - centers[b, indices[b, n, 0]])).sum(0)
                for k in range(3)
            ]) for n in range(3)
        ]) for b in range(2)
    ]).permute(0, 3, 1, 2)
    upstream = torch.randn_like(output)
    expected = torch.autograd.grad(output, (scores, points, centers), upstream)
    actual = task.cpu_assign_score_withk_backward_vectorized(
        scores.detach(), points.detach(), centers.detach(), indices, upstream)
    for got, reference in zip(actual, expected):
        torch.testing.assert_close(got, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('bad', [False, True])
def test_unrouted_quant_control_checks_actual_candidate_and_exact_arithmetic(runner, monkeypatch, bad):
    task = runner('triton2triton/vllm/triton_fused_moe_gptq_awq')
    calls = []
    def candidate(A, qweight, scales, zeros, ids, weights, **options):
        calls.append(options.copy())
        inputs = dict(A=A, qweight=qweight, scales=scales, zeros=zeros, ids=ids, weights=weights)
        output = task.reference(inputs, options)
        return output * weights[:, None].to(output.dtype) if bad else output
    monkeypatch.setattr(task, 'load_module', lambda: SimpleNamespace(fused_moe_gptq_awq=candidate))
    ok, error = task.run_correctness(control='int4_unrouted')
    assert ok is (not bad), error
    assert len(calls) == 1 and calls[0]['mul_routed_weight'] is False


@pytest.mark.parametrize('control', ['m_tail', 'n_tail', 'k_below64', 'k_above64'])
@pytest.mark.parametrize('bad', [False, True])
def test_mmk_new_tails_detect_missing_reduction_tile(runner, monkeypatch, control, bad):
    task = runner('triton2triton/vllm/triton_moe_mmk')
    def candidate(a, b):
        return ((a[:, :32].float() @ b[:32].float()) if bad else
                (a.float() @ b.float())).to(a.dtype)
    monkeypatch.setattr(task, 'load_module', lambda: SimpleNamespace(moe_matmul=candidate))
    ok, error = task.run_correctness(control=control)
    assert ok is (not bad), error
