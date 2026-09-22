"""Full action/guard tests for packed attention segments and zero denominators."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2triton/vllm/triton_reduce_segments'


def load(path, name, cpu=False):
    tree = ast.parse(path.read_text())
    if cpu:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 'cuda':
                node.value = 'cpu'
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.chdir(ROOT)
    monkeypatch.setitem(sys.modules, 'triton', SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    yield
    torch.set_num_threads(previous)


def candidate(partial, maxima, sums, output, lengths, starts, tile_size=16, fault=None):
    """Independent scalar sequence/token traversal with FP64 arithmetic."""
    segments = partial.shape[2]
    for seq, length in enumerate(lengths.tolist()):
        per_segment = (length + segments * tile_size - 1) // (segments * tile_size)
        active = (length + per_segment * tile_size - 1) // (per_segment * tile_size)
        if fault == 'all_segments':
            active = segments
        for token in range(int(starts[seq]), int(starts[seq + 1])):
            for head in range(partial.shape[1]):
                maxes = maxima[token, head, :active].double()
                weights = (maxes - maxes.max()).exp()
                denominator = (sums[token, head, :active].double() * weights).sum()
                numerator = (partial[token, head, :active].double() * weights[:, None]).sum(0)
                value = numerator / denominator if denominator else torch.zeros_like(numerator)
                if fault == 'zero_denominator' and not denominator:
                    value = numerator
                output[token, head].copy_(value[:output.shape[-1]])
    if fault == 'readonly':
        partial.zero_()
    return output.clone() if fault == 'return_copy' else output


def harness(monkeypatch, fault=None):
    controls = load(TASK / '_upstream_controls.py', 'upstream', cpu=True)
    monkeypatch.setitem(sys.modules, '_upstream_controls', controls)
    h = load(TASK / 'scripts/task_runner.py', 'harness', cpu=True)
    checks = load(TASK / '_arena_checks.py', 'checks')
    def public(partial, maxima, sums, output, lengths, starts, tile_size=16):
        return candidate(partial, maxima, sums, output, lengths, starts, tile_size, fault)
    h.load_module = lambda: SimpleNamespace(reduce_attention_segments=public)
    checks.install(h)
    action = load(TASK / '_arena_eval.py', 'action')
    action.load_harness = lambda: h
    return action, h, checks


@pytest.mark.parametrize('role', ['baseline', 'candidate'])
def test_full_correctness_action_accepts_original_and_both_added_domains(monkeypatch, role):
    action, _, _ = harness(monkeypatch)
    result = action.evaluate(role, 'correctness')
    assert result['status'] == 'PASS', result
    assert len(result['cases']) == 7
    assert sum('performance' in row['checks'] for row in result['cases']) == 5


@pytest.mark.parametrize('fault,index', [('all_segments', 10000), ('zero_denominator', 10001),
                                      ('readonly', 10000), ('return_copy', 10001)])
def test_actual_installed_guard_rejects_wrong_control_candidate(monkeypatch, fault, index):
    _, h, _ = harness(monkeypatch, fault)
    ok, reason = h.run_correctness(case_index=index)
    assert not ok and reason


def test_original_five_reference_values_and_replay_values_are_bit_identical(monkeypatch):
    _, h, checks = harness(monkeypatch)
    for shape in h.TEST_SHAPES:
        partial, maxima, sums, output, lengths, starts = h.make_test_data(*shape, device='cpu')
        for replay in (False, True):
            if replay:
                partial.neg_()
                maxima.copy_(maxima.roll(1, dims=-1))
                sums.mul_(0.75)
            before = h.reference_reduce(partial, maxima, sums, output.shape[-1])
            after = checks.reference(h, (partial, maxima, sums, lengths, starts), output)
            assert before.dtype == after.dtype == torch.float16
            assert torch.equal(before, after), shape


def test_known_answer_for_packed_routing_inactive_segment_and_zero_denominator(monkeypatch):
    _, h, checks = harness(monkeypatch)
    partial = torch.tensor([[[[2.], [900.], [900.], [900.]]],
                            [[[3.], [6.], [9.], [900.]]],
                            [[[3.], [6.], [9.], [900.]]]])
    maxima = torch.zeros(3, 1, 4)
    sums = torch.ones(3, 1, 4)
    sums[2].zero_()
    lengths = torch.tensor([1, 127, 65], dtype=torch.int32)
    starts = torch.tensor([0, 1, 1, 3], dtype=torch.int32)
    output = torch.empty(3, 1, 1, dtype=torch.float32)
    expected = torch.tensor([2., 6., 0.]).reshape(3, 1, 1)
    actual = checks.reference(h, (partial, maxima, sums, lengths, starts), output)
    assert torch.equal(actual, expected)
    assert torch.equal(candidate(partial, maxima, sums, output, lengths, starts), expected)
