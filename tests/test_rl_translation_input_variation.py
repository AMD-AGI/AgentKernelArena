"""Scaled FP8 GEMM timing must recompute independently varying X and W."""
import ast
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/triton2flydsl/aiter/gemm_a8w8"


def _load_controls():
    name = "a8w8_input_variation_controls"
    package = types.ModuleType(name)
    package.__path__ = [str(TASK / "scripts")]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(
        name + ".sample_controls", TASK / "scripts/sample_controls.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


controls = _load_controls()


def _inputs(seed=17):
    generator = torch.Generator().manual_seed(seed)
    raw = (torch.randn((4, 32), generator=generator),
           torch.randn((8, 32), generator=generator))
    scale = tuple(v.abs().amax(1, keepdim=True) / 448 for v in raw)
    return ((raw[0] / scale[0]).to(torch.float8_e4m3fn),
            (raw[1] / scale[1]).to(torch.float8_e4m3fn),
            scale[0], scale[1].T.contiguous(), None)


def _reference(x, w, xs, ws, bias=None, dtype=torch.bfloat16):
    value = (x.float() @ w.float().T) * (xs @ ws)
    return (value + bias if bias is not None else value).to(dtype)


class Timed:
    def __init__(self, fn):
        self.fn = fn
        self.bound = True
        self.outputs = None

    def rerun(self):
        self.outputs = self.fn()
        return self.outputs


def _stream(inputs, *, samples=5):
    return controls.MeasuredScaledGemmInputs(
        *inputs, seed=0, case_index=0, samples=samples,
        output_dtype=torch.bfloat16, reference=lambda: _reference(*inputs))


def _run(stream, fn, *, calibration=3):
    for _ in range(calibration):
        stream.prepare()
        fn()
    timed = Timed(fn)
    for _ in range(stream.samples):
        stream.prepare()
        stream.before_sample(1)
        timed.outputs = fn()
        stream.observe(timed.outputs)
    stream.finish()
    return timed


def test_calibration_and_first_scored_call_all_receive_independent_operands():
    inputs = _inputs()
    originals = tuple(v.clone() for v in inputs if v is not None)
    seen = []
    stream = _stream(inputs)
    def compute():
        seen.append((inputs[0].view(torch.uint8).clone(),
                     inputs[1].view(torch.uint8).clone()))
        return _reference(*inputs)
    timed = _run(stream, compute)
    assert all(not torch.equal(seen[i][side], seen[j][side])
               for side in (0, 1) for i in range(8) for j in range(i + 1, 8))
    assert not torch.equal(seen[0][0], originals[0].view(torch.uint8))
    assert [row["prepare_index"] for row in stream.measured_draws] == list(range(3, 8))
    measured = seen[3:]
    assert all(not torch.equal(measured[i][side], measured[j][side])
               for side in (0, 1) for i in range(5) for j in range(i + 1, 5))
    metadata = stream.validate_replays(timed)
    assert metadata["measured_sample_outputs_checked"] == 5
    assert metadata["replay_operand_controls_checked"] == 3
    stream.restore()
    controls.require_unchanged(stream.inputs, originals)


def test_cached_accumulator_passes_old_scale_only_replay_but_fails_fresh_operands():
    inputs = _inputs()
    x, w, xs, ws, _ = inputs
    accumulator = x.float() @ w.float().T
    def cached():
        return (accumulator * (xs @ ws)).to(torch.bfloat16)
    xs.mul_(0.5)  # The old replay changed only this scale.
    torch.testing.assert_close(cached(), _reference(*inputs), atol=0.02, rtol=0.01)
    stream = _stream(inputs)
    with pytest.raises(AssertionError, match="Numerical mismatch"):
        _run(stream, cached)
    assert stream.checked == 0


@pytest.mark.parametrize("bad_sample", [1, 2, 3])
def test_reusing_any_middle_measured_output_fails(bad_sample):
    inputs = _inputs()
    stream = _stream(inputs)
    previous = None
    def compute():
        nonlocal previous
        result = _reference(*inputs)
        if stream.pending and stream.checked == bad_sample:
            return previous.clone()
        previous = result.clone()
        return result
    with pytest.raises(AssertionError, match="Numerical mismatch"):
        _run(stream, compute)
    assert stream.checked == bad_sample


@pytest.mark.parametrize("operand", [0, 1, 2, 3])
def test_measured_input_mutation_cannot_be_erased_by_next_preparation(operand):
    inputs = _inputs()
    stream = _stream(inputs)
    def compute():
        result = _reference(*inputs)
        if stream.pending:
            inputs[operand].zero_()
        return result
    with pytest.raises(AssertionError, match="read-only input"):
        _run(stream, compute)
    assert stream.checked == 0


def test_per_replay_call_batching_and_missing_observations_fail_closed():
    stream = _stream(_inputs())
    stream.prepare()
    with pytest.raises(AssertionError, match="one fresh checked GEMM call"):
        stream.before_sample(2)
    with pytest.raises(AssertionError, match="Not every reported"):
        stream.finish()


@pytest.mark.parametrize("cached_operand", ["x", "w", "both"])
def test_unseen_single_operand_and_joint_replays_reject_caches(cached_operand):
    inputs = _inputs()
    stream = _stream(inputs)
    timed = _run(stream, lambda: _reference(*inputs))
    old_x, old_w = inputs[0].float().clone(), inputs[1].float().clone()
    def cached():
        x = old_x if cached_operand in {"x", "both"} else inputs[0].float()
        w = old_w if cached_operand in {"w", "both"} else inputs[1].float()
        return ((x @ w.T) * (inputs[2] @ inputs[3])).to(torch.bfloat16)
    timed.fn = cached
    with pytest.raises(AssertionError, match="Numerical mismatch"):
        stream.validate_replays(timed)
    controls.require_unchanged(stream.inputs, stream.prepared_inputs)


@pytest.mark.parametrize("behavior", ["correct", "cache", "wrong_middle", "warmup_mutates"])
def test_actual_benchmark_entrypoint_wires_preparation_and_every_output(
    behavior, monkeypatch, tmp_path,
):
    tree = ast.parse((TASK / "test_kernel_harness.py").read_text())
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "run_benchmark"]
    inputs = _inputs()
    calls = 0
    original = _reference(*inputs)
    def candidate(*args):
        nonlocal calls
        calls += 1
        output = _reference(*args)
        if behavior == "cache":
            return original.clone()
        if behavior == "wrong_middle" and calls == 15:
            return torch.zeros_like(output)
        if behavior == "warmup_mutates" and calls == 2:
            args[0].zero_()
        return output
    def benchmark(fn, *, warmup, repetition, timed_run, prepare_fn):
        assert warmup == 0 and repetition == 100
        for _ in range(3):
            prepare_fn()
            fn()
        for _ in range(repetition):
            prepare_fn()
            timed_run.before_sample(1)
            output = fn()
            timed_run.after_sample(output)
        timed_run.outputs = output
        timed_run.bound = True
        timed_run.rerun = lambda: (prepare_fn(), fn())[1]
        return 0.1, {"benchmark_method": "cuda_graph", "benchmark_effective_repeats": 1}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    ns = {"_load_source": lambda: types.SimpleNamespace(gemm_a8w8=candidate),
          "_fp8_e4m3_dtype": lambda: torch.float8_e4m3fn,
          "TEST_SHAPES": [{"name": "cpu", "M": 4, "N": 8, "K": 32}],
          "_make_inputs": lambda *a: inputs, "_torch_ref": _reference,
          "MeasuredScaledGemmInputs": controls.MeasuredScaledGemmInputs,
          "require_unchanged": controls.require_unchanged,
          "TimedRun": lambda: types.SimpleNamespace(),
          "benchmark_cuda_graph_or_events": benchmark,
          "SEED": 0, "ITERS": 100, "WARMUP": 10,
          "Path": Path, "_HERE": str(tmp_path), "math": math, "json": json}
    exec(compile(ast.Module(selected, type_ignores=[]), "run_benchmark", "exec"), ns)
    if behavior == "correct":
        result = ns["run_benchmark"](verbose=False)
        assert result[0]["measured_sample_outputs_checked"] == 100
        assert result[0]["replay_operand_controls_checked"] == 3
    else:
        with pytest.raises(AssertionError):
            ns["run_benchmark"](verbose=False)
    controls.require_unchanged(inputs[:4], _inputs()[:4])


def test_replay_mutation_is_rejected_and_original_inputs_can_be_restored():
    inputs = _inputs()
    stream = _stream(inputs)
    timed = _run(stream, lambda: _reference(*inputs))
    def mutating_replay():
        output = _reference(*inputs)
        inputs[1].zero_()
        return output
    timed.fn = mutating_replay
    with pytest.raises(AssertionError, match="read-only input"):
        stream.validate_replays(timed)
    controls.require_unchanged(stream.inputs, stream.prepared_inputs)
    stream.restore()
    controls.require_unchanged(stream.inputs, _inputs()[:4])


@pytest.mark.parametrize('calibration', [0, 3, 5])
def test_compute_on_change_cache_recomputes_all_100_actual_samples(calibration):
    inputs = _inputs()
    stream = _stream(inputs, samples=100)
    previous, cached = None, None
    measured_work = []
    def compute():
        nonlocal previous, cached
        now = tuple(value.view(torch.uint8).clone() for value in inputs[:4])
        changed = previous is None or any(not torch.equal(a,b) for a,b in zip(now,previous))
        if changed:
            cached = _reference(*inputs)
            previous = now
        if stream.pending:
            measured_work.append(int(changed))
        return cached.clone()
    try:
        timed = _run(stream, compute, calibration=calibration)
        assert measured_work == [1] * 100
        assert len(stream.measured_draws) == 100
        assert len({row['input_sha256'] for row in stream.measured_draws}) == 100
        assert [row['prepare_index'] for row in stream.measured_draws] == list(range(calibration,calibration+100))
        metadata = stream.validate_replays(timed)
        assert metadata['measured_input_distinct_draws'] == 100
    finally:
        stream.restore()
    controls.require_unchanged(inputs[:4], _inputs()[:4])


def test_paired_seed_plan_and_bytes_match_despite_different_original_values():
    records = []
    for seed in (17, 39):
        inputs = _inputs(seed)
        stream = _stream(inputs)
        try:
            _run(stream, lambda: _reference(*inputs), calibration=5)
            records.append(stream.measured_draws)
        finally:
            stream.restore()
    assert records[0] == records[1]
    assert all(row['operand_seeds'] == [10000+2*row['prepare_index'],10001+2*row['prepare_index']]
               for row in records[0])


def test_missing_preparation_and_duplicate_draw_identity_fail_closed():
    inputs = _inputs()
    stream = _stream(inputs, samples=2)
    stream.prepare()
    stream.before_sample(1)
    stream.observe(_reference(*inputs))
    with pytest.raises(AssertionError, match='one fresh checked'):
        stream.before_sample(1)
    stream.prepare()
    stream.before_sample(1)
    stream.observe(_reference(*inputs))
    stream.measured_draws[1]['input_sha256'] = stream.measured_draws[0]['input_sha256']
    with pytest.raises(AssertionError, match='identical operands'):
        stream.finish()
    stream.restore()


def test_sample_controller_remains_a_protected_task_file(tmp_path):
    import shutil
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
    from src.task_spec import load_task_spec
    workspace = tmp_path / 'workspace'
    shutil.copytree(TASK, workspace, ignore=shutil.ignore_patterns('build','__pycache__'))
    spec = load_task_spec(workspace/'config.yaml', task_id='triton2flydsl/aiter/gemm_a8w8')
    guard = snapshot_workspace_harness(workspace, task_spec=spec)
    helper = workspace / 'scripts/sample_controls.py'
    original = helper.read_bytes()
    helper.write_bytes(original + b'\n# unauthorized source mutation\n')
    with pytest.raises(RuntimeError, match='sample_controls.py'):
        verify_workspace_harness(guard, discard_added=False)
    helper.write_bytes(original)
    verify_workspace_harness(guard, discard_added=False)
