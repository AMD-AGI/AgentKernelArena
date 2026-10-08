"""Temperature task controls; CPU checks do not qualify the GPU task."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_temperature"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


@pytest.fixture
def task_modules(monkeypatch):
    replay = _load(TASK / "_arena_replay.py", "temperature_layout_replay")
    contract = _load(TASK / "_arena_contract.py", "temperature_layout_contract")
    monkeypatch.setitem(sys.modules, "_arena_replay", replay)
    return replay, contract


def _correct(contract, logits, mapping, temperature):
    logits.copy_(contract.reference(None, (logits, mapping, temperature)))


def test_manifest_keeps_five_original_scored_cases():
    manifest = json.loads((TASK / "workloads.json").read_text())
    assert manifest["input_table"] == [[4, 256], [8, 1024], [16, 4096],
                                       [32, 8192], [64, 32768]]
    assert [(row["test_case_id"], row["checks"]) for row in manifest["cases"]] == [
        (f"perf{index}", ["correctness", "performance"]) for index in range(1, 6)
    ] + [("contract_controls", ["correctness"])]


def test_native_controls_require_large_mapping_and_real_padded_stride(task_modules):
    replay, contract = task_modules
    seen = []
    padded = []

    def correct(logits, mapping, temperature):
        seen.append((tuple(logits.shape), logits.stride(), mapping.tolist()))
        before = logits.clone() if logits.stride(0) > logits.shape[1] else None
        _correct(contract, logits, mapping, temperature)
        if before is not None:
            padded.append((before, logits.clone()))

    contract.controls(None, replay.Recorder(None, contract).wrap(correct), "cpu")
    assert seen[1][0] == (64, 32768)
    assert seen[1][2] != list(range(64))
    assert seen[2][0] == (4, 256)
    assert seen[2][1] == (264, 1)
    before, after = padded[0]
    torch.testing.assert_close(after[0], before[0] * 2, atol=0, rtol=0)
    torch.testing.assert_close(after[1], before[1], atol=0, rtol=0)
    torch.testing.assert_close(after[2], before[2] / 2, atol=0, rtol=0)
    torch.testing.assert_close(after[3], before[3], atol=0, rtol=0)


def test_ignored_mapping_fails_large_vocab_control(task_modules):
    replay, contract = task_modules
    large = list(contract.control_inputs(None))[1]

    def ignore_mapping(logits, mapping, temperature):
        identity = torch.arange(mapping.numel(), dtype=mapping.dtype)
        _correct(contract, logits, identity, temperature)

    wrapped = replay.Recorder(None, contract).wrap(ignore_mapping)
    with pytest.raises(AssertionError):
        wrapped(*large)


def test_contiguous_row_assumption_and_padding_corruption_fail(task_modules):
    replay, contract = task_modules

    def assume_contiguous(logits, mapping, temperature):
        if logits.stride(0) > logits.shape[1]:
            flat_rows = torch.as_strided(logits, logits.shape, (logits.shape[1], 1))
            _correct(contract, flat_rows, mapping, temperature)
        else:
            _correct(contract, logits, mapping, temperature)

    with pytest.raises(AssertionError):
        contract.controls(None, replay.Recorder(None, contract).wrap(assume_contiguous), "cpu")

    def corrupt_padding(logits, mapping, temperature):
        _correct(contract, logits, mapping, temperature)
        if logits.stride(0) > logits.shape[1]:
            all_columns = torch.as_strided(logits, (logits.shape[0], logits.stride(0)),
                                           logits.stride())
            all_columns[:, logits.shape[1]:].fill_(-999)

    with pytest.raises(AssertionError):
        contract.controls(None, replay.Recorder(None, contract).wrap(corrupt_padding), "cpu")


def test_replay_varies_mapping_and_restores_it(task_modules):
    replay, contract = task_modules
    args = next(contract.control_inputs(None))
    before = replay.clone(args)
    contract.fresh(args)
    assert not torch.equal(args[1], before[1])
    assert not torch.equal(args[2], before[2])
    replay.restore(args, before)
    for actual, expected in zip(args, before):
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/ROCm GPU")
def test_actual_gpu_kernel_passes_native_controls_and_mapping_replay(task_modules, monkeypatch):
    pytest.importorskip("triton")
    replay, contract = task_modules
    harness = _load(TASK / "scripts/task_runner.py", "temperature_layout_harness")
    original_loader = harness.load_module
    seen = []

    def load_with_stride_probe():
        module = original_loader()
        actual = module.apply_temperature

        def inspected(logits, idx_mapping, temperature):
            seen.append((tuple(logits.shape), logits.stride()))
            return actual(logits, idx_mapping, temperature)

        module.apply_temperature = inspected
        return module

    harness.load_module = load_with_stride_probe
    replay.install(harness, contract)
    ok, error = harness.run_correctness(case_index=contract.CONTROL_INDEX)
    assert ok, error
    assert ((64, 32768), (32768, 1)) in seen
    assert ((4, 256), (264, 1)) in seen

    source = original_loader()
    logits = torch.arange(8 * 1024, dtype=torch.float32, device="cuda").reshape(8, 1024) + 1
    mapping = torch.arange(8, dtype=torch.int32, device="cuda")
    temperature = torch.tensor([0.5, 2, 1.5, 0.75, 3, 0.25, 1.25, 2.5], device="cuda")
    args = (logits, mapping, temperature)
    pristine = replay.clone(args)
    replay_mappings = []

    def observed(logits, idx_mapping, temperature):
        replay_mappings.append(idx_mapping.cpu().clone())
        return source.apply_temperature(logits, idx_mapping, temperature)

    class SingleRun:
        def _bind(self, run, output):
            self.run, self.outputs = run, output

        def rerun(self):
            self.outputs = self.run()
            return self.outputs

    def one_gpu_call(run, *, timed_run, **_options):
        timed_run._bind(run, run())
        return 1.0, {"benchmark_method": "single_gpu_call_test"}

    # The committed task runner has only the generated benchmark stub. This
    # single-invocation regression supplies its own replay holder.
    monkeypatch.setattr(harness, "_TimedRun", SingleRun, raising=False)
    recorder = replay.Recorder(harness, contract)
    function = recorder.wrap(observed)
    _, metadata = recorder.benchmark(one_gpu_call, lambda: function(*args))
    assert metadata["timed_output_checked"] and metadata["replay_input_control_checked"]
    assert any(not torch.equal(item, pristine[1].cpu()) for item in replay_mappings)
    for actual, expected in zip(args, pristine):
        assert torch.equal(actual, expected)
