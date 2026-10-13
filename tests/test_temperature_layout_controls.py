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
    assert "partial final 8192-element block" in manifest["cases"][-1]["params"]["coverage"]


@pytest.mark.parametrize("replacement", [
    "def apply_temperature(logits, idx_mapping, temperature, extra=1):",
    "def apply_temperature(logits, idx_mapping, temperature, *args):",
    "def apply_temperature(logits, idx_mapping, temperature, **kwargs):",
    "def apply_temperature(logits, idx_mapping, temperature=1):",
    "def apply_temperature(logits, idx_mapping, *, temperature):",
    "def apply_temperature(logits, idx_mapping, /, temperature):",
    "def apply_temperature(logits, temperature, idx_mapping):",
    "def apply_temperature(logits, mapping, temperature):",
])
def test_public_wrapper_signature_mutations_fail_before_harness(replacement, tmp_path, monkeypatch):
    adapter = _load(TASK / "_arena_eval.py", "temperature_signature_adapter")
    original = "def apply_temperature(logits, idx_mapping, temperature):"
    source = (TASK / "source/triton_temperature.py").read_text()
    assert source.count(original) == 1
    target = tmp_path / "source/triton_temperature.py"
    target.parent.mkdir()
    target.write_text(source.replace(original, replacement, 1))
    monkeypatch.setattr(adapter, "ROOT", tmp_path)

    def unexpected_harness():
        pytest.fail("invalid signature reached the GPU harness")

    monkeypatch.setattr(adapter, "load_harness", unexpected_harness)
    result = adapter.evaluate("candidate", "compile")
    assert result["status"] == "FAIL"
    assert result["failure_kind"] == "execution_failure"
    assert "exact signature apply_temperature(logits, idx_mapping, temperature)" in result["reason"]


def test_public_wrapper_signature_accepts_original_and_ignores_annotations(tmp_path, monkeypatch):
    adapter = _load(TASK / "_arena_eval.py", "temperature_signature_adapter_valid")
    data = adapter.load_manifest()
    assert adapter.inspect_candidate(data, require_implemented=True) == "implemented"
    original = "def apply_temperature(logits, idx_mapping, temperature):"
    annotated = "def apply_temperature(logits: object, idx_mapping: object, temperature: object) -> object:"
    source = (TASK / "source/triton_temperature.py").read_text()
    target = tmp_path / "source/triton_temperature.py"
    target.parent.mkdir()
    target.write_text(source.replace(original, annotated, 1))
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    assert adapter.inspect_candidate(data, require_implemented=True) == "implemented"


@pytest.mark.parametrize("wrapper_source, accepted", [
    ("def apply_temperature(logits, idx_mapping, temperature):\n    return None\n", True),
    ("def apply_temperature(logits, idx_mapping, temperature):\n    return None\n"
     "apply_temperature.__defaults__ = ()\n"
     "apply_temperature.__kwdefaults__ = {}\n", True),
    ("def apply_temperature(logits, idx_mapping, temperature):\n    return None\n"
     "_original_apply_temperature = apply_temperature\n"
     "apply_temperature = lambda logits, idx_mapping, temperature, extra=None: "
     "_original_apply_temperature(logits, idx_mapping, temperature)\n", False),
    ("import inspect\n"
     "def apply_temperature(logits, idx_mapping, temperature):\n    return None\n"
     "_original_apply_temperature = apply_temperature\n"
     "apply_temperature = lambda logits, idx_mapping, temperature, extra=None: "
     "_original_apply_temperature(logits, idx_mapping, temperature)\n"
     "apply_temperature.__signature__ = inspect.signature(_original_apply_temperature)\n", False),
    ("import functools\n"
     "def variadic(fn):\n"
     "    @functools.wraps(fn)\n"
     "    def wrapper(*args, **kwargs):\n"
     "        return fn(*args, **kwargs)\n"
     "    return wrapper\n"
     "@variadic\n"
     "def apply_temperature(logits, idx_mapping, temperature):\n"
     "    return None\n", False),
])
def test_exported_wrapper_signature_checked_by_native_loader(
        wrapper_source, accepted, task_modules, tmp_path, monkeypatch):
    replay, contract = task_modules
    monkeypatch.setitem(sys.modules, "_arena_contract", contract)
    adapter = _load(TASK / "_arena_eval.py", "temperature_exported_signature_adapter")
    source_path = tmp_path / "source/triton_temperature.py"
    source_path.parent.mkdir()
    source_path.write_text(
        "class triton:\n"
        "    @staticmethod\n"
        "    def jit(fn): return fn\n"
        "@triton.jit\n"
        "def _temperature_kernel(): return None\n"
        + wrapper_source)
    runner_path = tmp_path / "scripts/task_runner.py"
    runner_path.parent.mkdir()
    runner_path.write_bytes((TASK / "scripts/task_runner.py").read_bytes())
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    assert adapter.inspect_candidate(adapter.load_manifest(), require_implemented=True) == "implemented"
    cwd = os.getcwd()
    try:
        result = adapter.evaluate("candidate", "compile")
    finally:
        os.chdir(cwd)
    assert result["status"] == ("PASS" if accepted else "FAIL")
    if not accepted:
        assert result["failure_kind"] == "execution_failure"
        assert "exact signature apply_temperature(logits, idx_mapping, temperature)" in result["reason"]


def test_native_controls_require_large_mapping_partial_block_and_padded_stride(task_modules):
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
    assert seen[2] == ((4, 8449), (8449, 1), [2, 0, 3, 1])
    assert seen[3][0] == (4, 256)
    assert seen[3][1] == (264, 1)
    tail = list(contract.control_inputs(None))[2]
    expected = contract.reference(None, tail)
    torch.testing.assert_close(expected[0, 8192:], tail[0][0, 8192:] * 2, atol=0, rtol=0)
    torch.testing.assert_close(expected[2, 8192:], tail[0][2, 8192:] / 2, atol=0, rtol=0)
    assert not torch.equal(expected[0, 8192:], tail[0][0, 8192:])
    assert not torch.equal(expected[2, 8192:], tail[0][2, 8192:])
    before, after = padded[0]
    torch.testing.assert_close(after[0], before[0] * 2, atol=0, rtol=0)
    torch.testing.assert_close(after[1], before[1], atol=0, rtol=0)
    torch.testing.assert_close(after[2], before[2] / 2, atol=0, rtol=0)
    torch.testing.assert_close(after[3], before[3], atol=0, rtol=0)


def test_native_controls_reject_candidate_skipping_partial_final_block(task_modules):
    replay, contract = task_modules
    seen = []

    def skip_partial_final_block(logits, mapping, temperature):
        seen.append(tuple(logits.shape))
        if logits.shape[1] > 8192 and logits.shape[1] % 8192:
            _correct(contract, logits[:, :8192], mapping, temperature)
        else:
            _correct(contract, logits, mapping, temperature)

    # The mutation satisfies the existing small and 32768-wide controls.
    for args in list(contract.control_inputs(None))[:2]:
        replay.Recorder(None, contract).wrap(skip_partial_final_block)(*args)
    assert seen == [(4, 4), (64, 32768)]

    with pytest.raises(AssertionError):
        contract.controls(None, replay.Recorder(None, contract).wrap(skip_partial_final_block), "cpu")
    assert seen[-1] == (4, 8449)


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
    assert ((4, 8449), (8449, 1)) in seen
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
