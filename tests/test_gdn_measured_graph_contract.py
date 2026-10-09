"""CPU protocol negatives; real graph timing requires the task's GPU validator."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/triton2triton/vllm/triton_fused_gdn_gating"


def load(path):
    spec = importlib.util.spec_from_file_location("_gdn_sample_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fault", [
    None, "skip_g", "skip_beta", "skip_middle", "partial_beta", "mutate_input",
    "missing_observer", "batched_calls", "missing_repeat_metadata",
])
def test_each_graph_sample_overwrites_both_outputs_and_checks_inputs(monkeypatch, fault):
    monkeypatch.chdir(ROOT)
    checks = load(TASK / "_arena_checks.py")
    harness = load(TASK / "scripts/task_runner.py")
    harness._TimedRun = lambda: SimpleNamespace(bound=False)
    inputs = harness.make_inputs(2, 3)
    pristine = tuple(x.clone() for x in inputs)
    mod = SimpleNamespace(fused_gdn_gating=harness.reference)
    original = mod.fused_gdn_gating
    prepared = []

    def fn():
        return mod.fused_gdn_gating(*inputs)

    def prior_prepare():
        prepared.append(True)

    def benchmark(measured, *, timed_run, prepare_fn, **options):
        assert options == {"warmup": 10, "repetition": 100}
        outputs = measured()

        def write_outputs():
            fresh = harness.reference(*inputs)
            for dst, src in zip(outputs, fresh):
                dst.copy_(src)
            return outputs

        for index in range(options["repetition"]):
            prepare_fn()
            assert all(torch.isnan(output).all() for output in outputs)
            fresh = harness.reference(*inputs)
            # These faults model a captured device launch that stops writing,
            # even though the eager capture and later diagnostic replay work.
            for output_index, (dst, src) in enumerate(zip(outputs, fresh)):
                if (fault == "skip_middle" and index == 49
                        or fault == "skip_g" and output_index == 0
                        or fault == "skip_beta" and output_index == 1):
                    continue
                if fault == "partial_beta" and output_index == 1:
                    dst.flatten()[:-1].copy_(src.flatten()[:-1])
                else:
                    dst.copy_(src)
            if fault == "mutate_input" and index == 49:
                inputs[0].add_(1)
            if not (fault == "missing_observer" and index == 49):
                timed_run.after_sample(outputs)
        timed_run.bound, timed_run.outputs, timed_run.rerun = True, outputs, write_outputs
        metadata = {"benchmark_method": "cuda_graph"}
        if fault != "missing_repeat_metadata":
            metadata["benchmark_effective_repeats"] = 2 if fault == "batched_calls" else 1
        return .125, metadata

    def run():
        return checks.checked_benchmark(harness, benchmark, fn, warmup=10,
                                        repetition=100, prepare_fn=prior_prepare)

    if fault is None:
        ms, metadata = run()
        assert ms == .125 and metadata["measured_samples_checked"] == 100
        assert metadata["timed_output_checked"] and metadata["perturbed_input_replay_checked"]
        assert len(prepared) == 100
    else:
        with pytest.raises(AssertionError):
            run()
    assert mod.fused_gdn_gating is original
    for value, saved in zip(inputs, pristine):
        assert torch.equal(value, saved)
