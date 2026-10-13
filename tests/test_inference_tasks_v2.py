"""CPU regression tests for the task contracts, not GPU qualification."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
from src.task_spec import load_task_spec
from src.tools.perf.aka_benchmark import hip_source_graph_capture_policy

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "wvsplitk": ROOT / "tasks/hip2hip/vllm/wvsplitk",
    "paged_attention_large": ROOT / "tasks/hip2hip/vllm/paged_attention_large",
    "topk_forward": ROOT / "tasks/triton2triton/triton_kernels/topk_forward",
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def api(name):
    return load(TASKS[name] / "scripts/task_api.py", f"api_{name}")


def runner(name, monkeypatch):
    monkeypatch.setitem(sys.modules, "task_api", api(name))
    return load(TASKS[name] / "scripts/evaluate.py", f"eval_{name}")


@pytest.mark.parametrize("name,count", [("wvsplitk", 18), ("paged_attention_large", 14), ("topk_forward", 7)])
def test_manifest_and_isolated_materialization(name, count, tmp_path, monkeypatch):
    workspace = tmp_path / name
    shutil.copytree(TASKS[name], workspace)
    spec = load_task_spec(workspace / "config.yaml", task_id=name)
    assert len(spec.actions) == 7
    harness = runner(name, monkeypatch)
    monkeypatch.setattr(harness, "ROOT", workspace)
    rows = harness.load_manifest()
    assert len(rows) == count
    assert all(row["checks"] == ["correctness", "performance"] for row in rows)
    generated = materialize_perf_helpers_in_workspace(workspace)
    assert workspace / "scripts/_aka_benchmark.py" in generated
    snapshot = snapshot_workspace_harness(workspace, task_spec=spec)
    (workspace / "workloads.json").write_text('{"cases": []}')
    with pytest.raises(Exception, match="workloads.json"):
        verify_workspace_harness(snapshot, discard_added=False)


def test_wvsplitk_restores_original_shapes_and_distinct_batch_workloads():
    rows = json.loads((TASKS["wvsplitk"] / "workloads.json").read_text())["cases"]
    original = [r for r in rows if r["test_case_id"].startswith("sig_")]
    assert len(original) == 14
    assert {tuple(r["shape"]) for r in original} == {
        (1,128,4096), (1,37984,4096), (1,512,2048), (1,16,2048),
        (1,37984,2048), (1,128,2048), (1,151936,2048), (1,256,7168),
        (1,16160,7168), (1,64,7168), (1,5120,2880), (1,2880,4096),
        (1,128,2880), (1,201088,2880),
    }
    assert {r["params"]["tokens"] for r in rows} == {1, 2, 3, 4}
    assert len({json.dumps(r["params"], sort_keys=True) for r in rows}) == len(rows)


def test_topk_known_ties_and_bit_word_boundaries():
    module = api("topk_forward")
    p = dict(rows=1, experts=64, topk=4, distribution="ties")
    values = module.make_inputs(p, device="cpu")
    values["logits"].fill_(-10)
    values["logits"][0, [0, 31, 32, 63]] = 1
    output = module.reference(values, p)
    assert output[1].tolist() == [[0, 31, 32, 63]]
    assert output[0].tolist() == [[0.25] * 4]
    assert output[2].tolist() == [[0x80000001], [0x80000001]]
    module.extra_negative_checks(output, p)


def test_wvsplitk_reference_includes_bias():
    module = api("wvsplitk")
    p = dict(tokens=2, n=2, k=8, bias=True, dtype="bfloat16")
    values = dict(weight=torch.arange(16).reshape(2, 8).bfloat16(),
                  activation=torch.ones(2, 8).bfloat16(), bias=torch.tensor([2., 4.]).bfloat16())
    assert module.reference(values, p).tolist() == [[30., 96.], [30., 96.]]


def attention_params():
    return dict(sequences=3, query_rows=5, heads=8, kv_heads=2, head_size=128,
                block_size=16, context=17, cache_blocks=9, table_cols=2,
                ragged=True, layout="permuted")


def test_paged_reference_matches_scalar_page_gather_and_ragged_gqa():
    module = api("paged_attention_large")
    p = attention_params()
    values = module.make_inputs(p, device="cpu")
    got = module.reference(values, p)
    expected = torch.zeros_like(got)
    for seq in range(p["sequences"]):
        length = int(values["lengths"][seq])
        for head in range(p["heads"]):
            kv_head = head // (p["heads"] // p["kv_heads"])
            keys, vals = [], []
            for position in range(length):
                page = int(values["tables"][seq, position // 16])
                keys.append(values["key"][page, kv_head, :, position % 16, :].flatten().float())
                vals.append(values["value"][page, kv_head, :, position % 16].float())
            keys, vals = torch.stack(keys), torch.stack(vals)
            weights = torch.softmax(keys @ values["query"][seq, head].float() / 128**0.5, dim=0)
            expected[seq, head] = weights @ vals
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    module.extra_negative_checks(expected, p)


@pytest.mark.parametrize("name", TASKS)
def test_negative_outputs_and_readonly_input_mutation_are_rejected(name, monkeypatch):
    harness = runner(name, monkeypatch)
    p = (dict(tokens=2, n=16, k=32, bias=True, dtype="bfloat16") if name == "wvsplitk"
         else attention_params() if name == "paged_attention_large"
         else dict(rows=3, experts=64, topk=4, distribution="normal"))
    values = harness.api.make_inputs(p, device="cpu")
    output = harness.api.reference(values, p)
    harness.reject_wrong_outputs(output, p)
    snapshot = harness.immutable_snapshot(values)
    next(iter(harness.api.readonly(values).values())).flatten()[0] += 1
    with pytest.raises(AssertionError, match="modified"):
        harness.assert_unchanged(values, snapshot)


@pytest.mark.parametrize("name", ["wvsplitk", "paged_attention_large"])
def test_native_launches_have_current_stream_provenance(name):
    sources = [p for p in (TASKS[name] / "src").rglob("*") if p.is_file()]
    assert hip_source_graph_capture_policy(*sources) == (True, None)


@pytest.mark.parametrize("name", TASKS)
@pytest.mark.parametrize("cached", [False, True])
def test_measured_output_checks_reject_cached_answers(name, cached, monkeypatch):
    # A deterministic CPU timing adapter exercises replay validation; actual
    # GPU graph/Event behavior is qualified by the Docker validator separately.
    harness = runner(name, monkeypatch)
    p = (dict(tokens=2, n=16, k=32, bias=True, dtype="bfloat16") if name == "wvsplitk"
         else attention_params() if name == "paged_attention_large"
         else dict(rows=3, experts=64, topk=4, distribution="normal"))
    values = harness.api.make_inputs(p, device="cpu")
    cached_answer = harness.api.reference(values, p)
    monkeypatch.setattr(harness.api, "timing_options", lambda: {}, raising=False)

    class Replay:
        def rerun(self):
            self.prepare()
            self.outputs = self.fn()
            return self.outputs

        def rerun_ms(self):
            self.rerun()
            return 1.0

    def benchmark(fn, *, timed_run, prepare_fn, repetition, **kwargs):
        timed_run.fn, timed_run.prepare = fn, prepare_fn
        for _ in range(repetition):
            output = timed_run.rerun()
            timed_run.after_sample(output)
        return 1.0, {"benchmark_method": "CUDA_GRAPH", "benchmark_effective_repeats": 1}

    monkeypatch.setitem(sys.modules, "_aka_benchmark", SimpleNamespace(
        TimedRun=Replay, benchmark_cuda_graph_or_events=benchmark))

    def invoke(values, params):
        if cached:
            return harness.tensor_copies(cached_answer)
        return harness.api.reference(values, params)

    if cached:
        with pytest.raises(AssertionError):
            harness.measure_case(invoke, values, p)
    else:
        result = harness.measure_case(invoke, values, p)
        assert result["metadata"]["checked_timed_outputs"] == 8
        assert result["metadata"]["poisoned_replay_validated"]
