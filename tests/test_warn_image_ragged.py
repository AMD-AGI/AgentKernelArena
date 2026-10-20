"""CPU checks for per-sequence ragged inputs; GPU qualification is separate."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


TASK = Path(__file__).resolve().parents[1] / "tasks/image_kernel/mi300x_sglang_hip_pa_ragged"


def load(name):
    spec = importlib.util.spec_from_file_location("ragged_test_" + name, TASK / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lengths_require_exact_cardinality_and_declared_maximum():
    h = load("task_runner")
    assert h._context_lengths(17, 3) == [17, 17, 17]
    assert h._context_lengths(17, 3, [1, 16, 17]) == [1, 16, 17]
    for bad in ([1, 17], [1, 0, 17], [1, 16, 18], [1, 15, 16], [True, 16, 17], [1., 16, 17]):
        with pytest.raises(ValueError):
            h._context_lengths(17, 3, bad)


def test_manifest_preserves_originals_and_binds_added_lengths(tmp_path, monkeypatch):
    h, adapter = load("task_runner"), load("task_adapter")
    data = json.loads((TASK / "workloads.json").read_text())
    adapter.validate_workloads(h)
    assert len(h.CASES) == len(h.PERF_CASES) == 2
    assert [r["test_case_id"] for r in data["cases"][:4]] == [
        "correctness-0", "correctness-1", "pa_ragged_ctx128_s128", "pa_ragged_ctx4097_s128"]
    for _, cfg in h.EXTRA_PERF_CASES:
        lengths = cfg["context_lengths"]
        assert len(lengths) == cfg["num_seqs"] and max(lengths) == cfg["ctx_lens"]
        assert {1, 15, 16, 17, 255, 256, 257, 4097} <= set(lengths)
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    altered = deepcopy(data)
    altered["cases"][-1]["params"]["context_lengths"][0] = 4097
    (tmp_path / "workloads.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="identities"):
        adapter.validate_workloads(h)


def test_builder_matches_variable_page_and_reference_indices(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("einops")
    h = load("task_runner")
    real_randperm = torch.randperm
    monkeypatch.setattr(torch, "set_default_device", lambda device: None)
    monkeypatch.setattr(torch, "randperm", lambda n, **kw: real_randperm(n, **{**kw, "device": "cpu"}))

    def cache_factory(blocks, block_size, layers, heads, head_size, cache_dtype, dtype, seed, device):
        assert layers == 1 and device == "cuda:0"
        return ([torch.zeros((blocks, heads, head_size // 16, block_size, 16), dtype=dtype)],
                [torch.zeros((blocks, heads, head_size, block_size), dtype=dtype)])

    monkeypatch.setitem(sys.modules, "csrc.cpp_itfs.pa", SimpleNamespace(
        pa_ragged_test=SimpleNamespace(kv_cache_factory=cache_factory)))
    cfg = h.EXTRA_CASES[0]
    case = h._make_case(**cfg)
    lengths = cfg["context_lengths"]
    assert case["seq_lens"].tolist() == lengths
    assert case["params"]["context_lengths"] == lengths
    counts = [(length + 15) // 16 for length in lengths]
    expected_indptr = [0]
    for count in counts:
        expected_indptr.append(expected_indptr[-1] + count)
    assert case["kv_indptr"].tolist() == expected_indptr
    assert case["kv_last_page_lens"].tolist() == [(length - 1) % 16 + 1 for length in lengths]
    for index, count in enumerate(counts):
        actual = case["kv_page_indices"][expected_indptr[index]:expected_indptr[index + 1]]
        torch.testing.assert_close(actual, case["block_tables"][index, :count], atol=0, rtol=0)
    assert case["max_seq_len"] == 129 and case["max_num_partitions"] == 1


def test_all_original_and_extra_shapes_are_numerically_checked(monkeypatch):
    torch = pytest.importorskip("torch")
    h = load("task_runner")
    seen = []
    monkeypatch.setattr(h, "_make_case", lambda **cfg: seen.append(cfg) or cfg)
    monkeypatch.setattr(h, "_run_aiter", lambda case: torch.tensor([float(case["ctx_lens"])]))
    monkeypatch.setattr(h, "_run_torch", lambda case: torch.tensor([float(case["ctx_lens"])]))
    h.run_correctness()
    assert seen == [*h.CASES, *[cfg for _, cfg in [*h.PERF_CASES, *h.EXTRA_PERF_CASES]], *h.EXTRA_CASES]
    monkeypatch.setattr(h, "_run_aiter", lambda case: torch.tensor([0.]) if "context_lengths" in case else torch.tensor([float(case["ctx_lens"])]))
    with pytest.raises(AssertionError):
        h.run_correctness()
