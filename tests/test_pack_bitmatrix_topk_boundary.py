"""Regression coverage for assignments beyond the first 32 top-k slots."""

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_pack_bitmatrix"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous_directory = Path.cwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_directory)
    return module


@pytest.mark.parametrize("case_index,topk", [(10003, 33), (10004, 65)])
def test_native_controls_reject_first_32_only_candidate(monkeypatch, case_index, topk):
    controls = _load(TASK / "_upstream_controls.py", "_pack_topk_controls")
    runner = _load(TASK / "scripts/task_runner.py", "_pack_topk_runner")
    adapter = _load(TASK / "_arena_eval.py", "_pack_topk_adapter")
    monkeypatch.setitem(sys.modules, "_upstream_controls", controls)
    original_generator = controls.make_boundary_topk_ids
    monkeypatch.setattr(
        controls, "make_boundary_topk_ids",
        lambda rows, experts, count, device: original_generator(rows, experts, count, "cpu"),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    original_to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor, "to",
        lambda tensor, *args, **kwargs: tensor if args == ("cuda",)
        else original_to(tensor, *args, **kwargs),
    )

    manifest = adapter.load_manifest()
    rows = {row["params"]["case_index"]: row for row in manifest["cases"]
            if row["test_case_id"].startswith("control-upstream-")}
    assert rows[case_index]["params"]["configuration"] == [1, 64, topk]
    assert rows[case_index]["checks"] == ["correctness"]
    assert manifest["upstream_controls"] == json.loads(json.dumps(controls.EXTRA_CASES))
    assert runner.TEST_SHAPES == [(32, 8, 2), (64, 16, 2), (128, 32, 2),
                                  (256, 64, 4), (512, 64, 2)]
    assert (runner.WARMUP_ITERATIONS, runner.BENCHMARK_ITERATIONS) == (10, 100)

    assignments = original_generator(1, 64, topk, "cpu")
    assert assignments[0, :32].tolist() == [0] * 32
    assert assignments[0, -1].item() == 63
    assert controls.reference_pack_bitmatrix(assignments, 64).tolist() == [[1, 2147483648]]

    good = SimpleNamespace(pack_topk_to_bitmatrix=controls.reference_pack_bitmatrix)
    monkeypatch.setattr(runner, "load_module", lambda: good)
    assert runner.run_correctness(case_index=case_index) == (True, None)

    truncated = SimpleNamespace(
        pack_topk_to_bitmatrix=lambda ids, experts: controls.reference_pack_bitmatrix(ids[:, :32], experts)
    )
    monkeypatch.setattr(runner, "load_module", lambda: truncated)
    passed, reason = runner.run_correctness(case_index=case_index)
    assert not passed and "1 mismatched elements" in reason


@pytest.mark.parametrize("topk,experts", [(33, 64), (65, 64), (33, 70), (65, 70)])
def test_kernel_packs_extra_tiles_and_output_word_tail_on_gpu(monkeypatch, topk, experts):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    pytest.importorskip("triton")
    controls = _load(TASK / "_upstream_controls.py", "_pack_topk_gpu_controls")
    kernel = _load(TASK / "source/triton_pack_bitmatrix.py", "_pack_topk_gpu_kernel")
    assignments = torch.zeros((3, topk), dtype=torch.int16)
    assignments[0, -1] = experts - 1
    assignments[1, 0] = 31
    assignments[1, 31] = 32
    assignments[1, -1] = experts - 1
    assignments[2, :] = experts - 1
    assignments[2, -1] = 0

    actual = kernel.pack_topk_to_bitmatrix(assignments.cuda(), experts)
    expected = controls.reference_pack_bitmatrix(assignments, experts).cuda()
    assert actual.shape == (3, (experts + 31) // 32)
    assert actual.dtype == torch.uint32
    assert torch.equal(actual, expected)

    if experts == 64:
        runner = _load(TASK / "scripts/task_runner.py", "_pack_topk_gpu_runner")
        monkeypatch.setitem(sys.modules, "_upstream_controls", controls)
        passed, reason = runner.run_correctness(case_index=10003 if topk == 33 else 10004)
        assert passed, reason
