"""Native and device regressions for assignments beyond the first 32 top-k slots."""

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from src.task_prompt import build_task_prompt
from src.task_spec import load_task_spec


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


def _assert_tile_signatures(controls, assignments, experts):
    """Every extra tile must add a bit, and different rows must stay different."""
    reference = controls.reference_pack_bitmatrix
    full = reference(assignments, experts)
    first = reference(assignments[:, :32], experts)
    assert full.shape == (5, (experts + 31) // 32)
    assert all(not torch.equal(full[row], first[row]) for row in range(5))
    assert len({tuple(row) for row in full.tolist()}) == 5
    assert assignments[:, -1].tolist() == [experts - 1 - row for row in range(5)]
    assert torch.equal(assignments[:, 0], assignments[:, 1])
    if assignments.shape[1] == 65:
        first_two = reference(assignments[:, :64], experts)
        missing_middle = reference(torch.cat((assignments[:, :32], assignments[:, 64:]), dim=1), experts)
        assert assignments[:, 32].tolist() == [33, 34, 35, 36, 37]
        assert all(not torch.equal(first_two[row], first[row]) for row in range(5))
        assert all(not torch.equal(full[row], first_two[row]) for row in range(5))
        assert all(not torch.equal(full[row], missing_middle[row]) for row in range(5))
    return full


@pytest.mark.parametrize("case_index,topk", [(10003, 33), (10004, 65)])
def test_native_controls_reject_tile_and_row_mutants(monkeypatch, case_index, topk):
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
    assert rows[case_index]["params"]["configuration"] == [5, 70, topk]
    assert rows[case_index]["params"]["source"] == "repository"
    assert rows[case_index]["checks"] == ["correctness"]
    assert all("source_revision" in rows[10000 + i]["params"] for i in range(3))
    assert manifest["upstream_controls"] == json.loads(json.dumps(controls.EXTRA_CASES))
    assert runner.TEST_SHAPES == [(32, 8, 2), (64, 16, 2), (128, 32, 2),
                                  (256, 64, 4), (512, 64, 2)]
    assert (runner.WARMUP_ITERATIONS, runner.BENCHMARK_ITERATIONS) == (10, 100)

    assignments = original_generator(5, 70, topk, "cpu")
    assert assignments.dtype == torch.int16
    assert {0, 31, 32, 63, 64, 69} <= set(assignments.flatten().tolist())
    expected = _assert_tile_signatures(controls, assignments, 70)
    assert torch.all(expected[:, 2] != 0)

    reference = controls.reference_pack_bitmatrix
    good = SimpleNamespace(pack_topk_to_bitmatrix=reference)
    monkeypatch.setattr(runner, "load_module", lambda: good)
    assert runner.run_correctness(case_index=case_index) == (True, None)

    def wrong_extra_row(ids, experts):
        wrong = ids.clone()
        wrong[1:, 32:] = ids[0, 32:]
        return reference(wrong, experts)

    mutants = {
        "first-32-only": lambda ids, experts: reference(ids[:, :32], experts),
        "wrong-extra-row": wrong_extra_row,
    }
    if topk == 65:
        mutants["missing-middle"] = lambda ids, experts: reference(
            torch.cat((ids[:, :32], ids[:, 64:]), dim=1), experts)
    for name, mutant in mutants.items():
        monkeypatch.setattr(runner, "load_module", lambda mutant=mutant: SimpleNamespace(
            pack_topk_to_bitmatrix=mutant))
        passed, reason = runner.run_correctness(case_index=case_index)
        assert not passed and "mismatched elements" in reason, (name, reason)


@pytest.mark.parametrize("topk,experts", [(33, 64), (65, 64), (33, 70), (65, 70)])
def test_kernel_packs_distinct_tiles_rows_and_output_word_tail_on_gpu(monkeypatch, topk, experts):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    pytest.importorskip("triton")
    controls = _load(TASK / "_upstream_controls.py", "_pack_topk_gpu_controls")
    kernel = _load(TASK / "source/triton_pack_bitmatrix.py", "_pack_topk_gpu_kernel")
    assignments = controls.make_boundary_topk_ids(5, experts, topk, "cpu")
    expected = _assert_tile_signatures(controls, assignments, experts)

    actual = kernel.pack_topk_to_bitmatrix(assignments.cuda(), experts)
    assert actual.shape == (5, (experts + 31) // 32)
    assert actual.dtype == torch.uint32
    assert torch.equal(actual, expected.cuda())

    if experts == 70:
        runner = _load(TASK / "scripts/task_runner.py", "_pack_topk_gpu_runner")
        monkeypatch.setitem(sys.modules, "_upstream_controls", controls)
        passed, reason = runner.run_correctness(case_index=10003 if topk == 33 else 10004)
        assert passed, reason


@pytest.mark.parametrize("case_index,layout,strides", [
    (10005, "row_padding", (4, 1)),
    (10006, "column_gaps", (4, 2)),
    (10007, "row_and_column_gaps", (7, 3)),
])
def test_native_strided_controls_reject_contiguous_addressing(
    monkeypatch, case_index, layout, strides,
):
    controls = _load(TASK / "_upstream_controls.py", "_pack_stride_controls")
    runner = _load(TASK / "scripts/task_runner.py", "_pack_stride_runner")
    adapter = _load(TASK / "_arena_eval.py", "_pack_stride_adapter")
    monkeypatch.setitem(sys.modules, "_upstream_controls", controls)
    original_generator = controls.make_strided_topk_ids
    monkeypatch.setattr(controls, "make_strided_topk_ids",
                        lambda rows, experts, count, kind, device:
                        original_generator(rows, experts, count, kind, "cpu"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to",
                        lambda tensor, *args, **kwargs: tensor if args == ("cuda",)
                        else original_to(tensor, *args, **kwargs))

    manifest = adapter.load_manifest()
    row = next(r for r in manifest["cases"] if r["params"].get("case_index") == case_index)
    assert row["params"]["configuration"] == [3, 34, 2]
    assert row["params"]["layout"] == layout
    assert row["params"]["source"] == "repository"
    assert row["checks"] == ["correctness"]
    assert len(manifest["cases"]) == 13
    ids = original_generator(3, 34, 2, layout, "cpu")
    assert ids.stride() == strides and not ids.is_contiguous()
    assert ids.tolist() == [[0, 31], [1, 32], [2, 33]]
    reference = controls.reference_pack_bitmatrix
    expected = reference(ids, 34)
    assert expected.tolist() == [[2147483649, 0], [2, 1], [4, 2]]

    good = SimpleNamespace(pack_topk_to_bitmatrix=reference)
    monkeypatch.setattr(runner, "load_module", lambda: good)
    assert runner.run_correctness(case_index=case_index) == (True, None)

    def wrong_linear_addressing(source, experts):
        wrong_view = source.as_strided(source.shape, (source.shape[1], 1))
        return reference(wrong_view, experts)

    assert not torch.equal(wrong_linear_addressing(ids, 34), expected)
    bad = SimpleNamespace(pack_topk_to_bitmatrix=wrong_linear_addressing)
    monkeypatch.setattr(runner, "load_module", lambda: bad)
    passed, reason = runner.run_correctness(case_index=case_index)
    assert not passed and "mismatched elements" in reason


@pytest.mark.parametrize("case_index,layout,strides", [
    (10005, "row_padding", (4, 1)),
    (10006, "column_gaps", (4, 2)),
    (10007, "row_and_column_gaps", (7, 3)),
])
def test_gpu_strided_inputs_reject_old_linear_load_and_pass_wrapper(
    monkeypatch, case_index, layout, strides,
):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    triton = pytest.importorskip("triton")
    controls = _load(TASK / "_upstream_controls.py", "_pack_stride_gpu_controls")
    source = _load(TASK / "source/triton_pack_bitmatrix.py", "_pack_stride_gpu_source")
    ids = controls.make_strided_topk_ids(3, 34, 2, layout, "cuda")
    assert ids.stride() == strides and not ids.is_contiguous()
    pristine = ids.clone()
    expected = controls.reference_pack_bitmatrix(ids.cpu(), 34).cuda()

    # Calling the unchanged kernel directly with a strided input reproduces
    # its former linear-addressing bug; the public wrapper must normalize it.
    old_output = torch.zeros((3, 2), device="cuda", dtype=torch.uint32)
    source.pack_bitmatrix[(triton.cdiv(3, 512),)](
        old_output, ids, 3, 2, 2, BLOCK_SIZE_M=512, BLOCK_SIZE_K=32, N_CHUNKS=1)
    assert not torch.equal(old_output, expected)
    actual = source.pack_topk_to_bitmatrix(ids, 34)
    assert torch.equal(actual, expected)
    assert torch.equal(ids, pristine)

    runner = _load(TASK / "scripts/task_runner.py", "_pack_stride_gpu_runner")
    monkeypatch.setitem(sys.modules, "_upstream_controls", controls)
    assert runner.run_correctness(case_index=case_index) == (True, None)


def test_boundary_instructions_are_in_actual_task_prompt():
    spec = load_task_spec(TASK / "config.yaml", task_id="triton2triton/vllm/triton_pack_bitmatrix")
    assert "BOUNDARY_CHECKS.md" in spec.to_mapping()["instructions"]
    instructions = (TASK / "BOUNDARY_CHECKS.md").read_text()
    prompt = build_task_prompt(spec, TASK, target_gpu="MI355X")
    assert f"Task instructions from BOUNDARY_CHECKS.md:\n{instructions}" in prompt
