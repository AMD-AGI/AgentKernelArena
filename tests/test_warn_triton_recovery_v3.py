"""CPU controls for output-state replay and declared partial-tile branches."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TASKS = Path(__file__).resolve().parents[1] / "tasks"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("wrong_branch", (None, "partial_m", "partial_n", "partial_k"))
def test_block_pointer_controls_check_each_partial_dimension(wrong_branch):
    task = TASKS / "triton2triton/rocmbench/hard/test_block_pointer_matmul"
    reference = load(task / "_arena_reference.py", "block_partial_reference")
    seen = []

    class FakeKernel:
        def __getitem__(self, grid):
            assert grid == (1,)

            def launch(**kwargs):
                a, b, out = kwargs["a_ptr"], kwargs["b_ptr"], kwargs["c_ptr"]
                bm, bn, bk = (kwargs[x] for x in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
                branch = ("partial_m" if bm < kwargs["M"] else
                          "partial_n" if bn < kwargs["N"] else "partial_k")
                seen.append(branch)
                if branch != wrong_branch:
                    out[:bm, :bn] = a[:bm, :bk] @ b[:bk, :bn]

            return launch

    context = {"a": torch.empty((1, 1))}
    module = SimpleNamespace(matmul_no_scf_with_advance_kernel=FakeKernel())
    if wrong_branch is None:
        reference.check_partial_tile_controls(context, module)
        assert seen == ["partial_m", "partial_n", "partial_k"]
    else:
        with pytest.raises(reference.NumericalMismatch):
            reference.check_partial_tile_controls(context, module)
        assert seen[-1] == wrong_branch
