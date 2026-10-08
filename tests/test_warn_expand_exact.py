"""Exact copy semantics for the expansion task's measured output oracle."""
import importlib.util
from pathlib import Path

import pytest
import torch


def test_small_float_perturbation_is_rejected():
    path = (Path(__file__).resolve().parents[1] /
            'tasks/triton2triton/vllm/triton_expand/_arena_checks.py')
    spec = importlib.util.spec_from_file_location('expand_exact_checks', path)
    checks = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(checks)
    expected = torch.tensor([1.25, -2.5, 6.125], dtype=torch.float32)
    checks.check_output(expected.clone(), expected)
    altered = expected.clone()
    altered[1] += 0.001
    with pytest.raises(AssertionError, match='differs'):
        checks.check_output(altered, expected)
