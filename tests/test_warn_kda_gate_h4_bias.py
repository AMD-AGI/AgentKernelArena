"""CPU controls for the optional bias at the scored H=4,D=32 geometry."""

from pathlib import Path
import subprocess
import sys

import pytest


TASK = Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/triton_kda_gate"


@pytest.mark.parametrize("ignore_h4_bias", [False, True])
def test_h4_bias_known_answer_and_shape_specific_negative(ignore_h4_bias):
    code = r'''
from types import SimpleNamespace
import torch
from scripts import task_runner as harness, semantic_controls as controls
from scripts.contract_checks import NumericalMismatch

cases = list(controls.control_cases('cpu'))
assert len(cases) == 8
assert [case['name'] for case in cases] == [
    'scalar_softplus_per_head_scale', 'bias_beta_threshold_and_tail',
    'head_dimension_tail_h1_d33', 'head_dimension_tail_h3_d65',
    'head_dimension_bias_h4_d32',
    'strided_g_h4_d32', 'strided_A_h4_d32', 'strided_g_bias_h4_d32',
]
case = cases[4]
kwargs = case['kwargs']
assert (kwargs['A'].numel(), kwargs['head_k_dim']) == (4, 32)
assert kwargs['g'].shape == (2, 3, 128)
assert kwargs['g_bias'].shape == (128,)
assert torch.all(kwargs['g_bias'] != 0)
assert torch.unique(kwargs['g_bias']).numel() > 4
torch.testing.assert_close(harness.reference(**kwargs), case['expected'], atol=.001, rtol=.001)
unbiased = harness.reference(**{**kwargs, 'g_bias': None})
assert not torch.allclose(unbiased, case['expected'], atol=.001, rtol=.001)
assert len(controls.reference_controls(harness)) == 8

def candidate(**inputs):
    if IGNORE_H4_BIAS and inputs['A'].numel() == 4 and inputs['head_k_dim'] == 32:
        inputs = {**inputs, 'g_bias': None}
    return harness.reference(**inputs)

if IGNORE_H4_BIAS:
    try:
        controls.run_controls(SimpleNamespace(fused_kda_gate=candidate), 'cpu')
    except NumericalMismatch:
        pass
    else:
        raise AssertionError('H=4-only bias omission passed the protected controls')
else:
    assert len(controls.run_controls(SimpleNamespace(fused_kda_gate=candidate), 'cpu')) == 8
'''.replace("IGNORE_H4_BIAS", repr(ignore_h4_bias))
    result = subprocess.run([sys.executable, "-c", code], cwd=TASK,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
