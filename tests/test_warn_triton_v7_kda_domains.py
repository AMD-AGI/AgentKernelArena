"""CPU known-answer and negative controls for variable KDA dimensions."""

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'tasks/triton2triton/vllm'


def _run(task, code):
    result = subprocess.run([sys.executable, '-c', code], cwd=TASKS / task,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('fault', [None, 'h1_d33', 'h3_d65', 'h4_bias'])
def test_kda_gate_independent_head_dimension_controls(fault):
    code = '''
from types import SimpleNamespace
import torch
from scripts import task_runner as h, semantic_controls as c
cases=list(c.control_cases('cpu'))
assert len(cases)==8
assert {(x['kwargs']['A'].numel(),x['kwargs']['head_k_dim']) for x in cases} == {(2,5),(1,33),(3,65),(4,32)}
for case in cases:
    torch.testing.assert_close(h.reference(**case['kwargs']),case['expected'],atol=.001,rtol=.001)
assert len(c.reference_controls(h))==8
fault=FAULT
def candidate(**kw):
    if fault=='h4_bias' and (kw['A'].numel(),kw['head_k_dim'])==(4,32):
        kw={**kw,'g_bias':None}
    out=h.reference(**kw)
    if fault=='h1_d33' and (kw['A'].numel(),kw['head_k_dim'])==(1,33): out[...,32]=0
    if fault=='h3_d65' and (kw['A'].numel(),kw['head_k_dim'])==(3,65): out[...,2,64]=0
    return out
if fault is None:
    assert len(c.run_controls(SimpleNamespace(fused_kda_gate=candidate),'cpu'))==8
else:
    try: c.run_controls(SimpleNamespace(fused_kda_gate=candidate),'cpu')
    except c.ContractFailure as exc: assert 'numerical mismatch' in str(exc).lower()
    else: raise AssertionError('shape-specific wrong candidate passed')
'''.replace('FAULT', repr(fault))
    _run('triton_kda_gate', code)


@pytest.mark.parametrize('fault', [None, 'wide_key_later_tile', 'wide_value_later_tile'])
def test_kda_gla_later_key_and_value_tiles_independently_checked(fault):
    code = '''
from types import SimpleNamespace
import torch
from scripts import task_runner as h, semantic_controls as c
cases=list(c.control_cases('cpu'))
assert len(cases)==6
wide={x['name']:x for x in cases if x['name'].startswith('wide_')}
assert {(x['kwargs']['q'].shape[-1],x['kwargs']['v'].shape[-1]) for x in wide.values()}=={(80,32),(32,160)}
for case in wide.values():
    torch.testing.assert_close(h.reference(**case['kwargs']),case['expected'],atol=.05,rtol=.05)
    assert not torch.allclose(torch.zeros_like(case['expected']),case['expected'],atol=.05,rtol=.05)
assert len(c.reference_controls(h))==6
fault=FAULT
def candidate(**kw):
    if fault=='wide_key_later_tile' and kw['q'].shape[-1]==80:
        missing=dict(kw)
        missing['q']=kw['q'].clone()
        missing['q'][...,64:]=0
        return h.reference(**missing)
    out=h.reference(**kw)
    if fault=='wide_value_later_tile' and kw['v'].shape[-1]==160: out[...,128:]=0
    return out
if fault is None:
    assert len(c.run_controls(SimpleNamespace(kda_gla_fwd_o=candidate),'cpu'))==6
else:
    try: c.run_controls(SimpleNamespace(kda_gla_fwd_o=candidate),'cpu')
    except c.ContractFailure as exc: assert 'numerical mismatch' in str(exc).lower()
    else: raise AssertionError('tile-specific wrong candidate passed')
'''.replace('FAULT', repr(fault))
    _run('triton_kda_gla_fwd_o', code)
