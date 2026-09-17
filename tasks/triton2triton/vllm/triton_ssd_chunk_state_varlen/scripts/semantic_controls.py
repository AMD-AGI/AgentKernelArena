"""Unscored independent known answers for the public SSD operator contract.

The five historical performance cases remain unchanged. These small, structured
inputs make omitted state, boundary, grouping or optional-output work observable.
"""
import math
import torch
from scripts.ssd_checks import InputSnapshot, check_outputs, comparator_control, ContractFailure


def _check(got, expected, atol, rtol):
    check_outputs(got, expected, atol=atol, rtol=rtol)
    comparator_control(expected, atol=atol, rtol=rtol)


def run_controls(mod, device="cuda"):
    rows = []
    for case in control_cases(device):
        kwargs = case['kwargs']
        readonly = InputSnapshot({k:v for k,v in kwargs.items()
                                  if isinstance(v, torch.Tensor) and k not in case.get('output_args', ())})
        actual = getattr(mod, case['entrypoint'])(**kwargs)
        if case.get('output_args'):
            buffer = kwargs[case['output_args'][0]]
            if case.get('returns_buffer') and actual is not buffer:
                raise ContractFailure('Public wrapper must return the supplied output buffer')
            actual = buffer
        readonly.check()
        check_outputs(actual, case['expected'], atol=case['atol'], rtol=case['rtol'],
                      inputs=[e[1] for e in readonly.entries])
        rows.append({'control':case['name'], 'status':'PASS', 'scored':False,
                     'full_output_contract':True, 'readonly_inputs':True})
    return rows


def control_cases(device):
    # Sequences [0,1),[1,3),[3,9),[9,14) exercise a start inside a
    # chunk, a same-chunk endpoint, crossing chunks and a partial final chunk.
    # With per-token decay 1/2 the closed-form local sums are 1,1.5,1,1.5.
    scale=torch.tensor([1.,2.,6.,8.],device=device)[None,:,None,None]
    for has_initial in (False,True):
        x=torch.arange(1,5,device=device)[None,:,None].expand(14,4,16).to(torch.float16).contiguous()
        B=torch.tensor([1.,2.],device=device)[None,:,None].expand(14,2,16).to(torch.float16).contiguous()
        past=(torch.tensor([10.,20.,30.,40.],device=device)[:,None,None,None]*scale).expand(4,4,16,16).contiguous()
        kwargs=dict(B=B,x=x,dt=torch.ones(4,4,4,device=device),
                    dA_cumsum=(torch.arange(1,5,device=device)*math.log(.5))[None,None,:].expand(4,4,4).contiguous(),
                    cu_seqlens=torch.tensor([0,1,3,9,14],dtype=torch.int32,device=device),chunk_states=past)
        if has_initial:
            kwargs['initial_states']=(torch.tensor([2.,4.,8.,16.],device=device)[:,None,None,None]*scale).expand(4,4,16,16).contiguous()
        values=[2.,2.5,16.,11.5] if has_initial else [1.,1.5,16.,11.5]
        expected=(torch.tensor(values,device=device)[:,None,None,None]*scale).expand(4,4,16,16).contiguous()
        yield dict(name=f'variable_lengths_partial_chunks_initial_{has_initial}',entrypoint='chunk_state_varlen',
                   kwargs=kwargs,expected=expected,atol=5e-2,rtol=5e-2)


def reference_controls(h):
    for c in control_cases('cpu'):
        k=c['kwargs']
        got=h.reference_chunk_state_varlen(k['B'],k['x'],k['dt'],k['dA_cumsum'],k['cu_seqlens'],k['chunk_states'],4,k.get('initial_states'))
        _check(got,c['expected'],5e-2,5e-2)
    return [{'control':'geometric_half_decay_varlen_with_and_without_initial','status':'PASS','negative_control':'rejected'}]
