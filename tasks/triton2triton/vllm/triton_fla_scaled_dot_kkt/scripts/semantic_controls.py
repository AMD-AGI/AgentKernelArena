"""Unscored independent known answers for the public operator contract.

The five historical performance cases remain unchanged. These small, structured
inputs make omitted state, boundary, grouping or optional-output work observable.
"""
import math
import torch
from scripts.contract_checks import InputSnapshot, check_outputs, comparator_control, ContractFailure


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
    # Two unit features give dot(k_i,k_j)=2; remaining features are zero.
    # beta depends on row, and exp(g_i-g_j)=2**(-(i-j)). Both roles of
    # beta/decay are visible across the lower triangle and the partial chunk.
    for gated in (False, True):
        k=torch.zeros(1,20,2,32,device=device)
        k[...,0:2]=1.
        beta=torch.tensor([.5 if i%2==0 else 1. for i in range(20)],device=device).reshape(1,20,1).expand(1,20,2).contiguous()
        g=torch.tensor([-i*math.log(2.) for i in range(20)],device=device).reshape(1,20,1).expand(1,20,2).contiguous() if gated else None
        expected=torch.zeros(1,20,2,16,device=device)
        for i in range(20):
            start=(i//16)*16
            for j in range(start,i):
                expected[0,i,:,j-start]=(.5 if i%2==0 else 1.)*2*(2.**(-(i-j)) if gated else 1.)
        yield dict(name='row_beta_decay_strict_triangle_partial_chunk' if gated else 'ungated_row_beta_strict_triangle_partial_chunk',
                   entrypoint='chunk_scaled_dot_kkt_fwd',kwargs=dict(k=k,beta=beta,g=g,chunk_size=16),
                   expected=expected,atol=1e-2,rtol=1e-2)


def reference_controls(h):
    rows=[]
    for c in control_cases('cpu'):
        _check(h.reference(**c['kwargs']),c['expected'],c['atol'],c['rtol'])
        rows.append(dict(control=c['name'],status='PASS',negative_control='rejected'))
    return rows
