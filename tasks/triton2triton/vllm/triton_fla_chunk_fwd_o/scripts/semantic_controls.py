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
    for gated in (False,True):
        T,BT,K,V=20,16,32,16
        q=torch.zeros(1,T,2,K,device=device);k=torch.zeros_like(q)
        v=torch.empty(1,T,2,V,device=device);h=torch.zeros(1,2,2,V,K,device=device)
        g=torch.empty(1,T,2,device=device) if gated else None
        expected=torch.empty_like(v)
        scale=.25 if gated else None;factor=scale if scale is not None else K**-.5
        for c in range(2):
            for hh in range(2):
                for d in range(V):
                    h[0,c,hh,d,0]=(c+1)*(hh+1)*(d+1)/V
                    h[0,c,hh,d,1]=2*(c+1)*(hh+1)*(d+1)/V
        for i in range(T):
            for hh in range(2):
                q[0,i,hh,:2]=1.
                k[0,i,hh,:2]=torch.tensor([.5,1.],device=device)
                if gated:g[0,i,hh]=-(i%BT)*math.log(2.)/8
                for d in range(V):
                    v[0,i,hh,d]=(i+1)*(d+1)/V
                    inter=3*(i//BT+1)*(hh+1)*(d+1)/V*(2.**(-(i%BT)/8) if gated else 1.)
                    intra=sum(1.5*(j+1)*(d+1)/V*(2.**(-(i-j)/8) if gated else 1.) for j in range(i//BT*BT,i+1))
                    expected[0,i,hh,d]=(inter+intra)*factor
        yield dict(name='gated_inter_intra_partial_chunks' if gated else 'ungated_default_scale_partial_chunks',
                   entrypoint='chunk_fwd_o',kwargs=dict(q=q,k=k,v=v,h=h,g=g,scale=scale,chunk_size=BT),
                   expected=expected,atol=5e-2,rtol=5e-2)


def reference_controls(h):
    rows=[]
    for c in control_cases('cpu'):
        _check(h.reference(**c['kwargs']),c['expected'],c['atol'],c['rtol'])
        rows.append(dict(control=c['name'],status='PASS',negative_control='rejected'))
    return rows
