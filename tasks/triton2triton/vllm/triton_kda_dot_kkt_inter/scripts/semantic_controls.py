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
    # Only two features contribute. Their distinct decay rates distinguish
    # feature-wise gating from a scalar gate. Two outputs have different
    # diagonals: A is strict lower triangular; Aqk includes the intra diagonal.
    T,BT=37,32
    q=torch.zeros(1,T,2,32,device=device);k=torch.zeros_like(q);gk=torch.zeros_like(q)
    beta=torch.empty(1,T,2,device=device)
    A=torch.zeros(1,T,2,BT,device=device);Aqk=torch.zeros_like(A)
    for i in range(T):
        for h in range(2):
            q[0,i,h,:2]=torch.tensor([1.,2.],device=device)*(h+1)
            k[0,i,h,:2]=torch.tensor([2.,1.],device=device)
            gk[0,i,h,0]=-i*math.log(2.)/16
            gk[0,i,h,1]=-i*math.log(2.)/32
            beta[0,i,h]=.5 if i%2==0 else 1.
            start=(i//BT)*BT
            for j in range(start,min(start+BT,T)):
                inter=(i-start)//16>(j-start)//16
                intra=(i-start)//16==(j-start)//16
                valid_a=inter if 'inter'=='inter' else (intra and i>j)
                valid_qk=inter if 'inter'=='inter' else (intra and i>=j)
                d0=2.**(-(i-j)/16);d1=2.**(-(i-j)/32)
                if valid_a:A[0,i,h,j-start]=(4*d0+d1)*(.5 if i%2==0 else 1.)
                if valid_qk:Aqk[0,i,h,j-start]=(2*d0+2*d1)*(h+1)*.25
    yield dict(name='two_outputs_feature_decay_subblocks_and_partial_chunk',entrypoint='kda_dot_kkt_inter',
               kwargs=dict(q=q,k=k,gk=gk,beta=beta,scale=.25,chunk_size=BT),expected=(A,Aqk),atol=1e-2,rtol=1e-2)


def reference_controls(h):
    rows=[]
    for c in control_cases('cpu'):
        _check(h.reference(**c['kwargs']),c['expected'],c['atol'],c['rtol'])
        rows.append(dict(control=c['name'],status='PASS',negative_control='rejected'))
    return rows
