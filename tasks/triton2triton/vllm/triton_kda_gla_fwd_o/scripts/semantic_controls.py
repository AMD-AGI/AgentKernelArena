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
    # q=[1,2], exp(g)=[2,1/2], h=[1,3]*head/chunk/value_scale:
    # the state contribution is exactly 5*scale*head/chunk/value_scale.
    # Future attention entries are large sentinels, so omitting the causal
    # mask is observably wrong; both chunks include independently known sums.
    T,BT,K,V=20,16,32,16
    q=torch.zeros(1,T,2,K,device=device);g=torch.zeros_like(q)
    v=torch.empty(1,T,2,V,device=device)
    h=torch.zeros(1,2,2,K,V,device=device)
    A=torch.empty(1,T,2,BT,device=device)
    expected=torch.empty_like(v)
    for c in range(2):
        for hh in range(2):
            for d in range(V):
                h[0,c,hh,0,d]=(c+1)*(hh+1)*(d+1)/V
                h[0,c,hh,1,d]=3*(c+1)*(hh+1)*(d+1)/V
    for i in range(T):
        for hh in range(2):
            q[0,i,hh,:2]=torch.tensor([1.,2.],device=device)
            g[0,i,hh,:2]=torch.tensor([math.log(2.),math.log(.5)],device=device)
            for j in range(BT):A[0,i,hh,j]=1+j/16 if j<=i%BT else 99.
            for d in range(V):
                v[0,i,hh,d]=(i+1)*(d+1)/V
                inter=5*.25*(i//BT+1)*(hh+1)*(d+1)/V
                intra=sum((1+(j%BT)/16)*(j+1)*(d+1)/V for j in range(i//BT*BT,i+1))
                expected[0,i,hh,d]=inter+intra
    yield dict(name='feature_gate_state_attention_and_causal_partial_chunks',entrypoint='kda_gla_fwd_o',
               kwargs=dict(q=q,v=v,g=g,A=A,h=h,scale=.25,chunk_size=BT),expected=expected,atol=5e-2,rtol=5e-2)


def reference_controls(h):
    rows=[]
    for c in control_cases('cpu'):
        _check(h.reference(**c['kwargs']),c['expected'],c['atol'],c['rtol'])
        rows.append(dict(control=c['name'],status='PASS',negative_control='rejected'))
    return rows
