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
    # Features [token,1] and [1,2*token] make each dot = row+2*column,
    # with a separate group multiplier. Chunk padding must be zero.
    for causal, output_dtype in ((False,None),(False,torch.float32),(True,None)):
        a=torch.zeros(5,2,16,device=device,dtype=torch.float16)
        b=torch.zeros_like(a)
        for t in range(5):
            for g in range(2):
                a[t,g,0]=(t+1)*(g+1); a[t,g,1]=g+1
                b[t,g,0]=1.; b[t,g,1]=2*(t+1)
        expected=torch.zeros(2,2,32,32,device=device,dtype=output_dtype or a.dtype)
        for c,(s,e) in enumerate(((0,3),(3,5))):
            for g in range(2):
                for i in range(s,e):
                    for j in range(s,e):
                        expected[c,g,i-s,j-s]=(g+1)*((i+1)+2*(j+1))
        yield dict(name='ragged_chunks_groups_padding_causal_'+str(causal)+'_'+str(output_dtype),entrypoint='bmm_chunk_fwd',
                   kwargs=dict(a=a,b=b,chunk_size=32,cu_chunk_seqlens=torch.tensor([0,3,5],device=device,dtype=torch.int32),
                               causal=causal,output_dtype=output_dtype),
                   expected=expected,atol=1e-1,rtol=1e-1)


def reference_controls(h):
    rows=[]
    for c in control_cases('cpu'):
        args=dict(c['kwargs']);dtype=args.pop('output_dtype')
        # The historical reference returns the input dtype. Optional output
        # dtype is a public storage contract, independent of this exact answer.
        _check(h.reference_bmm(**args),c['expected'].to(args['a'].dtype),c['atol'],c['rtol'])
        rows.append(dict(control=c['name'],status='PASS',negative_control='rejected'))
    return rows
