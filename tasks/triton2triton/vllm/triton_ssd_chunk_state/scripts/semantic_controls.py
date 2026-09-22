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
    # Sum_t (t+1)*(h+1)*(g+1) over ragged chunks [0,3),[3,5): 6 and 9.
    x = (torch.arange(1,6,device=device)[:,None,None]*torch.arange(1,5,device=device)[None,:,None]).expand(5,4,16).to(torch.float16).contiguous()
    B = torch.tensor([1.,2.],device=device)[None,:,None].expand(5,2,16).to(torch.float16).contiguous()
    scale = torch.tensor([1.,2.,6.,8.],device=device)[None,:,None,None]
    expected = (torch.tensor([6.,9.],device=device)[:,None,None,None]*scale).expand(2,4,16,16).contiguous()
    for mode in ('default_fp32','allocated_fp16','supplied_buffer'):
        kwargs=dict(B=B.clone(),x=x.clone(),dt=torch.ones(4,2,4,device=device),dA_cumsum=torch.zeros(4,2,4,device=device),
                    cu_chunk_seqlens=torch.tensor([0,3,5],device=device,dtype=torch.int32))
        dtype=torch.float32 if mode=='default_fp32' else torch.float16
        extras={}
        if mode=='allocated_fp16':kwargs['states_in_fp32']=False
        if mode=='supplied_buffer':
            kwargs['states']=torch.full((2,4,16,16),float('nan'),device=device,dtype=dtype)
            extras=dict(output_args=('states',),returns_buffer=True)
        yield dict(name='ragged_grouped_outer_sum_'+mode,entrypoint='chunk_state_fwd',kwargs=kwargs,
                   expected=expected.to(dtype),atol=5e-1,rtol=1e-1,**extras)


def reference_controls(h):
    c=next(control_cases('cpu'));k=c['kwargs']
    got=h.reference(k['B'],k['x'],k['dt'],k['dA_cumsum'],k['cu_chunk_seqlens'])
    _check(got,c['expected'],5e-1,1e-1)
    return [{'control':'independent_integer_outer_sum_grouped_ragged','status':'PASS','negative_control':'rejected'}]
