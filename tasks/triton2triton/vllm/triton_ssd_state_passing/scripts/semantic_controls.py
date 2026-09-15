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
    scale=(torch.arange(1,3,device=device)[:,None]*torch.arange(1,4,device=device)[None,:]).float()
    states=torch.arange(1,6,device=device)[:,None,None].float()*scale
    init=torch.tensor([8.,16.,32.],device=device)[:,None,None]*scale
    for has_initial,out_dtype in ((False,torch.float32),(True,torch.float32),(True,torch.float16)):
        expected_values=[5.,4.5,11.,9.5,21.] if has_initial else [1.,2.5,3.,5.5,5.]
        kwargs=dict(states=states.clone(),dA_cumsum=torch.full((2,5,4),math.log(.5),device=device),
                    cu_chunk_seqlens=torch.tensor([0,4,8,12,16,20],dtype=torch.int32,device=device),
                    seq_idx=torch.tensor([0,0,1,1,2],dtype=torch.int32,device=device),out_dtype=out_dtype)
        if has_initial:kwargs['initial_states']=init.clone()
        expected=(torch.tensor(expected_values,device=device)[:,None,None]*scale).to(out_dtype)
        yield dict(name=f'boundary_reset_initial_{has_initial}_{out_dtype}',entrypoint='state_passing_fwd',
                   kwargs=kwargs,expected=expected,atol=1e-2,rtol=1e-2)


def reference_controls(h):
    c=next(control_cases('cpu'));k=c['kwargs']
    _check(h.reference(k['states'],k['dA_cumsum'],k['seq_idx']),c['expected'],1e-2,1e-2)
    return [{'control':'hand_recurrence_half_decay_reset_all_chunks','status':'PASS','negative_control':'rejected'}]
