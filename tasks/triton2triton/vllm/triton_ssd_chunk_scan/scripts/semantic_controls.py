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


def control_cases(device, *, regular=False):
    boundaries=[0,4,8,12] if regular else [0,3,5,9]
    length=boundaries[-1];chunk_size=4
    scale=torch.arange(1,5,device=device).float()
    for mode in (('plain',) if regular else ('plain','initial','D_head','D_dim_z_initial')):
        x=(torch.arange(1,length+1,device=device)[:,None,None]*scale[None,:,None]).expand(length,4,16).half().contiguous()
        kwargs=dict(cb=torch.ones(3,2,4,4,device=device,dtype=torch.float16),x=x,
                    dt=torch.ones(4,3,4,device=device),dA_cumsum=torch.zeros(4,3,4,device=device),
                    C=torch.full((length,2,16),1/16,device=device,dtype=torch.float16),
                    states=(torch.tensor([2.,3.,5.],device=device)[:,None,None,None]*scale[None,:,None,None]).expand(3,4,16,16).contiguous(),
                    cu_chunk_seqlens=torch.tensor(boundaries,dtype=torch.int32,device=device),
                    out=torch.full((length,4,16),float('nan'),device=device),
                    seq_idx=torch.tensor([0,0,1],device=device,dtype=torch.int32))
        has_initial=mode in ('initial','D_dim_z_initial')
        if has_initial:
            kwargs['initial_states']=(torch.tensor([7.,11.],device=device)[:,None,None,None]*scale[None,:,None,None]).expand(2,4,16,16).contiguous()
        if mode=='D_head':kwargs['D']=scale.clone()
        if mode=='D_dim_z_initial':
            kwargs['D']=torch.arange(1,17,device=device).float()[None,:].expand(4,16).contiguous()/4
            kwargs['z']=torch.ones_like(x)
        # Closed-form prefix sums and scalar state contribution, independently
        # of the generic token-loop reference and the candidate implementation.
        expected=torch.empty(length,4,16,device=device)
        for c,(start,end) in enumerate(zip(boundaries,boundaries[1:])):
            prior=[7.,2.,11.][c] if has_initial else [0.,2.,0.][c]
            for t in range(start,end):
                prefix_sum=((t+1)*(t+2)-start*(start+1))/2
                for h in range(4):
                    for dim in range(16):
                        value=(prefix_sum+prior)*(h+1)
                        if mode=='D_head':value+=(t+1)*(h+1)*(h+1)
                        if mode=='D_dim_z_initial':value=(value+(t+1)*(h+1)*(dim+1)/4)/(1+math.exp(-1))
                        expected[t,h,dim]=value
        yield dict(name='causal_grouped_sequence_boundary_'+mode,entrypoint='chunk_scan_fwd',kwargs=kwargs,
                   expected=expected,atol=5e-2,rtol=5e-2,output_args=('out',))


def reference_controls(h):
    c=next(control_cases('cpu',regular=True));k=c['kwargs']
    got=h.reference_chunk_scan(k['cb'],k['x'],k['dt'],k['dA_cumsum'],k['C'],k['states'],k['seq_idx'],4)
    _check(got,c['expected'],5e-2,5e-2)
    return [{'control':'closed_form_causal_prefix_and_boundary_state','status':'PASS','negative_control':'rejected'}]
