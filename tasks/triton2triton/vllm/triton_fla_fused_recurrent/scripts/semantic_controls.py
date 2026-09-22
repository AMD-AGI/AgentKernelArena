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


def reference_outputs(q,k,v,g,beta,scale,initial_state=None):
    """Independent state recurrence, including every returned intermediate state.

    Runs outside timing; the original task reference still checks all scored
    output values separately. The public 'final_state' actually contains T
    states per batch, so comparing only the last state would omit its contract.
    """
    _device=q.device
    batch,time,heads,keys=q.shape;values=v.shape[-1]
    state=(initial_state.detach().cpu().double().clone() if initial_state is not None
           else torch.zeros(batch,heads,values,keys,dtype=torch.float64))
    q,k,v,g,beta=(x.detach().cpu().double() for x in (q,k,v,g,beta))
    outputs=[];states=[]
    for t in range(time):
        decayed=state*g[:,t].exp()[...,None,None]
        residual=v[:,t]-(decayed*k[:,t,:,None,:]).sum(-1)
        state=decayed+(residual*beta[:,t,:,None])[...,None]*k[:,t,:,None,:]
        outputs.append((state*q[:,t,:,None,:]*scale).sum(-1))
        states.append(state.clone())
    return (torch.stack(outputs,dim=1).to(dtype=torch.float32,device=initial_state.device if initial_state is not None else _device),
            torch.stack(states,dim=1).reshape(batch*time,heads,values,keys).to(dtype=torch.float32,device=initial_state.device if initial_state is not None else _device))


def control_cases(device):
    # q=k=e0, beta=1/2 and decay=1/2 imply h0' = h0/4 + v/2.
    # An independent second state column only decays; q does not observe it,
    # so requiring the entire returned history catches state-only corruption.
    for initialized in (False,True):
        q=torch.zeros(2,3,2,16,device=device);q[...,0]=1.;k=q.clone()
        v=torch.empty(2,3,2,16,device=device)
        h0=torch.zeros(2,2,16,16,device=device) if initialized else None
        o=torch.empty_like(v);history=torch.zeros(2,3,2,16,16,device=device)
        for batch in range(2):
            for head in range(2):
                for d in range(16):
                    factor=(batch+1)*(head+1)*(d+1)/16
                    first=4*factor if initialized else 0.
                    second=8*factor if initialized else 0.
                    if initialized:h0[batch,head,d,0]=first;h0[batch,head,d,1]=second
                    for t in range(3):
                        value=(t+1)*factor;v[batch,t,head,d]=value
                        first=first/4+value/2;second/=2
                        o[batch,t,head,d]=first*.5
                        history[batch,t,head,d,0]=first;history[batch,t,head,d,1]=second
        yield dict(name='nonzero_initial_and_full_state_history' if initialized else 'zero_initial_full_state_history',
                   entrypoint='fused_recurrent_gated_delta_rule_fwd',
                   kwargs=dict(q=q,k=k,v=v,g=torch.full((2,3,2),-math.log(2.),device=device),
                               beta=torch.full((2,3,2),.5,device=device),scale=.5,initial_state=h0),
                   expected=(o,history.reshape(6,2,16,16)),atol=5e-2,rtol=5e-2)


def reference_controls(h):
    rows=[]
    for c in control_cases('cpu'):
        _check(h.reference(**c['kwargs']),c['expected'][0],c['atol'],c['rtol'])
        _check(reference_outputs(**c['kwargs']),c['expected'],c['atol'],c['rtol'])
        rows.append(dict(control=c['name'],status='PASS',negative_control='rejected'))
    return rows
