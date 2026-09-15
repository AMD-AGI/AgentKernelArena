"""Unscored independent scalar state/slot controls for selective scan."""
import math
import torch
from scripts.contract_checks import InputSnapshot, check_outputs, comparator_control, ContractFailure


def reference_outputs(state,x,dt,A,B,C,D,z,reference):
    # Preserve the original scored oracle and expose its updated FP32 state.
    ref_state=state.detach().cpu().float().clone()
    values=[v.detach().cpu() if v is not None else None for v in (x,dt,A,B,C,D,z)]
    output=reference(ref_state,*values)
    return output.to(x.device),ref_state.to(state.device)


def control_cases(device):
    for mode in ('grouped','bias_softplus_tied','remapped','varlen_empty','speculative'):
        boundaries=[0,2,2,3] if mode=='varlen_empty' else ([0,2,4] if mode=='speculative' else [0,1,2])
        tokens=boundaries[-1];nseq=len(boundaries)-1;heads,dim,dstate=4,5,3
        state=torch.empty(8,heads,dim,dstate,device=device)
        for slot in range(8):
            for d in range(dim):state[slot,:,d,:]=(slot+1)*(d+1)/5
        x=torch.empty(tokens,heads,dim,device=device)
        for t in range(tokens):
            for head in range(heads):x[t,head]=(t+1)*(head+1)/4
        tied=mode=='bias_softplus_tied'
        dt=(torch.full((tokens,heads,1),-.5,device=device).expand(tokens,heads,dim) if tied
            else torch.full_like(x,.5))
        A=(torch.full((heads,1,1),-1.,device=device).expand(heads,dim,dstate) if tied
           else torch.full((heads,dim,dstate),-2*math.log(2.),device=device))
        bias=torch.full((heads,1),.5,device=device).expand(heads,dim) if tied else None
        B=torch.empty(tokens,2,dstate,device=device);B[:,0]=1.;B[:,1]=2.
        C=torch.tensor([1.,2.,3.],device=device).reshape(1,1,3).expand(tokens,2,3).contiguous()
        D=torch.tensor([.25,.5,.75,1.],device=device).reshape(heads,1).expand(heads,dim).contiguous() if tied else None
        z=torch.ones_like(x) if tied else None
        sources=None;destinations=None;accepted=None
        if mode=='remapped':sources=[2,0];destinations=[3,4]
        if mode=='varlen_empty':sources=[0,1,2];destinations=[3,4,5]
        if mode=='speculative':sources=[[0,1],[2,3]];destinations=[[4,5],[6,-1]];accepted=[2,1]
        expected_state=state.clone();expected=torch.empty_like(x)
        step=math.log(2.) if tied else .5
        # Closed scalar recurrence for A*dt=-log(2), constant B and C=[1,2,3].
        # Compute all output/state values without calling the task reference.
        for seq,(start,end) in enumerate(zip(boundaries,boundaries[1:])):
            if start==end:continue
            src=(sources[seq][accepted[seq]-1] if accepted else sources[seq]) if sources else seq
            dst=destinations[seq] if destinations else seq
            for head in range(heads):
                group=head//2
                for d in range(dim):
                    scalar=(src+1)*(d+1)/5
                    for t in range(start,end):
                        scalar=scalar*.5+(group+1)*step*(t+1)*(head+1)/4
                        value=scalar*6
                        if tied:value=(value+(t+1)*(head+1)/4*.25*(head+1))/(1+math.exp(-1.))
                        expected[t,head,d]=value
                        target=dst[t-start] if accepted else dst
                        if accepted and target!=-1:expected_state[target,head,d,:]=scalar
                    if not accepted:expected_state[dst,head,d,:]=scalar
        kwargs=dict(state=state,x=x,dt=dt,A=A,B=B,C=C,D=D,z=z,dt_bias=bias,dt_softplus=tied,
                    out=torch.full_like(x,float('nan')))
        if sources:kwargs['state_batch_indices']=torch.tensor(sources,device=device,dtype=torch.int32)
        if destinations:kwargs['dst_state_batch_indices']=torch.tensor(destinations,device=device,dtype=torch.int32)
        if mode in ('varlen_empty','speculative'):kwargs['cu_seqlens']=torch.tensor(boundaries,device=device,dtype=torch.int32)
        if accepted:kwargs['num_accepted_tokens']=torch.tensor(accepted,device=device,dtype=torch.int32)
        yield dict(name=mode,entrypoint='selective_state_update',kwargs=kwargs,
                   expected=(expected,expected_state),atol=1e-2,rtol=1e-2)


def run_controls(mod,device='cuda'):
    rows=[]
    for c in control_cases(device):
        kw=c['kwargs'];readonly=InputSnapshot({k:v for k,v in kw.items() if isinstance(v,torch.Tensor) and k not in ('state','out')})
        result=mod.selective_state_update(**kw)
        if result is not kw['out']:raise ContractFailure('Wrapper did not return supplied output')
        readonly.check()
        check_outputs((result,kw['state']),c['expected'],atol=c['atol'],rtol=c['rtol'],
                      inputs=[e[1] for e in readonly.entries])
        rows.append(dict(control=c['name'],status='PASS',scored=False,full_output_and_state=True,readonly_inputs=True))
    return rows


def reference_controls(h):
    c=next(control_cases('cpu'));kw=c['kwargs'];state=kw['state'].clone()
    # The original reference handles the dense single-step path; the manual
    # controls additionally define routing/varlen/speculative state evolution.
    output=h.reference(state[:2],kw['x'],kw['dt'],kw['A'],kw['B'],kw['C'],None,None)
    check_outputs((output,state),c['expected'],atol=c['atol'],rtol=c['rtol'])
    for c in control_cases('cpu'):
        comparator_control(c['expected'],atol=c['atol'],rtol=c['rtol'])
    return [dict(control='scalar_half_decay_grouped_full_state',status='PASS',negative_control='rejected')]
