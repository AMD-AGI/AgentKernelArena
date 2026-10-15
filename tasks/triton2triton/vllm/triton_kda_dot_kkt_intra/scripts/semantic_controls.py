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
                valid_a=inter if 'intra'=='inter' else (intra and i>j)
                valid_qk=inter if 'intra'=='inter' else (intra and i>=j)
                d0=2.**(-(i-j)/16);d1=2.**(-(i-j)/32)
                if valid_a:A[0,i,h,j-start]=(4*d0+d1)*(.5 if i%2==0 else 1.)
                if valid_qk:Aqk[0,i,h,j-start]=(2*d0+2*d1)*(h+1)*.25
    yield dict(name='two_outputs_feature_decay_subblocks_and_partial_chunk',entrypoint='kda_dot_kkt_intra',
               kwargs=dict(q=q,k=k,gk=gk,beta=beta,scale=.25,chunk_size=BT),expected=(A,Aqk),atol=1e-2,rtol=1e-2)

    yield from dimension_control_cases(device)


def control_declarations():
    return [
        {'name':'batch_heads_feature_tail_chunk64','B':2,'T':67,'H':3,'K':37,'chunk_size':64},
        {'name':'batch_heads_feature_tail_chunk32','B':2,'T':35,'H':1,'K':17,'chunk_size':32},
    ]


def dimension_control_cases(device):
    # Construct pristine inputs and independent closed forms on CPU, then copy
    # once. Three sparse coordinates include K-1, with distinct feature decay.
    # No BLAS/contraction or candidate/reference call supplies expected values.
    for declaration in control_declarations():
        B,T,H,K,BT=(declaration[key] for key in ('B','T','H','K','chunk_size'))
        q=torch.zeros(B,T,H,K);k=torch.zeros_like(q);gk=torch.zeros_like(q)
        beta=torch.empty(B,T,H);A=torch.zeros(B,T,H,BT);Aqk=torch.zeros_like(A)
        features=(0,1,K-1);denominators=(16.,32.,64.)
        for batch in range(B):
            for head in range(H):
                qc=(1.+batch,2.+head,.5*(batch+head+1))
                kc=(2.+batch,.5+head,1.5)
                for row in range(T):
                    qtime=1.+(row%3)/4.;ktime=1.+(row%5)/8.
                    beta[batch,row,head]=.5 if (row+batch)%2==0 else .75
                    for position,feature in enumerate(features):
                        q[batch,row,head,feature]=qc[position]*qtime
                        k[batch,row,head,feature]=kc[position]*ktime
                        gk[batch,row,head,feature]=-row*math.log(2.)/denominators[position]
                    start=(row//BT)*BT
                    for column in range(start,min(start+BT,T)):
                        same_subblock=(row-start)//16==(column-start)//16
                        earlier_subblock=(row-start)//16>(column-start)//16
                        valid_a=earlier_subblock if 'intra'=='inter' else (same_subblock and row>column)
                        valid_qk=earlier_subblock if 'intra'=='inter' else (same_subblock and row>=column)
                        other_time=1.+(column%5)/8.
                        kk=sum((kc[f]*ktime)*(kc[f]*other_time)*2.**(-(row-column)/denominators[f]) for f in range(3))
                        qk=sum((qc[f]*qtime)*(kc[f]*other_time)*2.**(-(row-column)/denominators[f]) for f in range(3))
                        if valid_a:A[batch,row,head,column-start]=kk*float(beta[batch,row,head])
                        if valid_qk:Aqk[batch,row,head,column-start]=qk*.25
        yield dict(name=declaration['name'],entrypoint='kda_dot_kkt_intra',
            kwargs=dict(q=q.to(device),k=k.to(device),gk=gk.to(device),beta=beta.to(device),scale=.25,chunk_size=BT),
            expected=(A.to(device),Aqk.to(device)),atol=1e-2,rtol=1e-2)


def reference_controls(h):
    rows=[]
    for c in control_cases('cpu'):
        _check(h.reference(**c['kwargs']),c['expected'],c['atol'],c['rtol'])
        rows.append(dict(control=c['name'],status='PASS',negative_control='rejected'))
    return rows
