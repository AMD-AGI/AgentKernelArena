"""Unscored known answers for slot mapping, decay and complete cache updates."""
import math
import torch
from scripts.contract_checks import InputSnapshot, check_outputs, comparator_control, ContractFailure


def control_cases(device):
    for padded in (False,True):
        slots=[2,-1,0] if padded else [2,0]
        batch=len(slots)
        q=torch.zeros(batch,2,1,32,device=device,dtype=torch.float16);q[...,0]=2.
        k=torch.zeros_like(q);k[...,0]=1.
        v=torch.arange(1,33,device=device,dtype=torch.float16).reshape(1,1,1,32).expand(batch,2,1,32).contiguous()
        cache=torch.empty(5,2,32,32,device=device,dtype=torch.float16)
        for sid in range(5):
            for head in range(2):cache[sid,head].fill_((sid+1)*(head+1))
        expected_cache=cache.float().clone();expected_out=torch.zeros(batch,2,1,32,device=device)
        for i,sid in enumerate(slots):
            if sid<0:continue
            for head,ratio in enumerate((.5,.25)):
                expected_cache[sid,head].mul_(ratio)
                expected_cache[sid,head,0]+=torch.arange(1,33,device=device)
                expected_out[i,head,0]=2*expected_cache[sid,head,0]
        yield dict(name='padded_and_remapped_slots' if padded else 'remapped_slots_and_unused_cache',
                   entrypoint='linear_attn_decode_forward',
                   kwargs=dict(q=q,k=k,v=v,kv_caches=cache,
                               slope_rate=torch.tensor([math.log(2.),math.log(4.)],device=device),
                               slot_idx=torch.tensor(slots,device=device,dtype=torch.int32)),
                   expected=(expected_out,expected_cache),active=torch.tensor([s>=0 for s in slots],device=device),
                   atol=1e-2,rtol=1e-2)


def run_controls(mod,device='cuda'):
    rows=[]
    for c in control_cases(device):
        kw=c['kwargs'];readonly=InputSnapshot({k:v for k,v in kw.items() if k!='kv_caches'})
        output=mod.linear_attn_decode_forward(**kw)
        readonly.check()
        if (not isinstance(output,torch.Tensor) or output.shape!=c['expected'][0].shape
                or output.dtype!=kw['q'].dtype or output.device!=kw['q'].device):
            raise ContractFailure('Output shape, dtype or device differs from public contract')
        # A padded (-1) lane is explicitly not written by this public kernel.
        # Validate every active output and the entire cache, including untouched
        # slots; never reinterpret undefined padded storage as a zero promise.
        check_outputs((output[c['active']],kw['kv_caches']),
                      (c['expected'][0][c['active']],c['expected'][1]),
                      atol=c['atol'],rtol=c['rtol'],output_dtypes=(kw['q'].dtype,kw['kv_caches'].dtype),
                      inputs=[e[1] for e in readonly.entries])
        rows.append(dict(control=c['name'],status='PASS',scored=False,full_cache_checked=True,
                         active_output_checked=True,padded_output_contract='unwritten'))
    return rows


def reference_controls(h):
    c=next(control_cases('cpu'));kw=dict(c['kwargs']);kw['kv_caches']=kw['kv_caches'].float().clone()
    output=h.reference_linear_attn_decode(**kw)
    check_outputs((output,kw['kv_caches']),c['expected'],atol=c['atol'],rtol=c['rtol'])
    comparator_control(c['expected'],atol=c['atol'],rtol=c['rtol'])
    return [dict(control=c['name'],status='PASS',negative_control='rejected')]
