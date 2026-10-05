"""Write bounded CPU snapshots for an unscored native/port precision diagnostic."""
import hashlib
import json
from pathlib import Path


def save(root,seed,tensors):
    import torch
    if sum(value.numel()*value.element_size() for value in tensors.values())>256<<20:raise ValueError('Targeted diagnostic exceeds256MiB; select a bounded case')
    destination=Path(root)/('seed-'+str(seed));destination.mkdir(parents=True,exist_ok=False)
    entries={}
    for name,value in tensors.items():
        if value.device.type!='cpu':raise ValueError('Diagnostic snapshots must already be on CPU')
        data=value.contiguous().view(torch.uint8).numpy().tobytes();path=destination/(name+'.bin');path.write_bytes(data)
        entries[name]={'path':path.name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'shape':list(value.shape),'dtype':str(value.dtype)}
    actual=tensors['candidate'].float();expected=tensors['native'].float();delta=(actual-expected).abs();bad=delta>(0.02+0.02*expected.abs());positions=bad.nonzero().tolist()
    q1_bad=tensors['q1'].contiguous().view(torch.uint8)!=tensors['native_q1'].contiguous().view(torch.uint8)
    s1_bad=tensors['s1']!=tensors['native_s1']
    report={'seed':seed,'arrays':entries,'mismatched_outputs':int(bad.sum()),'total_outputs':bad.numel(),'max_absolute_error':float(delta.max()),'rmse':float(delta.square().mean().sqrt()),
        'bad_output_values':[{'index':p,'candidate':float(actual[tuple(p)]),'native':float(expected[tuple(p)]),'absolute_error':float(delta[tuple(p)])} for p in positions[:100]],
        'native_repeat_max_absolute_error':float((tensors['native_repeat'].float()-expected).abs().max()),'native_repeat_mismatches':int(((tensors['native_repeat'].float()-expected).abs()>(0.02+0.02*expected.abs())).sum()),
        'input_quant_byte_mismatches':int(q1_bad.sum()),'input_scale_mismatches':int(s1_bad.sum()),
        'input_quant_mismatches_by_row':{str(i):int(n) for i,n in enumerate(q1_bad.reshape(q1_bad.shape[0],-1).sum(1)) if n},
        'native_hidden_stage_intermediates':'Native fused code-object internal gate/up and intermediate quant outputs are not exposed; native_q1/native_s1 and native final output are actual native arrays.',
        'tolerance':{'rtol':0.02,'atol':0.02},'score_claim':False}
    (destination/'ARRAYS.json').write_text(json.dumps(report,indent=2)+'\n');return report
