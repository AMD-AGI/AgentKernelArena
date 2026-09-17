"""Pristine MoE LoRA routing/weights and actual prepared shrink-expand replay."""
from contextlib import contextmanager


def clone(value):
    if isinstance(value,list):return [clone(v) for v in value]
    return value.clone() if hasattr(value,'clone') else value


def capture(args,options=None):
    options={} if options is None else options
    return dict(args=tuple(args[:14]),saved=tuple(clone(v) for v in args[:14]),
                weight_objects=(tuple(args[2]),tuple(args[3])),
                mul=args[14] if len(args)>14 else options.get('mul_routed_weight',False),
                offset=args[15] if len(args)>15 else options.get('offset',0))


def tensors(values):
    import torch
    for value in values:
        if isinstance(value,torch.Tensor):yield value
        elif isinstance(value,(list,tuple)):yield from tensors(value)


def equal(value,before):
    import torch
    if value.shape!=before.shape or value.dtype!=before.dtype or value.device!=before.device or not torch.equal(value,before):
        raise AssertionError('MoE LoRA modified a read-only tensor')


def unchanged(case):
    for values,before in zip(case['args'][2:4],case['weight_objects']):
        if len(values)!=len(before) or any(a is not b for a,b in zip(values,before)):
            raise AssertionError('MoE LoRA modified a weight list')
    for value,before in zip(tensors(case['args'][1:]),tensors(case['saved'][1:])):equal(value,before)


def restore(case,*,output=False):
    for values,before in zip(case['args'][2:4],case['weight_objects']):values[:]=before
    for value,before in zip(tensors(case['args'][0 if output else 1:]),tensors(case['saved'][0 if output else 1:])):value.copy_(before)


def flat_experts(saved):
    import torch
    _,_,_,_,weights,sorted_ids,experts,padded,mapping,rank,topk,lora_ids,active,_=saved
    if sorted_ids is None:return experts
    result=torch.full((weights.shape[0]*topk,),-1,device=experts.device,dtype=experts.dtype)
    # Sorted routing explicitly groups each adapter/expert into 64-token blocks.
    # Convert those protected routing tables into per-route expert identities
    # for the original independent unsorted oracle.
    for lid in lora_ids[:active].tolist():
        if lid<0:continue
        for block in range(int(padded[lid])//64):
            expert=int(experts[lid,block])
            if expert<0:continue
            indices=sorted_ids[lid,block*64:(block+1)*64]
            indices=indices[(indices>=0)&(indices<len(result))].long()
            result[indices]=expert
    return result


def answer(harness,case,*,reset=False):
    import torch
    s=case['saved'];out,x,wa,wb,weights,_,_,_,mapping,rank,topk,_,_,enabled=s
    experts=flat_experts(s);M=weights.shape[0];width=wb[0].shape[2];offset=case['offset']
    if not case['mul']:
        # Retain the original full-FP32 two-matmul oracle and FP16 result.
        contribution=harness.reference_fused_moe_lora(x,wa,wb,weights,experts,mapping,topk,enabled,False)
    else:
        # The weighted/down-projection path consumes flattened per-route
        # activations, matching token_mapping_factor=1 in the public wrapper.
        if x.shape[0]!=M*topk:raise ValueError('Weighted MoE LoRA needs M*top_k activation rows')
        contribution=torch.zeros_like(out[:,:,:len(wa)*width]).float()
        for token in range(M):
            lid=int(mapping[token])
            if lid<0 or not int(enabled[lid]):continue
            for route in range(topk):
                flat=token*topk+route;expert=int(experts[flat])
                if expert<0:continue
                for slice_id,(a,b) in enumerate(zip(wa,wb)):
                    inter=x[flat].float()@a[lid,expert].float().t()
                    values=inter@b[lid,expert].float().t()
                    contribution[token,route,slice_id*width:(slice_id+1)*width]=values*weights[token,route]
        contribution=contribution.to(out.dtype)
    base=torch.zeros_like(out) if reset else out.clone()
    result=base.float();result[:,:,offset:offset+len(wa)*width]+=contribution.float()
    writable=torch.zeros_like(out,dtype=torch.bool)
    for token in range(M):
        lid=int(mapping[token])
        if lid<0 or not int(enabled[lid]):continue
        for route in range(topk):
            if int(experts[token*topk+route])>=0:writable[token,route,offset:offset+len(wa)*width]=True
    return result.to(out.dtype),writable,base


def check_output(value,expected,template):
    import torch
    reference,writable,base=expected
    if not isinstance(value,torch.Tensor) or value.shape!=template.shape or value.dtype!=template.dtype or value.device!=template.device:
        raise AssertionError('MoE LoRA output shape/dtype/device is invalid')
    if not torch.isfinite(value).all():raise AssertionError('MoE LoRA output must be finite')
    torch.testing.assert_close(value.float(),reference.float(),atol=5e-2,rtol=5e-2)
    if not torch.equal(value[~writable],base[~writable]):raise AssertionError('MoE LoRA modified inactive routes or surrounding columns')


def diagnostic_cases(device):
    import torch
    M,K,R,N,NL,E,TK=83,35,19,67,3,3,2
    mapping=torch.arange(M,device=device,dtype=torch.int64)%NL;mapping[:67]=0;mapping[0]=-1
    flat=torch.arange(M*TK,device=device,dtype=torch.int64)%E;flat[:67*TK]=0;flat[-3]=-1
    lora_ids=torch.tensor([2,0,1],device=device,dtype=torch.int64);enabled=torch.tensor([1,0,1],device=device,dtype=torch.int32)
    weights=(.5+(torch.arange(M*TK,device=device).reshape(M,TK)%7)/8).float();weights[::3,0]*=-1;weights[::5,1]=0
    wa=[];wb=[]
    for slice_id in range(2):
        a=(.125+(torch.arange(NL*E*R*K,device=device).reshape(NL,E,R,K)%7+slice_id)/32).half();a[...,-3:]=2
        b=(.125+(torch.arange(NL*E*N*R,device=device).reshape(NL,E,N,R)%11+slice_id)/32).half();b[...,-3:]=2
        wa.append(a);wb.append(b)
    blocks=[];block_experts=[]
    for lid in range(NL):
        entries=[];assigned=[]
        for expert in range(E):
            ids=((mapping.repeat_interleave(TK)==lid)&(flat==expert)).nonzero().flatten().flip(0)
            for start in range(0,len(ids),64):
                chunk=ids[start:start+64];entries.extend(chunk.tolist()+[M*TK]*(64-len(chunk)));assigned.append(expert)
        blocks.append(entries);block_experts.append(assigned)
    capacity=max(map(len,blocks));sorted_ids=torch.full((NL,capacity),M*TK,device=device,dtype=torch.int64)
    sorted_experts=torch.full((NL,capacity//64),-1,device=device,dtype=torch.int64)
    padded=torch.tensor([len(v) for v in blocks],device=device,dtype=torch.int64)
    for lid in range(NL):
        sorted_ids[lid,:len(blocks[lid])]=torch.tensor(blocks[lid],device=device,dtype=torch.int64)
        sorted_experts[lid,:len(block_experts[lid])]=torch.tensor(block_experts[lid],device=device,dtype=torch.int64)
    for weighted in (False,True):
        rows=M*TK if weighted else M
        x=(.125+(torch.arange(rows*K,device=device).reshape(rows,K)%13)/32).half();x[...,-3:]=2
        if weighted:x[1::2].mul_(.25)  # Distinct per-route activations.
        for sorted_mode in (False,True):
            output=(32+(torch.arange(M*TK*(2*N+12),device=device).reshape(M,TK,2*N+12)%13)).half()
            yield (output,x,wa,wb,weights,sorted_ids if sorted_mode else None,sorted_experts if sorted_mode else flat,
                   padded if sorted_mode else None,mapping,R,TK,lora_ids,NL,enabled),dict(mul_routed_weight=weighted,offset=5)


@contextmanager
def prepared_launches(harness):
    prepare=harness.prepare_direct_launch;current=[]
    def checked_prepare(mod,*args,**options):
        case=capture(args,options);direct=None
        try:
            direct=prepare(mod,*args,**options);unchanged(case)
            readonly=[(direct[name],direct[name].clone()) for name in ('lora_a_ptrs','lora_b_ptrs')]
            intermediate=direct['intermediate'].clone()
            fused=direct['fused']
            def checked_fused():
                expected=answer(harness,case,reset=True)
                try:
                    result=fused();unchanged(case)
                    for value,before in readonly:equal(value,before)
                    check_output(direct['output'],expected,case['saved'][0]);return result
                finally:
                    restore(case)
                    for value,before in readonly:value.copy_(before)
                    direct['intermediate'].copy_(intermediate)
            direct['fused']=checked_fused
            current[:]=[(case,direct,readonly,intermediate)]
            return direct
        except BaseException:
            restore(case,output=True);raise
    harness.prepare_direct_launch=checked_prepare
    try:yield current
    finally:harness.prepare_direct_launch=prepare


@contextmanager
def checked_modules(harness):
    loader=harness.load_module;patched=[]
    def load():
        module=loader();original=module.fused_moe_lora;patched.append((module,original));diagnosed=False
        def verify(args,options):
            case=capture(args,options);expected=answer(harness,case)
            try:
                result=original(*args,**options);unchanged(case)
                check_output(args[0],expected,case['saved'][0]);return result
            finally:restore(case)
        def checked(*args,**options):
            nonlocal diagnosed
            result=verify(args,options)
            if not diagnosed:
                for extra,opts in diagnostic_cases(args[0].device):
                    verify(extra,opts)
                    harness.prepare_direct_launch(module,*extra,**opts)['fused']()
                diagnosed=True
            return result
        module.fused_moe_lora=checked;return module
    harness.load_module=load
    try:yield
    finally:
        harness.load_module=loader
        for module,original in reversed(patched):module.fused_moe_lora=original


def checked_benchmark(harness,benchmark,fn,prepared,**options):
    import torch
    case,direct,readonly,intermediate=prepared;args=case['args']
    def measured():fn();return direct['output']
    try:
        unchanged(case)
        timed=harness._TimedRun();ms,metadata=benchmark(measured,timed_run=timed,**options)
        unchanged(case)
        for value,before in readonly:equal(value,before)
        check_output(timed.outputs,answer(harness,case,reset=True),case['saved'][0])
        args[1].mul_(.5).add_(.25)
        for weight in args[2]:weight.mul_(-.25).add_(.125)
        for weight in args[3]:weight.mul_(.5).add_(-.2)
        args[4].mul_(-.5);args[6].copy_((case['saved'][6]+1)%args[2][0].shape[1])
        args[8].copy_((case['saved'][8]+1)%args[2][0].shape[0]);args[11].copy_(case['saved'][11].flip(0))
        args[13].copy_(case['saved'][13]);args[13][::2]=0
        replay=capture(args,dict(mul_routed_weight=case['mul'],offset=case['offset']))
        expected=answer(harness,replay,reset=True)
        direct['output'].fill_(float('nan'));direct['intermediate'].fill_(float('nan'))
        result=timed.rerun()  # Preserve the original output.zero_ prepare_fn.
        unchanged(replay)
        for value,before in readonly:equal(value,before)
        check_output(result,expected,case['saved'][0])
        return ms,{**metadata,'timed_output_checked':True,'perturbed_input_replay_checked':True,'source_buffers_unchanged':True}
    finally:
        for value,before in readonly:value.copy_(before)
        direct['intermediate'].copy_(intermediate);restore(case,output=True)


def install(harness):
    correctness,performance=harness.run_correctness,harness.run_performance
    def checked_correctness(*args,**kwargs):
        with prepared_launches(harness),checked_modules(harness):return correctness(*args,**kwargs)
    def checked_performance():
        benchmark=harness._benchmark_cuda_graph_or_events
        with prepared_launches(harness) as current:
            harness._benchmark_cuda_graph_or_events=lambda fn,**kwargs:checked_benchmark(harness,benchmark,fn,current[0],**kwargs)
            try:return performance()
            finally:harness._benchmark_cuda_graph_or_events=benchmark
    harness.run_correctness,harness.run_performance=checked_correctness,checked_performance
