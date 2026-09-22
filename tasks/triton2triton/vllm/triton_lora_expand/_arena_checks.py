"""Pristine LoRA operands/routing and exact measured raw-kernel replay."""
from contextlib import contextmanager
import inspect

EXPAND = True
SYMBOL = 'lora_expand' if EXPAND else 'lora_shrink'
FIELDS = ('inputs','weights','output','mapping','indices','counts','starts','ids','active')
READONLY = ('inputs','mapping','indices','counts','starts','ids')


def capture(args, options=None):
    options = {} if options is None else options
    case = dict(zip(FIELDS,args[:9]))
    case['weight_objects'] = tuple(case['weights'])
    case['saved'] = {name:case[name].clone() for name in (*READONLY,'output')}
    case['saved']['weights'] = [value.clone() for value in case['weights']]
    if EXPAND:
        case['options'] = dict(offset_start=args[9] if len(args)>9 else options.get('offset_start',0),
                               add_inputs=args[10] if len(args)>10 else options.get('add_inputs',False))
    else:
        case['options'] = dict(scaling=args[9] if len(args)>9 else options['scaling'])
    return case


def equal_input(value,saved):
    import torch
    if value.shape!=saved.shape or value.dtype!=saved.dtype or value.device!=saved.device or not torch.equal(value,saved):
        raise AssertionError('LoRA modified a read-only operand or routing table')


def unchanged(case):
    if len(case['weights'])!=len(case['weight_objects']) or any(a is not b for a,b in zip(case['weights'],case['weight_objects'])):
        raise AssertionError('LoRA modified the weight list')
    for name in READONLY:equal_input(case[name],case['saved'][name])
    for value,saved in zip(case['weights'],case['saved']['weights']):equal_input(value,saved)


def restore(case, *, output=False):
    case['weights'][:]=case['weight_objects']
    for name in READONLY:case[name].copy_(case['saved'][name])
    for value,saved in zip(case['weights'],case['saved']['weights']):value.copy_(saved)
    if output:case['output'].copy_(case['saved']['output'])


def expected(harness,case):
    s=case['saved'];ids=s['ids'][:case['active']]
    if not EXPAND:
        # Keep the original FP16-rounded reference even though the supplied
        # split-K accumulation buffer is FP32.
        return harness.reference_lora_shrink(s['inputs'],s['weights'],s['indices'],s['counts'],s['starts'],ids,case['options']['scaling'])
    widths=[w.shape[-2] for w in s['weights']]
    offset,add=case['options']['offset_start'],case['options']['add_inputs']
    if len(set(widths))==1:
        return harness.reference_lora_expand(s['inputs'],s['weights'],s['indices'],s['counts'],s['starts'],ids,offset,add,s['output'])
    # Optional heterogeneous slices use cumulative offsets in the actual
    # implementation. The original equal-width oracle remains used above.
    result=s['output'].clone().float()
    for slot,lora_id in enumerate(ids.tolist()):
        if lora_id==-1:continue
        start,count=int(s['starts'][slot]),int(s['counts'][slot])
        for token in s['indices'][start:start+count].tolist():
            column=offset
            for index,weight in enumerate(s['weights']):
                weight=weight.squeeze(1) if weight.ndim==4 else weight
                width=weight.shape[1]
                values=s['inputs'][index,token].float()@weight[lora_id].float().t()
                if add:result[token,column:column+width]+=values
                else:result[token,column:column+width]=values
                column+=width
    return result.to(s['inputs'].dtype)


def check_output(value,answer,template):
    import torch
    if not isinstance(value,torch.Tensor) or (value.shape!=template.shape or value.dtype!=template.dtype or value.device!=template.device):
        raise AssertionError('LoRA output shape/dtype/device is invalid')
    if not torch.isfinite(value).all():raise AssertionError('LoRA output must be finite')
    torch.testing.assert_close(value.float(),answer.float(),atol=5e-2,rtol=5e-2)


def diagnostic_cases(device):
    import torch
    M=83;K=19 if EXPAND else 515
    shape=(2,M,K) if EXPAND else (M,K)
    values=torch.arange(2*M*K if EXPAND else M*K,device=device).reshape(shape)
    inputs=(.25+(values%11)/16).half();inputs[...,-3:]=4 if EXPAND else 8
    widths=(17,33) if EXPAND else (19,19)
    weights=[]
    for width in widths:
        values=torch.arange(3*width*K,device=device).reshape(3,1,width,K)
        w=(.125+(values%7)/16).half();w[...,-3:]=4 if EXPAND else 8
        weights.append(w)
    counts=torch.tensor([65,13,0,5],device=device,dtype=torch.int64)
    starts=torch.tensor([0,65,78,78,83],device=device,dtype=torch.int64)
    ids=torch.tensor([2,-1,0,1],device=device,dtype=torch.int64)
    indices=torch.arange(M-1,-1,-1,device=device,dtype=torch.int64)
    mapping=torch.full((M,),-1,device=device,dtype=torch.int64)
    for slot,lid in enumerate(ids.tolist()):
        mapping[indices[int(starts[slot]):int(starts[slot+1])]]=lid
    if EXPAND:
        output=(32+(torch.arange(M*66,device=device).reshape(M,66)%13)).half()
        for add in (False,True):
            yield (inputs,weights,output.clone(),mapping,indices,counts,starts,ids,4),dict(offset_start=9,add_inputs=add)
    else:
        output=torch.full((2,M,19),2.,device=device,dtype=torch.float32)
        yield (inputs,weights,output,mapping,indices,counts,starts,ids,4,1.25),{}


@contextmanager
def checked_modules(harness):
    loader=harness.load_module;patched=[]
    def load():
        module=loader();original=getattr(module,SYMBOL);patched.append((module,original));diagnosed=False
        def verify(args,kwargs):
            case=capture(args,kwargs);answer=expected(harness,case)
            try:
                result=original(*args,**kwargs);unchanged(case)
                check_output(case['output'],answer,case['saved']['output'])
                return result
            finally:restore(case)
        def checked(*args,**kwargs):
            nonlocal diagnosed
            result=verify(args,kwargs)
            if not diagnosed:
                for extra,options in diagnostic_cases(args[0].device):verify(extra,options)
                diagnosed=True
            return result
        setattr(module,SYMBOL,checked);return module
    harness.load_module=load
    try:yield
    finally:
        harness.load_module=loader
        for module,original in reversed(patched):setattr(module,SYMBOL,original)


def tensor_leaves(value):
    import torch
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,(tuple,list)):
        for item in value:yield from tensor_leaves(item)


def replay_case(case):
    import torch
    case['inputs'].mul_(.5).add_(.5)
    for weight in case['weights']:weight.mul_(-.5).add_(.25)
    case['indices'].copy_(case['saved']['indices'].flip(0))
    case['ids'].copy_((case['saved']['ids']+1)%case['weights'][0].shape[0])
    counts=case['counts'];counts.copy_(case['saved']['counts'])
    donor=next(i for i in range(len(counts)) if int(counts[i])>0)
    recipient=(donor+1)%len(counts)
    counts[donor]-=1;counts[recipient]+=1
    case['starts'][0]=0;case['starts'][1:]=counts.cumsum(0)
    case['mapping'].fill_(-1)
    for slot,lid in enumerate(case['ids'].tolist()):
        start,end=int(case['starts'][slot]),int(case['starts'][slot+1])
        case['mapping'][case['indices'][start:end]]=lid
    data=[case[name] for name in FIELDS]
    return capture(data,case['options'])


def checked_benchmark(harness,benchmark,fn,case,**options):
    descriptors=[]
    def measured():fn();return case['output']
    try:
        unchanged(case)  # Snapshot predates the editable pointer-builder call.
        state=inspect.getclosurevars(fn).nonlocals;seen=set()
        for value in state.values():
            for tensor in tensor_leaves(value):
                if tensor is case['output'] or id(tensor) in seen:continue
                seen.add(id(tensor));descriptors.append((tensor,tensor.clone()))
        answer=expected(harness,case)
        timed=harness._TimedRun()
        ms,metadata=benchmark(measured,timed_run=timed,**options)
        unchanged(case)
        for value,saved in descriptors:equal_input(value,saved)
        check_output(timed.outputs,answer,case['saved']['output'])
        replay=replay_case(case);replay_answer=expected(harness,replay)
        replay_descriptors=[(v,v.clone()) for v,_ in descriptors]
        case['output'].fill_(float('nan'))
        replayed=timed.rerun()  # Shrink keeps its original zero_ prepare_fn.
        unchanged(replay)
        for value,saved in replay_descriptors:equal_input(value,saved)
        check_output(replayed,replay_answer,case['saved']['output'])
        return ms,{**metadata,'timed_output_checked':True,'perturbed_input_replay_checked':True,'source_buffers_unchanged':True}
    finally:
        for value,saved in descriptors:value.copy_(saved)
        restore(case,output=True)


def install(harness):
    correctness,performance=harness.run_correctness,harness.run_performance
    def checked_correctness(*args,**kwargs):
        with checked_modules(harness):return correctness(*args,**kwargs)
    def checked_performance():
        factory=harness.make_test_data;benchmark=harness._benchmark_cuda_graph_or_events;current=[]
        def make(*args,**kwargs):
            if current:restore(current[0],output=True)
            data=factory(*args,**kwargs);current[:]=[capture(data)];return data
        harness.make_test_data=make
        harness._benchmark_cuda_graph_or_events=lambda fn,**kwargs:checked_benchmark(harness,benchmark,fn,current[0],**kwargs)
        try:return performance()
        finally:
            if current:restore(current[0],output=True)
            harness.make_test_data=factory;harness._benchmark_cuda_graph_or_events=benchmark
    harness.run_correctness,harness.run_performance=checked_correctness,checked_performance
