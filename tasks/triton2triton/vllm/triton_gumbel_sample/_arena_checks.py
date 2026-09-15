"""Pristine Gumbel-max oracle with independent CPU Philox and exact timed replay."""
from contextlib import contextmanager
import inspect

# Algorithm/seed layout verified against Triton v3.8.0 language/random.py:
# https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/language/random.py
# CPU known-answer tests also use Random123's independent published vectors.
FIELDS=('logits','idx_mapping','temp','seed','pos')


def philox_words(counter,key):
    import numpy as np
    mask=np.uint64(0xffffffff)
    words=[np.asarray(v,dtype=np.uint64) for v in counter]
    low,high=(np.asarray(v,dtype=np.uint64) for v in key)
    for _ in range(10):
        a=words[0]*np.uint64(0xD2511F53);b=words[2]*np.uint64(0xCD9E8D57)
        words=[(b>>np.uint64(32))^words[1]^low,b&mask,(a>>np.uint64(32))^words[3]^high,a&mask]
        low=(low+np.uint64(0x9E3779B9))&mask;high=(high+np.uint64(0xBB67AE85))&mask
    return tuple(words)


def randint(seed,offset):
    import numpy as np
    seed=np.asarray(seed,dtype=np.uint64);offset=np.asarray(offset,dtype=np.uint64);mask=np.uint64(0xffffffff)
    return philox_words((offset&mask,offset>>np.uint64(32),np.zeros_like(offset),np.zeros_like(offset)),
                        (seed&mask,seed>>np.uint64(32)))[0]


def reference(saved,apply_temperature):
    import numpy as np
    import torch
    logits,mapping,temp,seed,pos=saved
    x=logits.float().cpu().numpy().copy();mapping=mapping.cpu().numpy().astype(np.int64)
    t=temp.float().cpu().numpy()[mapping];seeds=seed.cpu().numpy()[mapping].astype(np.uint64)
    positions=pos.cpu().numpy().astype(np.uint64)
    stream=randint(seeds,positions)
    bits=randint(stream[:,None],np.arange(x.shape[1],dtype=np.uint64)[None,:])
    folded=np.where(bits>>np.uint64(31),(~bits)&np.uint64(0xffffffff),bits)
    uniform=(folded.astype(np.float32)*np.float32(4.6566127342e-10)).astype(np.float64)
    noise=(-np.log(-np.log(uniform+1e-20)+1e-20)).astype(np.float32)
    if apply_temperature:x=x/np.where(t!=0,t,np.float32(1))[:,None]
    x=np.where((t!=0)[:,None],x+noise,x)
    return torch.from_numpy(x.argmax(axis=1)).to(device=logits.device,dtype=torch.int64)


def snapshot(args):return tuple(v.clone() for v in args)


def unchanged(args,saved):
    import torch
    for value,before in zip(args,saved):
        if value.shape!=before.shape or value.dtype!=before.dtype or value.device!=before.device or not torch.equal(value,before):
            raise AssertionError('Gumbel sampling modified a read-only input')


def restore(args,saved):
    for value,before in zip(args,saved):value.copy_(before)


def check_output(value,answer):
    import torch
    if not isinstance(value,torch.Tensor) or value.shape!=answer.shape or value.dtype!=torch.int64 or value.device!=answer.device:
        raise AssertionError('Gumbel output shape/dtype/device is invalid')
    if not torch.equal(value,answer):raise AssertionError('Gumbel sample differs from pristine Philox/Gumbel-max reference')


def diagnostic_cases(device):
    import torch
    values=torch.arange(14*1031,device=device).reshape(14,1031)
    logits=((values%53)/8-4).float()[::2]
    logits[0].zero_();logits[1].fill_(-10);logits[1,-1]=20
    mapping=torch.tensor([6,0,4,2,1,5,3],device=device,dtype=torch.int32)
    temp=torch.tensor([0.,.25,2.,1.,4.,0.,.5],device=device)
    seed=torch.arange(7,device=device,dtype=torch.int64)*(2**32+17)+123
    pos=torch.tensor([2**32+17,3,5,7,11,13,19],device=device,dtype=torch.int64)
    for apply in (False,True):yield (logits,mapping,temp,seed,pos,apply)


@contextmanager
def checked_modules(harness):
    loader=harness.load_module;patched=[]
    def load():
        module=loader();original=module.gumbel_sample;patched.append((module,original));diagnosed=False
        def verify(args):
            saved=snapshot(args[:5]);answer=reference(saved,args[5])
            try:
                result=original(*args);unchanged(args[:5],saved);check_output(result,answer);return result
            finally:restore(args[:5],saved)
        def checked(*args):
            nonlocal diagnosed
            result=verify(args)
            if not diagnosed:
                for extra in diagnostic_cases(args[0].device):verify(extra)
                diagnosed=True
            return result
        module.gumbel_sample=checked;return module
    harness.load_module=load
    try:yield
    finally:
        harness.load_module=loader
        for module,original in reversed(patched):module.gumbel_sample=original


def checked_benchmark(harness,benchmark,fn,**options):
    import torch
    state=inspect.getclosurevars(fn).nonlocals;args=tuple(state[k] for k in FIELDS);saved=snapshot(args)
    answer=reference(saved,False);module=state['mod'];original=module.gumbel_sample;returned=[]
    def record(*values):
        result=original(*values);returned[:]=[result];return result
    def measured():fn();return returned[0]
    module.gumbel_sample=record
    try:
        timed=harness._TimedRun();ms,metadata=benchmark(measured,timed_run=timed,**options)
        unchanged(args,saved);check_output(timed.outputs,answer)
        logits,mapping,temp,seed,pos=args
        logits.mul_(-.5).add_(.25);mapping.copy_(saved[1].flip(0))
        temp.fill_(2.);temp[::2]=0.;seed.add_(2**32+31);pos.add_(2**32+17)
        replay=snapshot(args);answer=reference(replay,False)
        timed.outputs.fill_(-999);result=timed.rerun()
        unchanged(args,replay);check_output(result,answer)
        return ms,{**metadata,'timed_output_checked':True,'perturbed_input_replay_checked':True,'source_buffers_unchanged':True}
    finally:module.gumbel_sample=original;restore(args,saved)


def install(harness):
    correctness,performance=harness.run_correctness,harness.run_performance
    def checked_correctness(*args,**kwargs):
        with checked_modules(harness):return correctness(*args,**kwargs)
    def checked_performance():
        benchmark=harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events=lambda fn,**kwargs:checked_benchmark(harness,benchmark,fn,**kwargs)
        try:return performance()
        finally:harness._benchmark_cuda_graph_or_events=benchmark
    harness.run_correctness,harness.run_performance=checked_correctness,checked_performance
