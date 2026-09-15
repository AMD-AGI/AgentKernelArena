"""Exact accept/reject semantics from pristine probabilities and timed replay."""
from contextlib import contextmanager
import inspect

FIELDS=('output','cu','draft_ids','draft_probs','target_probs','bonus','recovered','uniform','is_greedy','max_spec_len','vocab_size')


def snapshot(args):
    return tuple(value.clone() if hasattr(value,'clone') else value for value in args)


def unchanged(args,saved):
    import torch
    for value,before in zip(args[1:9],saved[1:9]):
        if value is None:
            if before is not None:raise AssertionError('Missing probability buffer')
        elif value.shape!=before.shape or value.dtype!=before.dtype or value.device!=before.device or not torch.equal(value,before):
            raise AssertionError('Rejection sampling modified a read-only input')


def restore(args,saved,*,output=False):
    for value,before in zip(args[:9] if output else args[1:9],saved[:9] if output else saved[1:9]):
        if value is not None:value.copy_(before)


def reference(saved):
    """Original scalar probability-ratio rule, extended to actual ragged/greedy API."""
    import torch
    output,cu,ids,dp,tp,bonus,recovered,uniform,greedy,max_spec,vocab=[v.cpu() if hasattr(v,'cpu') else v for v in saved]
    result=output.clone();written=torch.zeros_like(output,dtype=torch.bool)
    begin=0
    for request,end in enumerate(cu.tolist()):
        if not bool(greedy[request]):
            accepted=True
            for position,index in enumerate(range(begin,end)):
                token=int(ids[index]);draft=1. if dp is None else float(dp[index,token]);target=float(tp[index,token])
                accepted=draft>0 and target/draft>=float(uniform[index])
                result[request,position]=token if accepted else recovered[index]
                written[request,position]=True
                if not accepted:break
            if accepted:
                result[request,end-begin]=bonus[request];written[request,end-begin]=True
        begin=end
    return result.to(saved[0].device),written.to(saved[0].device)


def check_output(value,expected,template):
    import torch
    if not isinstance(value,torch.Tensor) or value.shape!=template.shape or value.dtype!=template.dtype or value.device!=template.device:
        raise AssertionError('Rejection output shape/dtype/device is invalid')
    if not torch.equal(value,expected):raise AssertionError('Rejection token IDs or untouched output cells mismatch')


def diagnostic_cases(device):
    import torch
    lengths=[0,1,3,4,2,0,3];vocab=17;total=sum(lengths)
    cu=torch.tensor(lengths,device=device,dtype=torch.int32).cumsum(0)
    ids=(torch.arange(total,device=device)%vocab).int()
    bonus=(torch.arange(7,device=device)+4).int();recovered=(ids+7)%vocab
    greedy=torch.tensor([False,True,False,False,False,True,False],device=device)
    uniform=torch.tensor([.5]*total,device=device,dtype=torch.float64)
    # Valid distributions, with exact binary ratios for accept equality,
    # immediate/late rejection, and zero draft probability (even at u=0).
    selected_dp=torch.full((total,),.5,device=device)
    selected_tp=torch.full((total,),.5,device=device)
    selected_tp[2]=.125;selected_tp[7]=.125
    selected_dp[8]=0.;uniform[8]=0.
    selected_tp[10]=.375;selected_tp[11]=.125
    for no_draft in (False,True):
        dp=((1-selected_dp)/(vocab-1))[:,None].expand(total,vocab).clone()
        tp=((1-selected_tp)/(vocab-1))[:,None].expand(total,vocab).clone()
        dp[torch.arange(total,device=device),ids.long()]=selected_dp
        tp[torch.arange(total,device=device),ids.long()]=selected_tp
        output=torch.full((7,6),-7,device=device,dtype=torch.int32)
        yield (output,cu,ids,None if no_draft else dp,tp,bonus,recovered,uniform,greedy,5,vocab)


@contextmanager
def checked_modules(harness):
    loader=harness.load_module;patched=[]
    def load():
        module=loader();original=module.rejection_random_sample;patched.append((module,original));diagnosed=False
        def verify(args):
            saved=snapshot(args);answer,_=reference(saved)
            try:
                result=original(*args);unchanged(args,saved)
                check_output(args[0],answer,saved[0]);check_output(result,answer,saved[0]);return result
            finally:restore(args,saved)
        def checked(*args):
            nonlocal diagnosed
            result=verify(args)
            if not diagnosed:
                for extra in diagnostic_cases(args[0].device):verify(extra)
                diagnosed=True
            return result
        module.rejection_random_sample=checked;return module
    harness.load_module=load
    try:yield
    finally:
        harness.load_module=loader
        for module,original in reversed(patched):module.rejection_random_sample=original


def perturb(args,saved):
    import torch
    output,cu,ids,dp,tp,bonus,recovered,uniform,greedy,max_spec,vocab=args
    output.copy_(saved[0]);ids.copy_((saved[2]+1)%vocab)
    if dp is not None:dp.copy_(saved[3].roll(2,dims=1))
    tp.copy_(saved[4].roll(1,dims=1));bonus.copy_((saved[5]+3)%vocab);recovered.copy_((saved[6]+5)%vocab)
    uniform.copy_(1-saved[7]);greedy.copy_(torch.arange(len(greedy),device=greedy.device)%3==0)
    lengths=torch.cat((saved[1][:1],saved[1][1:]-saved[1][:-1]));lengths[0]-=1;lengths[-1]+=1
    cu.copy_(lengths.cumsum(0))


def checked_benchmark(harness,benchmark,fn,**options):
    state=inspect.getclosurevars(fn).nonlocals
    args=tuple(state[k] for k in FIELDS);saved=snapshot(args);answer,_=reference(saved)
    module=state['mod'];original=module.rejection_random_sample;returned=[]
    def record(*values):
        result=original(*values);returned[:]=[result];return result
    def measured():fn();return returned[0]
    module.rejection_random_sample=record
    try:
        timed=harness._TimedRun();ms,metadata=benchmark(measured,timed_run=timed,**options)
        unchanged(args,saved);check_output(args[0],answer,saved[0]);check_output(timed.outputs,answer,saved[0])
        perturb(args,saved);replay=snapshot(args);answer,written=reference(replay)
        args[0][written]=-999
        result=timed.rerun()
        unchanged(args,replay);check_output(args[0],answer,saved[0]);check_output(result,answer,saved[0])
        return ms,{**metadata,'timed_output_checked':True,'perturbed_input_replay_checked':True,'source_buffers_unchanged':True}
    finally:
        module.rejection_random_sample=original;restore(args,saved,output=True)


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
