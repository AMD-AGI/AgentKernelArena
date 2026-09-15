"""Check cache scatter against pristine operands and the same prepared replay."""
from contextlib import contextmanager
import inspect

DIFFERENT_DIMS = True
SYMBOL = 'reshape_and_cache_flash_diffkv'


def clone(values):
    return [v.clone() if v is not None else None for v in values]


def unchanged(values, saved):
    import torch
    for actual, old in zip(values, saved):
        if actual is None:
            continue
        # Byte comparison also supports float8 tensors lacking torch.equal.
        if (actual.shape != old.shape or actual.dtype != old.dtype or actual.device != old.device or
                not torch.equal(actual.contiguous().reshape(-1).view(torch.uint8), old.contiguous().reshape(-1).view(torch.uint8))):
            raise AssertionError('Cache scatter modified a read-only input, mapping or scale')


def reference(harness, key, value, caches, slots, kv_cache_dtype='auto', k_scale=None, v_scale=None):
    import torch
    expected = clone(caches)
    head_major = not DIFFERENT_DIMS and caches[0].ndim == 5
    if kv_cache_dtype == 'auto' and not head_major:
        if DIFFERENT_DIMS:
            harness.reference_reshape_and_cache_diffkv(key, value, expected[0], slots)
        else:
            harness.reference_reshape_and_cache(key, value, *expected, slots)
        return expected
    block_size = caches[0].shape[3] if head_major else caches[0].shape[1]
    for token, slot in enumerate(slots.cpu().tolist()):
        if slot < 0: continue
        block, offset = divmod(slot, block_size)
        k, v = key[token].float(), value[token].float()
        if kv_cache_dtype.startswith('fp8'):
            if not str(key.dtype).startswith('torch.float8_'): k = k/(1. if k_scale is None else k_scale)
            if not str(value.dtype).startswith('torch.float8_'): v = v/(1. if v_scale is None else v_scale)
        if DIFFERENT_DIMS:
            expected[0][block,offset,:,:key.shape[-1]].copy_(k)
            expected[0][block,offset,:,key.shape[-1]:].copy_(v)
        elif head_major:
            expected[0][block,:,:,offset,:].copy_(k.reshape(key.shape[1], -1, caches[0].shape[-1]))
            expected[1][block,:,:,offset].copy_(v)
        else:
            expected[0][block,offset].copy_(k)
            expected[1][block,offset].copy_(v)
    return expected


def check_caches(caches, expected):
    import torch
    if not isinstance(caches, (tuple,list)) or len(caches) != len(expected):
        raise AssertionError('Missing cache output')
    for actual, answer in zip(caches,expected):
        if not isinstance(actual,torch.Tensor) or (actual.shape != answer.shape or
                actual.dtype != answer.dtype or actual.device != answer.device):
            raise AssertionError('Cache output shape/dtype/device is invalid')
        if not torch.isfinite(actual.float()).all():
            raise AssertionError('Cache output must be finite')
        torch.testing.assert_close(actual.float(),answer.float(),atol=1e-3,rtol=1e-3)


def call(function,key,value,caches,slots,**options):
    return function(key,value,*caches,slots,**options)


def diagnose(harness,original,key,slots):
    import torch
    count,heads,hk,hv = (7,3,17,25) if DIFFERENT_DIMS else (7,3,24,24)
    k=((torch.arange(count*heads*hk,device=key.device).reshape(count,heads,hk)%19-9)/4).to(key.dtype)
    v=((torch.arange(count*heads*hv,device=key.device).reshape(count,heads,hv)%13-6)/4).to(key.dtype)
    mapping=torch.tensor([5,-1,0,7,2,9,13],device=slots.device,dtype=slots.dtype)
    ks=torch.tensor(.5,device=key.device);vs=torch.tensor(2.,device=key.device)
    def caches(dtype,head_major=False):
        shapes=[(4,4,heads,hk+hv)] if DIFFERENT_DIMS else (
            [(4,heads,hk//8,4,8),(4,heads,hv,4)] if head_major else [(4,4,heads,hk),(4,4,heads,hv)])
        return [torch.full(shape,.25 if i==0 else -.25,device=key.device).to(dtype) for i,shape in enumerate(shapes)]
    def verify(kk,vv,out,**options):
        values=[kk,vv,mapping,ks,vs];saved=clone(values)
        expected=reference(harness,*saved[:2],out,saved[2],k_scale=saved[3],v_scale=saved[4],**options)
        call(original,kk,vv,out,mapping,k_scale=ks,v_scale=vs,**options)
        unchanged(values,saved);check_caches(out,expected)
    verify(k,v,caches(key.dtype))
    if not DIFFERENT_DIMS:
        verify(k,v,caches(key.dtype,head_major=True))
    # Current MI355X task environment supports this explicit FP8 cache format.
    dtype=torch.float8_e4m3fnuz
    verify(k,v,caches(dtype),kv_cache_dtype='fp8')
    verify(k.to(dtype),v.to(dtype),caches(dtype),kv_cache_dtype='fp8')


@contextmanager
def checked_modules(harness):
    loader=harness.load_module;patched=[]
    def load():
        module=loader();original=getattr(module,SYMBOL);patched.append((module,original));diagnosed=False
        def checked(key,value,*args,**options):
            nonlocal diagnosed
            caches=list(args[:1 if DIFFERENT_DIMS else 2]);slots=args[1 if DIFFERENT_DIMS else 2]
            # Preserve positional optional arguments accepted by the public API.
            extra=args[2 if DIFFERENT_DIMS else 3:]
            if len(extra)>3:raise TypeError('Too many cache options')
            for name,val in zip(('kv_cache_dtype','k_scale','v_scale'),extra):
                if name in options:raise TypeError('Duplicate cache option '+name)
                options[name]=val
            readonly=[key,value,slots,options.get('k_scale'),options.get('v_scale')];saved=clone(readonly)
            reference_options=dict(options)
            if 'k_scale' in options:reference_options['k_scale']=saved[3]
            if 'v_scale' in options:reference_options['v_scale']=saved[4]
            expected=reference(harness,saved[0],saved[1],caches,saved[2],**reference_options)
            result=call(original,key,value,caches,slots,**options)
            unchanged(readonly,saved);check_caches(caches,expected)
            if not diagnosed:diagnose(harness,original,key,slots);diagnosed=True
            return result
        setattr(module,SYMBOL,checked);return module
    harness.load_module=load
    try:yield
    finally:
        harness.load_module=loader
        for module,original in reversed(patched):setattr(module,SYMBOL,original)


def checked_benchmark(harness,benchmark,fn,**options):
    import torch
    state=inspect.getclosurevars(fn).nonlocals
    caches=[state['kv_cache']] if DIFFERENT_DIMS else [state['key_cache'],state['value_cache']]
    readonly=[state[k] for k in ('key','value','slot_mapping','k_scale','v_scale')]
    saved=clone(readonly);saved_caches=clone(caches)
    reset=options['prepare_fn'];checking_replay=False
    valid_slots=readonly[2][readonly[2]>=0].clone()
    def prepare():
        result=reset()
        if checking_replay:
            # Only the unscored rerun poisons written slots AFTER original reset.
            for cache in caches:cache.view(-1,cache.shape[-2]*cache.shape[-1]).index_fill_(0,valid_slots,float('nan'))
        return result
    def measured():fn();return tuple(caches)
    def expected():
        return reference(harness,readonly[0],readonly[1],[torch.zeros_like(c) for c in caches],readonly[2],
                         k_scale=readonly[3],v_scale=readonly[4])
    answer=expected()
    try:
        timed=harness._TimedRun()
        ms,metadata=benchmark(measured,timed_run=timed,**{**options,'prepare_fn':prepare})
        unchanged(readonly,saved);check_caches(timed.outputs,answer)
        readonly[0].mul_(-.5).add_(.125);readonly[1].mul_(.75).sub_(.25)
        readonly[2].copy_(saved[2].flip(0))
        readonly[3].mul_(2);readonly[4].mul_(.5)
        replay_saved=clone(readonly);replay_answer=expected();checking_replay=True
        result=timed.rerun()
        unchanged(readonly,replay_saved);check_caches(result,replay_answer)
        return ms,{**metadata,'timed_output_checked':True,'perturbed_input_replay_checked':True,
                   'replay_poison_after_original_prepare':True,'scored_reset_unchanged':True}
    finally:
        for value,old in zip(readonly,saved):value.copy_(old)
        for cache,old in zip(caches,saved_caches):cache.copy_(old)


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
