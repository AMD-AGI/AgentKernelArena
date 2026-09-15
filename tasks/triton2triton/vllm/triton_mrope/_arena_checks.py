"""Validate both rotary outputs and the exact prepared raw-kernel replay."""
from contextlib import contextmanager
import inspect


def snapshot(values):
    return tuple(v.clone() for v in values)


def unchanged(values,pristine):
    import torch
    for value,saved in zip(values,pristine):
        if not torch.equal(value,saved):
            raise AssertionError('MRoPE modified a read-only input')


def reference(harness,inputs,sections,head_size,rotary_dim,interleaved=False):
    import torch
    q,k,cos,sin=inputs
    if not interleaved:
        return harness.reference_mrope(q,k,cos,sin,sections,head_size,rotary_dim)
    # Interleaving changes which T/H/W axis supplies each coefficient, not
    # the half-split rotation pairing. Enumerate H/W positions independently.
    half=rotary_dim//2
    axes=[0]*half
    for axis,count in ((1,sections[1]),(2,sections[2])):
        for ordinal in range(count):
            position=3*ordinal+axis
            if position<half:axes[position]=axis
    c=torch.stack([cos[axis,:,j] for j,axis in enumerate(axes)],dim=-1).float()[:,None,:]
    s=torch.stack([sin[axis,:,j] for j,axis in enumerate(axes)],dim=-1).float()[:,None,:]
    answers=[]
    for value in (q,k):
        shaped=value.reshape(len(value),-1,head_size)
        left=shaped[...,:half].float()
        right=shaped[...,half:rotary_dim].float()
        answer=shaped.clone()
        answer[...,:half]=(left*c-right*s).to(value.dtype)
        answer[...,half:rotary_dim]=(right*c+left*s).to(value.dtype)
        answers.append(answer.reshape_as(value))
    return tuple(answers)


def check_outputs(result,expected):
    import torch
    if not isinstance(result,tuple) or len(result)!=2:
        raise AssertionError('MRoPE must return both query and key outputs')
    for actual,wanted in zip(result,expected):
        if not isinstance(actual,torch.Tensor) or (actual.shape!=wanted.shape or
                actual.dtype!=wanted.dtype or actual.device!=wanted.device):
            raise AssertionError('MRoPE output shape/dtype/device is invalid')
        if not torch.isfinite(actual).all():
            raise AssertionError('MRoPE outputs must be finite')
        torch.testing.assert_close(actual,wanted,atol=1e-2,rtol=1e-2)


@contextmanager
def checked_modules(harness):
    loader=harness.load_module
    patched=[]

    def load():
        module=loader()
        original=module.triton_mrope
        patched.append((module,original))
        diagnosed=False

        def verify(q,k,cos,sin,sections,head_size,rotary_dim,interleaved):
            inputs=(q,k,cos,sin)
            pristine=snapshot(inputs)
            saved_sections=list(sections)
            expected=reference(harness,pristine,saved_sections,head_size,rotary_dim,interleaved)
            try:
                result=original(q,k,cos,sin,sections,head_size,rotary_dim,interleaved)
                unchanged((cos,sin),pristine[2:])
                if list(sections)!=saved_sections:
                    raise AssertionError('MRoPE modified section metadata')
                check_outputs(result,expected)
                # Scored and diagnostic buffers are contiguous: both must be
                # updated in place, as well as returned by the public wrapper.
                check_outputs((q,k),expected)
                return result
            finally:
                cos.copy_(pristine[2]);sin.copy_(pristine[3])
                sections[:]=saved_sections

        def checked(q,k,cos,sin,mrope_section,head_size,rotary_dim,mrope_interleaved):
            nonlocal diagnosed
            import torch
            result=verify(q,k,cos,sin,mrope_section,head_size,rotary_dim,mrope_interleaved)
            if not diagnosed:
                for sections,interleaved in (([7,12,13],False),([12,10,10],True)):
                    values=torch.arange(17*3*64,device=q.device).reshape(17,3*64)
                    dq=(.25+(values%17)/8).to(q.dtype)
                    values=torch.arange(17*2*64,device=q.device).reshape(17,2*64)
                    dk=(-.5+(values%13)/8).to(k.dtype)
                    values=torch.arange(3*17*32,device=q.device).reshape(3,17,32)
                    dc=(.25+(values%11)/16).to(cos.dtype)
                    ds=(-.75+(values%7)/8).to(sin.dtype)
                    verify(dq,dk,dc,ds,sections,64,64,interleaved)
                diagnosed=True
            return result

        module.triton_mrope=checked
        return module

    harness.load_module=load
    try:
        yield
    finally:
        harness.load_module=loader
        for module,original in reversed(patched):module.triton_mrope=original


def checked_benchmark(harness,benchmark,fn,**options):
    state=inspect.getclosurevars(fn).nonlocals
    prepared=inspect.getclosurevars(options['prepare_fn']).nonlocals
    inputs=(prepared['q'],prepared['k'],state['cos'],state['sin'])
    outputs=(state['q_tmp'],state['k_tmp'])
    buffers=(*inputs,*outputs)
    pristine=snapshot(buffers)
    sections=[state['mrope_t'],state['mrope_h'],state['mrope_w']]
    head_size,rotary_dim=state['head_size'],state['rotary_dim']
    expected=reference(harness,pristine[:4],sections,head_size,rotary_dim)

    def measured():
        fn()
        return outputs

    try:
        timed=harness._TimedRun()
        ms,metadata=benchmark(measured,timed_run=timed,**options)
        unchanged(inputs,pristine[:4])
        check_outputs(timed.outputs,expected)
        inputs[0].mul_(-.5).add_(.25)
        inputs[1].mul_(.75).add_(-.5)
        inputs[2].mul_(.25).add_(.25)
        inputs[3].mul_(-.5).add_(.125)
        replay_inputs=snapshot(inputs)
        replay_expected=reference(harness,replay_inputs,sections,head_size,rotary_dim)
        for output in outputs:output.fill_(float('nan'))
        # The original prepare_fn restores q_tmp/k_tmp from the perturbed
        # input seeds before replay; correctness is checked after that rotation.
        replayed=timed.rerun()
        unchanged(inputs,replay_inputs)
        check_outputs(replayed,replay_expected)
        return ms,{**metadata,'timed_output_checked':True,
                   'perturbed_input_replay_checked':True,'source_buffers_unchanged':True}
    finally:
        for value,saved in zip(buffers,pristine):value.copy_(saved)


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
