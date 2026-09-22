"""Full valid-token SiLU/FP8 contract and actual measured pair replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'silu_mul_fp8_quant'


def fp8_dtype():
    """Use the original wrapper's availability test, independently of candidate code."""
    import torch
    try:
        dtype = torch.float8_e4m3fnuz
        torch.tensor([1.0]).to(dtype)
        return dtype
    except (RuntimeError, AttributeError):
        return torch.float8_e4m3fn


def snapshot(values):
    return tuple(v.clone() for v in values)


def unchanged(values, pristine):
    import torch
    for value, saved in zip(values,pristine):
        if not torch.equal(value,saved):
            raise AssertionError('SiLU/FP8 modified read-only activations or token counts')


def reference(harness, inputs, group_size):
    y, counts = inputs
    return harness.reference_silu_mul_fp8(y.cpu(),counts.cpu(),group_size).to(y.device)


def check_outputs(result, expected, inputs, group_size):
    import torch
    y, counts = inputs
    E, T, H = expected.shape
    if not isinstance(result,tuple) or len(result) != 2:
        raise AssertionError('SiLU/FP8 must return quantized values and scales')
    q, scales = result
    for value,shape,dtype in ((q,(E,T,H),fp8_dtype()),(scales,(E,T,H//group_size),torch.float32)):
        if not isinstance(value,torch.Tensor) or (value.shape != shape or
                value.dtype != dtype or value.device != y.device):
            raise AssertionError('SiLU/FP8 output shape/dtype/device is invalid')
    valid = torch.arange(T,device=y.device)[None,:] < counts[:,None]
    # Unused token rows come from torch.empty and are outside the output contract.
    active_q = q.float()[valid]
    active_scales = scales[valid]
    if not torch.isfinite(active_q).all() or not torch.isfinite(active_scales).all() or not (active_scales>0).all():
        raise AssertionError('Valid FP8 values must be finite and scales positive finite')
    dequant = active_q * active_scales.repeat_interleave(group_size,dim=-1)
    # Preserve the original dequantization gate, extending its first-four-token
    # sample to every valid token. Do not impose a new exact-FP8/scale algorithm.
    torch.testing.assert_close(dequant,expected[valid],atol=.5,rtol=.2)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = getattr(module,SYMBOL)
        patched.append((module,original))
        diagnosed = False

        def verify(y,counts,group_size):
            inputs = (y,counts)
            pristine = snapshot(inputs)
            expected = reference(harness,pristine,group_size)
            try:
                result = original(y,counts,group_size)
                unchanged(inputs,pristine)
                check_outputs(result,expected,pristine,group_size)
                return result
            finally:
                for value,saved in zip(inputs,pristine):
                    value.copy_(saved)

        def checked(y,tokens_per_expert,group_size=128):
            nonlocal diagnosed
            import torch
            result = verify(y,tokens_per_expert,group_size)
            if not diagnosed:
                counts = torch.tensor([0,5,7],device=y.device,dtype=tokens_per_expert.dtype)
                for H,G in ((256,128),(192,64)):
                    values = torch.arange(6*14*4*H,device=y.device).reshape(6,14,4*H)
                    data = (2+(values%13)/4).to(y.dtype)[::2,::2,::2]
                    data[1,0,:] = 0  # The zero group still has a positive epsilon scale.
                    data[2,1,:H], data[2,1,H:] = -4, 4
                    verify(data,counts,G)
                diagnosed = True
            return result

        setattr(module,SYMBOL,checked)
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module,original in reversed(patched):
            setattr(module,SYMBOL,original)


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    module, group_size = state['mod'], state['group_size']
    inputs = (state['y'],state['tokens_per_expert'])
    pristine = snapshot(inputs)
    expected = reference(harness,pristine,group_size)
    original = getattr(module,SYMBOL)
    captured = None

    def collect(*args,**kwargs):
        nonlocal captured
        captured = original(*args,**kwargs)
        return captured

    def measured():
        nonlocal captured
        captured = None
        fn()
        if captured is None:
            raise AssertionError('Timed SiLU/FP8 did not return both outputs')
        return captured

    setattr(module,SYMBOL,collect)
    try:
        import torch
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured,timed_run=timed,**options)
        unchanged(inputs,pristine)
        check_outputs(timed.outputs,expected,pristine,group_size)
        y, counts = inputs
        y.mul_(-3).add_(4)
        counts.copy_(y.shape[1]-(torch.arange(len(counts),device=y.device)%y.shape[1]))
        counts[0] = 0
        replay_inputs = snapshot(inputs)
        replay_expected = reference(harness,replay_inputs,group_size)
        q,scales = timed.outputs
        q.copy_(torch.full_like(q,float('nan'),dtype=torch.float32).to(q.dtype))
        scales.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(inputs,replay_inputs)
        check_outputs(replayed,replay_expected,replay_inputs,group_size)
        return ms, {**metadata,'timed_output_checked':True,
                    'perturbed_input_replay_checked':True,'source_buffers_unchanged':True}
    finally:
        setattr(module,SYMBOL,original)
        for value,saved in zip(inputs,pristine):
            value.copy_(saved)


def install(harness):
    correctness, performance = harness.run_correctness,harness.run_performance

    def checked_correctness(*args,**kwargs):
        with checked_modules(harness):
            return correctness(*args,**kwargs)

    def checked_performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn,**kwargs: checked_benchmark(harness,benchmark,fn,**kwargs)
        try:
            return performance()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_correctness,harness.run_performance = checked_correctness,checked_performance
