"""Independent output checks for the performance inputs; never timed or editable."""
from contextlib import contextmanager
import numpy as np
import torch


class NumericalMismatch(AssertionError):
    pass


# L and m are FP32 online-softmax state. The independent FP32 matmul oracle
# can differ slightly from Triton's FP16-input/FP32-accumulation reduction;
# 1% of L and 0.01 absolute cover that rounding without accepting absent stores.
STATS_ATOL = 1e-2
STATS_RTOL = 1e-2


def compare(actual, expected, *, atol=None, rtol=None, check_dtype=True, exact=False, equal_nan=False):
    if not isinstance(actual, torch.Tensor):
        raise TypeError('The candidate did not produce its declared tensor output')
    if actual.shape != expected.shape or actual.device != expected.device:
        raise ValueError('Candidate output shape/device violates the contract')
    if check_dtype and actual.dtype != expected.dtype:
        raise ValueError('Candidate output dtype violates the contract')
    if not equal_nan and not torch.isfinite(actual).all():
        raise ValueError('Candidate output contains nonfinite values')
    try:
        if exact:
            if not torch.equal(actual, expected):
                raise AssertionError('Exact output mismatch')
        else:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol,
                                       check_dtype=check_dtype, equal_nan=equal_nan)
    except AssertionError as exc:
        raise NumericalMismatch(str(exc)) from exc


def philox32(seed, count):
    """Counter-based Philox4x32-10, independently evaluated using NumPy integers."""
    mask = np.uint64(0xffffffff)
    c0 = np.arange(count, dtype=np.uint64)
    c1 = np.zeros(count,dtype=np.uint64); c2=c1.copy(); c3=c1.copy()
    k0=np.uint64(seed & 0xffffffff); k1=np.uint64((seed>>32)&0xffffffff)
    for _ in range(10):
        pa=c0*np.uint64(0xD2511F53); pb=c2*np.uint64(0xCD9E8D57)
        c0,c1,c2,c3=(pb>>np.uint64(32))^c1^k0,pb&mask,(pa>>np.uint64(32))^c3^k1,pa&mask
        k0=(k0+np.uint64(0x9E3779B9))&mask; k1=(k1+np.uint64(0xBB67AE85))&mask
    return c0.astype(np.uint32)


def swizzle_reference(rows, cols, group, *, dtype, device):
    expected = torch.empty((rows,cols),dtype=dtype,device=device)
    for i in range(rows):
        for j in range(cols):
            linear=i*cols+j
            first=(linear//(group*cols))*group
            width=min(group,rows-first)
            ni=first+(linear%(group*cols))%width; nj=(linear%(group*cols))//width
            expected[ni,nj]=linear
    return expected


def _cast_like(expected, actual):
    return expected.to(device=actual.device,dtype=actual.dtype)


def prepare(c, module):
    q,k,v=c['q'],c['k'],c['v'];scale=c['sm_scale']
    expected=torch.empty_like(q)
    # Preserve half matmul/scale -> float softmax -> half probabilities from the
    # original oracle, but bound the reference's temporary attention matrix size.
    for start in range(0,q.shape[-2],128):
        end=min(start+128,q.shape[-2]);scores=(q[:,:,start:end]@k.transpose(-1,-2))*scale
        mask=torch.arange(k.shape[-2],device=q.device)[None,:] > torch.arange(start,end,device=q.device)[:,None]
        scores.masked_fill_(mask,float('-inf'))
        prob=torch.softmax(scores.float(),dim=-1).to(q.dtype)
        expected[:,:,start:end]=prob@v
    return lambda result: compare(result,expected,atol=1e-2,rtol=0)


def prepare_full(c, module):
    """Independent O oracle plus FP32 causal row-max and exp-normalizer."""
    check_output = prepare(c, module)
    q, k, scale = c['q'], c['k'], c['sm_scale']
    batch, heads, length, _ = q.shape
    expected_l = torch.empty((batch, heads, length), device=q.device, dtype=torch.float32)
    expected_m = torch.empty_like(expected_l)
    k_float = k.float()
    for start in range(0, length, 128):
        end = min(start + 128, length)
        scores = (q[:, :, start:end].float() @ k_float.transpose(-1, -2)) * scale
        future = torch.arange(length, device=q.device)[None, :] > \
                 torch.arange(start, end, device=q.device)[:, None]
        scores.masked_fill_(future, float('-inf'))
        row_max = scores.max(dim=-1).values
        expected_m[:, :, start:end] = row_max
        expected_l[:, :, start:end] = torch.exp(scores - row_max[..., None]).sum(dim=-1)
    expected_l = expected_l.reshape(batch * heads, length)
    expected_m = expected_m.reshape(batch * heads, length)

    def check(result, side):
        if not isinstance(side, tuple) or len(side) != 2:
            raise TypeError('Missing actual L/m kernel output buffers')
        check_output(result)
        compare(side[0], expected_l, atol=STATS_ATOL, rtol=STATS_RTOL)
        compare(side[1], expected_m, atol=STATS_ATOL, rtol=STATS_RTOL)
    return check


def snapshot_inputs(c):
    return {name: (c[name].detach().clone(), c[name].shape, c[name].stride(),
                   c[name].dtype, c[name].device) for name in ('q', 'k', 'v')}


def check_inputs(c, saved):
    for name, (original, shape, stride, dtype, device) in saved.items():
        current = c[name]
        if (current.shape != shape or current.stride() != stride or current.dtype != dtype or
                current.device != device or not torch.equal(current, original)):
            raise AssertionError(f'Flash attention modified read-only {name}')


def restore_inputs(c, saved):
    for name, (original, *_rest) in saved.items():
        c[name].copy_(original)


@contextmanager
def observe_kernel_side_outputs(module):
    """Capture the actual L/m arguments; delegate every launch to the JIT."""
    kernel = module.flash_fwd_kernel

    class Observer:
        latest = None

        def __getitem__(self, grid):
            launch = kernel[grid]

            def observed(*args, **kwargs):
                if len(args) < 6 or not all(isinstance(arg, torch.Tensor) for arg in args[4:6]):
                    raise TypeError('Flash kernel launch did not expose L/m buffers')
                self.latest = (args[4], args[5])
                return launch(*args, **kwargs)
            return observed

    observer = Observer()
    module.flash_fwd_kernel = observer
    try:
        yield observer
    finally:
        module.flash_fwd_kernel = kernel


def check_scale_stride_control(module, device):
    """Unscored legal head/feature strides and a nondefault softmax scale."""
    base=torch.arange(2*128*128,device=device,dtype=torch.float32).reshape(1,2,128,128)
    q=(((base%17)-8)/8).to(torch.float16)[...,::2]
    k=((((base*3)%19)-9)/8).to(torch.float16)[...,::2]
    v=((((base*5)%23)-11)/8).to(torch.float16)[...,::2]
    c={'q':q,'k':k,'v':v,'sm_scale':0.35}
    snapshots=(q.clone(),k.clone(),v.clone())
    checked=prepare_full(c,module)
    with observe_kernel_side_outputs(module) as observed:
        result=module.attention(q,k,v,c['sm_scale'])
    checked(result,observed.latest)
    for live,saved in zip((q,k,v),snapshots):
        if not torch.equal(live,saved):
            raise AssertionError('Flash attention modified a control input')


def check_width_tail_controls(module, device):
    """Unscored accepted widths, independent operand strides and partial tiles."""
    for width in (16, 32, 128):
        generator=torch.Generator(device=device)
        generator.manual_seed(2026+width)
        batch,heads,length=2,3,141
        q=torch.randn((batch,heads,length,width*2),device=device,
                      dtype=torch.float16,generator=generator)[...,::2]
        k=torch.randn((batch,heads,length,width*3),device=device,
                      dtype=torch.float16,generator=generator)[...,::3]
        v=torch.randn((batch,heads,length,width*4),device=device,
                      dtype=torch.float16,generator=generator)[...,::4]
        inputs={'q':q,'k':k,'v':v,'sm_scale':0.27}
        snapshots=tuple(tensor.clone() for tensor in (q,k,v))
        checked=prepare_full(inputs,module)
        with observe_kernel_side_outputs(module) as observed:
            result=module.attention(q,k,v,inputs['sm_scale'])
        checked(result,observed.latest)
        for live,saved in zip((q,k,v),snapshots):
            if not torch.equal(live,saved):
                raise AssertionError('Flash attention modified a width/tail control input')


@contextmanager
def perturbed_inputs(c):
    """Change live operands for a bound replay; restore them on every exit."""
    names = ('q', 'k', 'v')
    saved = {name: c[name].clone() for name in names if isinstance(c[name], torch.Tensor)}
    try:
        c['q'].mul_(-1)
        c['k'].mul_(-4).add_(2)
        c['v'].mul_(-1)
        yield
    finally:
        for name, original in saved.items():
            c[name].copy_(original)


def poison_outputs(c, result, side=None):
    """Invalidate scored output buffers before checking the bound replay."""
    for output in (result, *(side or ())):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Missing scored output buffer")
        if output.dtype == torch.bool:
            output.logical_not_()
        elif output.is_floating_point():
            output.fill_(float("nan"))
        else:
            output.fill_(torch.iinfo(output.dtype).min)
