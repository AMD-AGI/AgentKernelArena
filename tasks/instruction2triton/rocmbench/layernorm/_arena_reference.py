"""Independent output checks for the performance inputs; never timed or editable."""
from contextlib import contextmanager
import numpy as np
import torch


class NumericalMismatch(AssertionError):
    pass


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
    expected=torch.nn.functional.layer_norm(c['x'],c['normalized_shape_arg'],c['w'],c['b'],c['eps'])
    xf=c['x'].float()
    expected_mean=xf.mean(-1)
    expected_rstd=torch.rsqrt(((xf-expected_mean[:,None])**2).mean(-1)+c['eps'])
    def check(result, statistics):
        compare(result,expected,atol=1e-2,rtol=1e-2)
        mean,rstd=statistics
        compare(mean,expected_mean,atol=1e-3,rtol=1e-3)
        compare(rstd,expected_rstd,atol=1e-3,rtol=1e-3)
    return check


def snapshot_inputs(c):
    return {name: (c[name].detach().clone(), c[name].shape, c[name].stride(),
                   c[name].dtype, c[name].device) for name in ('x', 'w', 'b')}


def check_inputs(c, saved):
    for name, (original, shape, stride, dtype, device) in saved.items():
        current=c[name]
        if (current.shape!=shape or current.stride()!=stride or current.dtype!=dtype or
                current.device!=device or not torch.equal(current,original)):
            raise AssertionError(f'Layer norm modified read-only {name}')


def restore_inputs(c, saved):
    for name, (original, *_rest) in saved.items():
        c[name].copy_(original)


@contextmanager
def capture_side_outputs(module):
    """Bind each returned y buffer to its own mean/rstd capture buffers.

    A graph may capture several operator calls per replay. The canonical timer
    exposes the first returned y, so storing only the last captured statistics
    would inspect a different call. Storage identity binds the exact trio.
    """
    original=module.layernorm_wrapper_fn
    statistics={}
    def key(output):
        if not isinstance(output,torch.Tensor):
            raise TypeError('Layer norm did not return a Tensor')
        return output.device,output.data_ptr(),tuple(output.shape),output.dtype
    def observed(grid,x,y,w,b,mean,rstd,*rest):
        statistics[key(y)]=(mean,rstd)
        return original(grid,x,y,w,b,mean,rstd,*rest)
    def lookup(output):
        try:
            return statistics[key(output)]
        except KeyError as exc:
            raise AssertionError('Timed layer norm output has no bound mean/rstd buffers') from exc
    def poison_all():
        for pair in statistics.values():
            poison_statistics(pair)
    lookup.poison_all=poison_all
    module.layernorm_wrapper_fn=observed
    try:
        yield lookup
    finally:
        module.layernorm_wrapper_fn=original


def poison_statistics(statistics):
    for output in statistics:
        if not isinstance(output,torch.Tensor) or output.dtype!=torch.float32:
            raise TypeError('Missing float32 layer norm statistics buffer')
        output.fill_(float('nan'))


def check_stats_stride_control(c, module):
    """Unscored row-strided output and the kernel's mean/rstd side outputs."""
    x_source=c['x']
    if x_source.ndim != 2:
        raise AssertionError('Layer norm control expects a matrix')
    rows,cols=min(7,x_source.shape[0]),min(129,x_source.shape[1])
    x_store=torch.empty((rows,cols+3),device=x_source.device,dtype=x_source.dtype)
    x=x_store[:,:cols]
    x.copy_(x_source[:rows,:cols])
    x[0].fill_(0.001)
    if rows > 1:
        x[1].zero_()
    y_store=torch.empty((rows,cols+5),device=x.device,dtype=x.dtype)
    y=y_store[:,:cols]
    mean=torch.full((rows,),float('nan'),device=x.device,dtype=torch.float32)
    rstd=torch.full_like(mean,float('nan'))
    w,b=c['w'][:cols],c['b'][:cols]
    eps=1e-3
    saved=x.clone(),w.clone(),b.clone()
    module.layernorm_wrapper_fn((rows,),x,y,w,b,mean,rstd,
                                x.stride(0),y.stride(0),rows,cols,rows,cols,eps,256)
    expected_y=torch.nn.functional.layer_norm(saved[0],(cols,),saved[1],saved[2],eps)
    compare(y,expected_y,atol=1e-2,rtol=1e-2)
    xf=saved[0].float()
    expected_mean=xf.mean(-1)
    expected_rstd=torch.rsqrt(((xf-expected_mean[:,None])**2).mean(-1)+eps)
    compare(mean,expected_mean,atol=1e-3,rtol=1e-3)
    compare(rstd,expected_rstd,atol=1e-3,rtol=1e-3)
    if any(not torch.equal(live,pristine) for live,pristine in zip((x,w,b),saved)):
        raise AssertionError('Layer norm modified a read-only control input')


@contextmanager
def perturbed_inputs(c):
    """Change live operands for a bound replay; restore them on every exit."""
    names = ('x', 'w', 'b')
    saved = {name: c[name].clone() for name in names if isinstance(c[name], torch.Tensor)}
    try:
        c['x'].mul_(-1)
        c['w'].add_(0.5)
        c['b'].add_(0.25)
        yield
    finally:
        for name, original in saved.items():
            c[name].copy_(original)


def poison_outputs(c, result, statistics):
    """Invalidate scored output buffers before checking the bound replay."""
    for output in (result,*statistics):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Missing scored output buffer")
        if output.dtype == torch.bool:
            output.logical_not_()
        elif output.is_floating_point():
            output.fill_(float("nan"))
        else:
            output.fill_(torch.iinfo(output.dtype).min)
