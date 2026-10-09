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
    x=c['x'].float(); gamma=c['g'].float()+(1 if c['ZERO_CENTERED_GAMMA'] else 0)
    rsigma=torch.rsqrt((x*x).mean(-1)+c['eps'])
    output_dtype={'fp16':torch.float16,'bf16':torch.bfloat16,'fp32':torch.float32}[c['out_dtype_str']]
    y=(x*rsigma[:,None]*gamma).to(output_dtype)
    # Match the original correctness policy, which is selected by output dtype.
    atol,rtol=(1e-3,1e-2) if output_dtype in (torch.float16,torch.bfloat16) else (1e-5,1e-5)
    def check(result):
        compare(c['y_buffer'],y,atol=atol,rtol=rtol)
        compare(c['rsigma_buffer'],rsigma,atol=atol,rtol=rtol)
    return check


def check_row_stride_control(c, module):
    """Unscored row-strided input/output and epsilon variant through the wrapper."""
    source=c['x']
    if source.ndim != 2:
        raise AssertionError('RMS norm control expects a matrix')
    rows,cols=min(7,source.shape[0]),min(129,source.shape[1])
    x_store=torch.empty((rows,cols+3),device=source.device,dtype=source.dtype)
    x=x_store[:,:cols]
    x.copy_(source[:rows,:cols])
    x[0].fill_(0.001)
    y_store=torch.empty((rows,cols+5),device=source.device,dtype=c['y_buffer'].dtype)
    y=y_store[:,:cols]
    rsigma=torch.full((rows,),float('nan'),device=source.device,dtype=torch.float32)
    g=c['g'][:cols]
    saved=x.clone(),g.clone()
    dummy=torch.empty(0,device=source.device)
    eps=1e-3
    module.rmsnorm(x,g,y,rsigma,dummy,dummy,dummy,rows,cols,
                   c['ZERO_CENTERED_GAMMA'],256,False,rows,eps)
    xf=saved[0].float()
    gamma=saved[1].float()+(1 if c['ZERO_CENTERED_GAMMA'] else 0)
    expected_rs=torch.rsqrt((xf*xf).mean(-1)+eps)
    expected_y=(xf*expected_rs[:,None]*gamma).to(y.dtype)
    atol,rtol=(1e-3,1e-2) if y.dtype in (torch.float16,torch.bfloat16) else (1e-5,1e-5)
    compare(y,expected_y,atol=atol,rtol=rtol)
    compare(rsigma,expected_rs,atol=atol,rtol=rtol)
    if not torch.equal(x,saved[0]) or not torch.equal(g,saved[1]):
        raise AssertionError('RMS norm modified a read-only control input')


def check_blocked_stride_control(c, module):
    """Exercise the full-width blocked path with independent row strides."""
    if not c['USE_BLOCKED_fwd']:
        return
    source=c['x']
    cols=source.shape[1]
    block_size=c['blk_size_fwd']
    if cols<=block_size or source.ndim!=2:
        raise AssertionError('Blocked RMS norm control needs its declared full width')
    rows=3
    # Keep 16-element alignment expected by the kernel while making the
    # input and output row strides different from each other and contiguous.
    x_store=torch.full((rows,cols+16),-19,device=source.device,dtype=source.dtype)
    x=x_store[:,:cols]
    x[0].copy_(source[0]);x[1].copy_(source[0].neg());x[2].fill_(0.001)
    y_store=torch.full((rows,cols+32),float('nan'),device=source.device,
                       dtype=c['y_buffer'].dtype)
    y=y_store[:,:cols]
    rsigma=torch.full((rows,),float('nan'),device=source.device,dtype=torch.float32)
    g=c['g']
    saved=x.clone(),g.clone()
    dummy=torch.empty(0,device=source.device)
    eps=c['eps']
    module.rmsnorm(x,g,y,rsigma,dummy,dummy,dummy,rows,cols,
                   c['ZERO_CENTERED_GAMMA'],block_size,True,
                   min(rows,c['NUM_PRGMS_fwd']),eps)
    xf=saved[0].float()
    gamma=saved[1].float()+(1 if c['ZERO_CENTERED_GAMMA'] else 0)
    expected_rs=torch.rsqrt((xf*xf).mean(-1)+eps)
    expected_y=(xf*expected_rs[:,None]*gamma).to(y.dtype)
    atol,rtol=(1e-3,1e-2) if y.dtype in (torch.float16,torch.bfloat16) else (1e-5,1e-5)
    compare(y,expected_y,atol=atol,rtol=rtol)
    compare(rsigma,expected_rs,atol=atol,rtol=rtol)
    if not torch.equal(x,saved[0]) or not torch.equal(g,saved[1]):
        raise AssertionError('Blocked RMS norm modified a read-only control input')


@contextmanager
def perturbed_inputs(c):
    """Change live operands for a bound replay; restore them on every exit."""
    names = ('x', 'g')
    saved = {name: c[name].clone() for name in names if isinstance(c[name], torch.Tensor)}
    try:
        c['x'].mul_(-1)
        c['g'].add_(0.5)
        yield
    finally:
        for name, original in saved.items():
            c[name].copy_(original)


def poison_outputs(c, result):
    """Invalidate scored output buffers before checking the bound replay."""
    for output in (c['y_buffer'], c['rsigma_buffer']):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Missing scored output buffer")
        if output.dtype == torch.bool:
            output.logical_not_()
        elif output.is_floating_point():
            output.fill_(float("nan"))
        else:
            output.fill_(torch.iinfo(output.dtype).min)
