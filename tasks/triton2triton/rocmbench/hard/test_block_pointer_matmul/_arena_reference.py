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
    expected=c['a']@c['b']
    return lambda result: compare(c['c'],expected,check_dtype=False)


def check_stride_controls(c, module):
    """Unscored independent A, B, and C stride-addressing checks."""
    dtype,device=c['a'].dtype,c['a'].device
    a0=(((torch.arange(256,device=device).reshape(16,16)%7)-3)/8).to(dtype)
    b0=(((torch.arange(256,device=device).reshape(16,16)%5)-2)/8).to(dtype)
    def strided(value):
        storage=torch.empty((16,32),device=device,dtype=value.dtype)
        view=storage[:,::2]
        view.copy_(value)
        return view
    for operand in ('a','b','c'):
        a=strided(a0) if operand=='a' else a0.clone()
        b=strided(b0) if operand=='b' else b0.clone()
        out=strided(torch.full((16,16),float('nan'),device=device,dtype=dtype)) if operand=='c' else torch.full((16,16),float('nan'),device=device,dtype=dtype)
        saved=a.clone(),b.clone()
        returned=module.block_pointer_matmul_triton_wrapper(a,b,out,4)
        if returned is not out:
            raise AssertionError('Block-pointer wrapper did not return C')
        compare(out,a0@b0,check_dtype=False)
        if not torch.equal(a,saved[0]) or not torch.equal(b,saved[1]):
            raise AssertionError('Block-pointer matmul modified a read-only operand')


def check_partial_tile_controls(c, module):
    """Exercise the declared single-tile M, N, and K boundaries independently."""
    device = c['a'].device
    m, n, k = 32, 32, 64
    a = ((torch.arange(m * k, device=device).reshape(m, k) % 17) + 1).to(torch.float16) / 16
    b = ((torch.arange(k * n, device=device).reshape(k, n) % 19) + 1).to(torch.float16) / 32
    saved_a, saved_b = a.clone(), b.clone()
    sentinel = -37.0
    for block_m, block_n, block_k in ((16, 32, 64), (32, 16, 64), (32, 32, 32)):
        out = torch.full((m, n), sentinel, device=device, dtype=torch.float32)
        module.matmul_no_scf_with_advance_kernel[(1,)](
            a_ptr=a, b_ptr=b, c_ptr=out, M=m, N=n, K=k,
            stride_am=a.stride(0), stride_ak=a.stride(1),
            stride_bk=b.stride(0), stride_bn=b.stride(1),
            stride_cm=out.stride(0), stride_cn=out.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=4)
        expected_tile = a[:block_m, :block_k] @ b[:block_k, :block_n]
        # Match the original FP16-PyTorch comparison for the FP32 accumulator.
        compare(out[:block_m, :block_n], expected_tile, check_dtype=False)
        untouched = torch.ones_like(out, dtype=torch.bool)
        untouched[:block_m, :block_n] = False
        if not torch.equal(out[untouched], torch.full_like(out[untouched], sentinel)):
            raise AssertionError('Single-tile kernel wrote outside its output tile')
        if not torch.equal(a, saved_a) or not torch.equal(b, saved_b):
            raise AssertionError('Single-tile kernel modified a read-only operand')


@contextmanager
def perturbed_inputs(c):
    """Change live operands for a bound replay; restore them on every exit."""
    names = ('a', 'b')
    saved = {name: c[name].clone() for name in names if isinstance(c[name], torch.Tensor)}
    try:
        c['a'].copy_(torch.flip(c['a'], (0,)))
        c['b'].copy_(torch.flip(c['b'], (1,)))
        yield
    finally:
        for name, original in saved.items():
            c[name].copy_(original)


def poison_outputs(c, result):
    """Invalidate scored output buffers before checking the bound replay."""
    for output in (c['c'],):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Missing scored output buffer")
        if output.dtype == torch.bool:
            output.logical_not_()
        elif output.is_floating_point():
            output.fill_(float("nan"))
        else:
            output.fill_(torch.iinfo(output.dtype).min)
