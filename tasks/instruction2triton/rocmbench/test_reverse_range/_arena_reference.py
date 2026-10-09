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
    source = c['data_perf']
    original = source.detach().clone()
    contract = (source.shape, source.dtype, source.device,
                source.stride(), source.storage_offset())
    expected = torch.flip(source[1:513], [0])

    def check(result):
        current = c['data_perf']
        if (not isinstance(current, torch.Tensor) or current is not source
                or (current.shape, current.dtype, current.device,
                    current.stride(), current.storage_offset()) != contract
                or not torch.equal(current.contiguous().view(torch.uint8),
                                   original.contiguous().view(torch.uint8))):
            raise AssertionError('Input argument data_perf was modified')
        compare(c['res_perf_buffer'], expected, exact=True)

    return check


def check_interior_transitions(c, invoke):
    """A correct output sentinel must not hide changes elsewhere in the input."""
    source, output = c['data_perf'], c['res_perf_buffer']
    saved_source, saved_output = source.clone(), output.clone()
    try:
        for index in (511, 257):
            source.copy_(saved_source)
            output.copy_(saved_output)
            # input[512], which determines output[0], stays unchanged.
            source[index].add_(3)
            check = prepare(c, None)
            check(invoke())
    finally:
        source.copy_(saved_source)
        output.copy_(saved_output)


@contextmanager
def perturbed_inputs(c):
    """Change live operands for a bound replay; restore them on every exit."""
    names = ('data_perf',)
    saved = {name: c[name].clone() for name in names if isinstance(c[name], torch.Tensor)}
    try:
        c['data_perf'].mul_(-1)
        yield
    finally:
        for name, original in saved.items():
            c[name].copy_(original)


def poison_outputs(c, result):
    """Invalidate scored output buffers before checking the bound replay."""
    for output in (c['res_perf_buffer'],):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Missing scored output buffer")
        if output.dtype == torch.bool:
            output.logical_not_()
        elif output.is_floating_point():
            output.fill_(float("nan"))
        else:
            output.fill_(torch.iinfo(output.dtype).min)
