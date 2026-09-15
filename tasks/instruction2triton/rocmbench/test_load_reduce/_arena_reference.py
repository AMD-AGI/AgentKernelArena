"""Independent output checks for the performance inputs; never timed or editable."""
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


def check_readonly(actual, original):
    if (actual.shape,actual.dtype,actual.device)!=(original.shape,original.dtype,original.device):
        raise ValueError('Read-only input metadata changed')
    # Value equality would miss a changed zero sign; preserve input bytes.
    if not torch.equal(actual.contiguous().view(torch.uint8),original.contiguous().view(torch.uint8)):
        raise ValueError('Read-only input was modified')


class ReductionCheck:
    """Independent input snapshots never become arguments to the candidate."""
    def __init__(self, x, output):
        self.x=x;self.output=output
        self.original_x=x.clone();self.original_output=output.clone()
        self.input_strides=x.stride()
        self.input_snapshot=self.original_x.clone()
        self.expected=self.input_snapshot.max(dim=1).values

    def __call__(self, result):
        if result is not self.output:
            raise ValueError('Timed reduction must expose the declared output buffer')
        if self.x.stride()!=self.input_strides:
            raise ValueError('Read-only input strides changed')
        check_readonly(self.x,self.input_snapshot)
        # Preserve the original task's comparison rule, including check_dtype.
        compare(self.output,self.expected,atol=1e-3,rtol=1e-2,check_dtype=False)

    def poison(self):
        self.output.fill_(float('nan'))

    def fresh(self):
        # Same allocations, shape, strides and dtype; no RNG draws. Reverse and
        # negate columns, with row-varying positive/negative offsets, so stale
        # maxima and zero-initialized reductions are meaningfully challenged.
        rows=torch.arange(self.x.shape[0],device=self.x.device)
        shift=torch.where(rows%2==0,-8-rows%3,8+rows%3).to(self.x.dtype)
        fresh=-self.original_x.flip(1)+shift[:,None]
        self.input_snapshot=fresh.clone()
        self.expected=self.input_snapshot.max(dim=1).values
        if torch.allclose(self.expected,self.original_x.max(dim=1).values,atol=1e-3,rtol=1e-2):
            raise RuntimeError('Replay control did not change the expected reduction')
        self.x.copy_(fresh)
        self.poison()

    def restore(self):
        self.x.copy_(self.original_x)
        self.output.copy_(self.original_output)


def prepare(c, module):
    return ReductionCheck(c['x'],c['y_buffer'])
