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


def prepare(c, module):
    return FlipContract(c['x_perf_tensor'], c['z_perf_buffer'])


class FlipContract:
    """Check the supplied output against each invocation's protected input."""

    def __init__(self, x, z):
        self.x, self.z = x, z
        self.pristine = x.clone()
        self.variants = (self.pristine, self.pristine + 17, self.pristine + 41)
        self.snapshot = torch.empty_like(x)
        self.next_variant = 0
        self.active_variant = None
        self.samples_checked = 0

    def prepare_sample(self):
        index = self.next_variant % len(self.variants)
        self.next_variant += 1
        self.active_variant = index
        self.x.copy_(self.variants[index])
        self.z.fill_(-12345)

    def force_next_variant(self, index):
        if index < 0 or index >= len(self.variants):
            raise ValueError('Unknown flip input variant')
        self.next_variant = index

    def before_sample(self, calls_per_replay):
        if calls_per_replay != 1:
            raise AssertionError('Every measured replay must contain one flip invocation')
        # Observation only: no reset or reference work inside a timed sample.
        self.snapshot.copy_(self.x)

    def check_current(self, result, *, measured=False):
        if self.active_variant is None:
            raise AssertionError('Flip input was not prepared')
        protected = self.snapshot if measured else self.variants[self.active_variant]
        compare(self.x, protected, exact=True)
        expected = torch.flip(protected, [1])
        compare(self.z, expected, exact=True)
        compare(result, expected, exact=True)
        if measured:
            self.samples_checked += 1

    def after_sample(self, result):
        self.check_current(result, measured=True)

    def restore(self):
        self.x.copy_(self.pristine)
        self.z.copy_(torch.flip(self.pristine, [1]))
