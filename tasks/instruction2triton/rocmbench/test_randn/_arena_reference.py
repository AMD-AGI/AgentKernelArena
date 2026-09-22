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


def expected_uniform(seed, count, device):
    """Triton rand's Philox4x32-10 and uint32-to-float32 mapping, not randn."""
    bits=philox32(seed,count).view(np.int32)
    nonnegative=np.where(bits<0,np.bitwise_not(bits),bits)
    uniform=nonnegative.astype(np.float32)*np.float32(4.6566127342e-10)
    return torch.from_numpy(uniform).to(device)


def check_seeded_output(output, seed, count):
    compare(output,expected_uniform(seed,count,output.device),exact=True)


class UniformCheck:
    def __init__(self, context):
        self.output=context['x_output_buffer']
        self.original=self.output.clone()
        self.expected=expected_uniform(context['seed_val'],context['N_elements'],self.output.device)

    def __call__(self, result):
        # The public wrapper returns the output buffer. Checking both protects
        # against accidentally observing an unrelated tensor after capture.
        if result is not self.output:
            raise ValueError('Timed RNG output must be the declared output buffer')
        compare(self.output,self.expected,exact=True)

    def poison(self):
        self.output.fill_(float('nan'))

    def restore(self):
        self.output.copy_(self.original)


def prepare(c, module):
    return UniformCheck(c)
