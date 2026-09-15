"""Protected FP32 softmax oracle and explicit output-rounding acceptance policy."""
import torch

ATOL=1e-8
RTOL=1e-5


class NumericalMismatch(AssertionError):
    pass


def check_readonly(actual, original):
    if (actual.shape,actual.dtype,actual.device)!=(original.shape,original.dtype,original.device):
        raise ValueError('Read-only input metadata changed')
    if not torch.equal(actual.contiguous().view(torch.uint8),original.contiguous().view(torch.uint8)):
        raise ValueError('Read-only input was modified')


class SoftmaxCheck:
    def __init__(self, x):
        if x.dtype not in (torch.float32,torch.float16,torch.bfloat16):
            raise ValueError('Unsupported softmax dtype')
        self.x=x;self.original=x.clone();self.snapshot=self.original
        self.strides=x.stride();self.poisoned=[]
        self._reference()

    def _reference(self):
        # Original correctness is FP32 torch.softmax + default torch.allclose.
        # Keep that accuracy interval BEFORE rounding to the declared output
        # dtype. Low-precision performance cases had no original numeric gate.
        self.expected=torch.softmax(self.snapshot.float(),dim=1)
        if self.x.dtype!=torch.float32:
            radius=ATOL+RTOL*self.expected.abs()
            self.lower=(self.expected-radius).to(self.x.dtype)
            self.upper=(self.expected+radius).to(self.x.dtype)

    def __call__(self, output):
        if not isinstance(output,torch.Tensor):
            raise TypeError('Softmax must return a tensor')
        if (output.shape,output.dtype,output.device)!=(self.original.shape,self.original.dtype,self.original.device):
            raise ValueError('Softmax output shape/dtype/device mismatch')
        if output.untyped_storage().data_ptr()==self.x.untyped_storage().data_ptr():
            raise ValueError('Softmax output must not alias the read-only input')
        if self.x.stride()!=self.strides:
            raise ValueError('Read-only input strides changed')
        check_readonly(self.x,self.snapshot)
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Softmax output contains nonfinite values')
        if self.x.dtype==torch.float32:
            if not torch.allclose(output,self.expected,atol=ATOL,rtol=RTOL):
                raise NumericalMismatch('Full FP32 output violates original allclose gate')
        elif not bool(((output>=self.lower)&(output<=self.upper)).all()):
            raise NumericalMismatch('Full output lies outside the FP32 accuracy interval rounded to output dtype')

    def fresh(self, output):
        # A row-wise additive offset would leave softmax unchanged. Use column-
        # varying offsets plus a reverse/negation; no RNG draws or timed changes.
        columns=torch.arange(self.x.shape[1],device=self.x.device)
        bias=((columns%7)-3).to(self.x.dtype)*0.5
        self.snapshot=-self.original.flip(1)+bias[None,:]
        self._reference()
        self.x.copy_(self.snapshot)
        self.poisoned.append((output,output.clone()))
        output.fill_(float('nan'))

    def restore(self):
        self.x.copy_(self.original)
        for output,original in self.poisoned:
            output.copy_(original)


def prepare(c, module):
    return SoftmaxCheck(c['x'])
