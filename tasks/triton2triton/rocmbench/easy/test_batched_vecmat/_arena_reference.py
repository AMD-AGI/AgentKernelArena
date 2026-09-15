"""FP16 products, FP32 reduction and complete readonly batched-vector replay."""
import torch


class NumericalMismatch(AssertionError):
    pass


def equal_bytes(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class VecmatCheck:
    def __init__(self, a, b):
        self.inputs = [a, b]
        self.original = [a.clone(), b.clone()]
        self.strides = [a.stride(), b.stride()]
        self.snapshots = self.original
        self.poisoned_outputs = []
        self.reference()

    def reference(self):
        a, b = self.snapshots
        if a.dtype not in (torch.float16, torch.float32) or b.dtype != a.dtype:
            raise ValueError('Declared VecMat operands require matching FP16/FP32 dtypes')
        # Keep input-dtype product rounding; only the reduction accumulates FP32.
        self.expected = (a[:, None, :] * b).sum(dim=2, dtype=torch.float32).to(a.dtype)

    def check_inputs(self):
        for x, expected, stride in zip(self.inputs, self.snapshots, self.strides):
            if (x.shape, x.dtype, x.device, x.stride()) != (expected.shape, expected.dtype, expected.device, stride):
                raise ValueError('Read-only input metadata changed')
            if not equal_bytes(x, expected): raise ValueError('Read-only input was modified')

    def __call__(self, output):
        if not isinstance(output, torch.Tensor): raise TypeError('Expected returned GEMM tensor')
        if (output.shape, output.dtype, output.device) != (self.expected.shape, self.expected.dtype, self.expected.device):
            raise ValueError('Output metadata violates the contract')
        self.check_inputs()
        if any(output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in self.inputs):
            raise ValueError('Output aliases a read-only input')
        if not bool(torch.isfinite(output).all()): raise ValueError('Nonfinite output')
        try:
            torch.testing.assert_close(output, self.expected, atol=1e-3, rtol=1e-2)
        except AssertionError as exc:
            raise NumericalMismatch(str(exc)) from exc

    def replace(self, values):
        self.snapshots = [x.clone() for x in values]
        self.reference()
        for x, values in zip(self.inputs, self.snapshots): x.copy_(values)

    def fresh(self, output):
        self.poisoned_outputs.append((output, output.clone()))
        a, b = self.original
        rows = ((torch.arange(a.shape[0], device=a.device) % 5) - 2).to(a.dtype)[:, None] * .125
        cols = ((torch.arange(b.shape[1], device=b.device) % 7) - 3).to(b.dtype)[None, :, None] * .125
        self.replace([-a.flip(1) + rows, b.flip(2) + cols])
        output.fill_(float('nan'))

    def restore(self):
        for actual, original in zip(self.inputs, self.original): actual.copy_(original)
        for output, original in self.poisoned_outputs: output.copy_(original)


def prepare(context, module):
    return VecmatCheck(context['A_tri'], context['B_tri'])
