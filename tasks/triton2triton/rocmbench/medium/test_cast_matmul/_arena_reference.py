"""Protected cast-before-matmul oracle; original atol=0.3, rtol=0.01."""
import torch


class NumericalMismatch(AssertionError):
    pass


def readonly(actual, expected, stride):
    if (actual.shape, actual.dtype, actual.device, actual.stride()) != (expected.shape, expected.dtype, expected.device, stride):
        raise ValueError('Read-only input metadata changed')
    if not torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
        raise ValueError('Read-only input was modified')


class CastMatmulCheck:
    def __init__(self, a, b, output):
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError('Invalid matmul dimensions')
        if a.dtype not in (torch.float16, torch.float32, torch.float64) or b.dtype not in (torch.float16, torch.float32, torch.float64):
            raise ValueError('Unsupported input dtype')
        if output.dtype not in (torch.float16, torch.float32):
            raise ValueError('Unsupported output dtype')
        if output.shape != (a.shape[0], b.shape[1]) or output.device != a.device or b.device != a.device:
            raise ValueError('Output shape/device mismatch')
        self.a, self.b, self.output = a, b, output
        self.strides = [a.stride(), b.stride()]
        self.output_stride = output.stride()
        self.original = [a.clone(), b.clone()]
        self.snapshots = self.original
        self.output_original = output.clone()
        self._reference()

    def _reference(self):
        # Match original reference casting BEFORE the product, not after it.
        a, b = self.snapshots
        self.expected = a.to(self.output.dtype) @ b.to(self.output.dtype)

    def __call__(self, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError('Expected tensor output')
        if output is not self.output:
            raise ValueError('Public wrapper must return its declared output buffer')
        if (output.shape, output.dtype, output.device) != (self.expected.shape, self.expected.dtype, self.expected.device):
            raise ValueError('Output shape/dtype/device mismatch')
        if output.stride() != self.output_stride:
            raise ValueError('Output strides changed')
        for actual, snapshot, stride in zip((self.a, self.b), self.snapshots, self.strides):
            readonly(actual, snapshot, stride)
            if output.untyped_storage().data_ptr() == actual.untyped_storage().data_ptr():
                raise ValueError('Output aliases a read-only input')
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Nonfinite output')
        try:
            torch.testing.assert_close(output, self.expected, atol=0.3, rtol=0.01)
        except AssertionError as exc:
            raise NumericalMismatch(str(exc)) from exc

    def fresh(self, output):
        if output is not self.output:
            raise ValueError('Unexpected replay buffer')
        a, b = self.original
        # Valid floating inputs, same layouts/pointers/shapes; no random draws.
        row = ((torch.arange(a.shape[0], device=a.device) % 5) - 2).to(a.dtype)[:, None] * 0.125
        col = ((torch.arange(b.shape[1], device=b.device) % 7) - 3).to(b.dtype)[None, :] * 0.125
        self.snapshots = [-a.flip(1) + row, b.flip(0) + col]
        self._reference()
        self.a.copy_(self.snapshots[0]); self.b.copy_(self.snapshots[1])
        output.fill_(float('nan'))

    def restore(self):
        self.a.copy_(self.original[0]); self.b.copy_(self.original[1])
        self.output.copy_(self.output_original)


def prepare(context, module):
    return CastMatmulCheck(context['a'], context['b'], context['out_triton'])
