"""GEMM then output-dtype rounding then row bias, with pristine replay inputs."""
import torch


class NumericalMismatch(AssertionError):
    pass


def equal_bytes(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def bias_bounds(a, b, bias):
    # BF16 products fit exactly in FP32. Enclose FP32 accumulation roundoff,
    # then apply the two specified BF16 rounding steps, including cancellation.
    x, y = a.double(), b.double()
    n_u = (2 * a.shape[1]) * 2.0 ** -24
    if n_u >= 1:
        raise ValueError('FP32 accumulation bound undefined for this dimension')
    center = x @ y
    radius = n_u / (1 - n_u) * (x.abs() @ y.abs())
    lower, upper = (center - radius).bfloat16(), (center + radius).bfloat16()
    if bias is not None:
        lower = (lower.double() + bias.double()[:, None]).bfloat16()
        upper = (upper.double() + bias.double()[:, None]).bfloat16()
    return lower, upper


class BiasCheck:
    def __init__(self, a, b, bias):
        self.inputs = [a, b] + ([bias] if bias is not None else [])
        self.original = [x.clone() for x in self.inputs]
        self.strides = [x.stride() for x in self.inputs]
        self.snapshots = self.original
        self.poisoned_outputs = []
        self.reference()

    def reference(self):
        a, b = self.snapshots[:2]
        bias = self.snapshots[2] if len(self.snapshots) == 3 else None
        if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
            raise ValueError('Declared GEMM operands must have matching FP16/BF16 dtypes')
        self.expected = a @ b
        if bias is not None: self.expected += bias[:, None]
        self.bounds = bias_bounds(a, b, bias) if a.dtype == torch.bfloat16 else None

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
            if self.bounds is None:
                # Original FP16 acceptance remains unchanged, including bias.
                torch.testing.assert_close(output, self.expected, atol=1e-3, rtol=1e-2)
            elif not bool(((output >= self.bounds[0]) & (output <= self.bounds[1])).all()):
                raise AssertionError('Output outside FP32 accumulation/BF16 rounding/bias interval')
        except AssertionError as exc:
            raise NumericalMismatch(str(exc)) from exc

    def replace(self, values):
        self.snapshots = [x.clone() for x in values]
        self.reference()
        for x, values in zip(self.inputs, self.snapshots): x.copy_(values)

    def fresh(self, output):
        self.poisoned_outputs.append((output, output.clone()))
        a, b = self.original[:2]
        rows = ((torch.arange(a.shape[0], device=a.device) % 5) - 2).to(a.dtype)[:, None] * .125
        cols = ((torch.arange(b.shape[1], device=b.device) % 7) - 3).to(b.dtype)[None, :] * .125
        values = [-a.flip(1) + rows, b.flip(0) + cols]
        if len(self.original) == 3: values.append(-self.original[2] + rows[:, 0])
        self.replace(values)
        output.fill_(float('nan'))

    def restore(self):
        for actual, original in zip(self.inputs, self.original): actual.copy_(original)
        for output, original in self.poisoned_outputs: output.copy_(original)


def prepare(context, module):
    return BiasCheck(context['a'], context['b'], context['bias'])
