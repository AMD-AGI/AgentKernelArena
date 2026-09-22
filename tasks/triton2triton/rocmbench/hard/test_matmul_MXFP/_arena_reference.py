"""Private input snapshots and complete MXFP output/replay validation."""
import torch


class NumericalMismatch(AssertionError):
    pass


def equal_bytes(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class OutputCheck:
    def __init__(self, inputs, output, oracle, *, atol=None, rtol=None, equal_nan=False):
        self.inputs = list(inputs)
        self.original = [x.clone() for x in inputs]
        self.snapshots = self.original
        self.strides = [x.stride() for x in inputs]
        self.output = output
        self.output_original = output.clone()
        self.output_stride = output.stride()
        self.oracle = oracle
        self.atol, self.rtol, self.equal_nan = atol, rtol, equal_nan
        self.expected = oracle(*self.snapshots)

    def check_inputs(self):
        for actual, expected, stride in zip(self.inputs, self.snapshots, self.strides):
            if (actual.shape, actual.dtype, actual.device, actual.stride()) != (expected.shape, expected.dtype, expected.device, stride):
                raise ValueError('Read-only input metadata changed')
            if not equal_bytes(actual, expected):
                raise ValueError('Read-only input was modified')

    def __call__(self, output):
        if output is not self.output:
            raise ValueError('Unexpected public output buffer')
        if (output.shape, output.dtype, output.device, output.stride()) != (self.expected.shape, self.expected.dtype, self.expected.device, self.output_stride):
            raise ValueError('Output metadata changed')
        self.check_inputs()
        if any(output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in self.inputs):
            raise ValueError('Output aliases a read-only input')
        if not self.equal_nan and not bool(torch.isfinite(output).all()):
            raise ValueError('Nonfinite output')
        try:
            torch.testing.assert_close(output, self.expected, atol=self.atol, rtol=self.rtol, equal_nan=self.equal_nan)
        except AssertionError as exc:
            raise NumericalMismatch(str(exc)) from exc

    def replace(self, values):
        if len(values) != len(self.inputs):
            raise ValueError('Replay input count mismatch')
        self.snapshots = [x.clone() for x in values]
        # The oracle sees private values, before the candidate can mutate them.
        self.expected = self.oracle(*self.snapshots)
        for actual, values in zip(self.inputs, self.snapshots): actual.copy_(values)
        self.output.fill_(float('nan'))

    def fresh(self, output):
        if output is not self.output:
            raise ValueError('Unexpected replay output')
        a, b = self.original
        rows = ((torch.arange(a.shape[0], device=a.device) % 5) - 2).to(a.dtype)[:, None] * .125
        cols = ((torch.arange(b.shape[1], device=b.device) % 7) - 3).to(b.dtype)[None, :] * .125
        self.replace([-a.flip(1) + rows, b.flip(0) + cols])

    def restore(self):
        for actual, original in zip(self.inputs, self.original): actual.copy_(original)
        self.output.copy_(self.output_original)


def prepare(context, module):
    if context['is_scaled_mode']:
        raise RuntimeError('Scaled performance is outside the declared six-case scoring manifest')
    a, b, output = (context[k] for k in ('a_tensor', 'b_tensor', 'output_buffer'))
    if a.dtype not in (torch.float16, torch.float32) or b.dtype != a.dtype or output.dtype != torch.float16:
        raise ValueError('Unscaled matmul input/output dtype violates the contract')
    # Preserve FP32 operand loads; do not round FP32 inputs to half first.
    return OutputCheck([a, b], output, lambda x, y: (x.float() @ y.float()).half())
