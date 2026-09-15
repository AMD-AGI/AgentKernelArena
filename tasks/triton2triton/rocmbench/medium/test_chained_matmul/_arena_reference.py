"""Independent two-GEMM rounding bounds, private inputs and complete replay checks."""
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
            self.compare_output(output)
        except AssertionError as exc:
            raise NumericalMismatch(str(exc)) from exc

    def compare_output(self, output):
        torch.testing.assert_close(output, self.expected, atol=0.0, rtol=0.0)

    def replace(self, values):
        if len(values) != len(self.inputs):
            raise ValueError('Replay input count mismatch')
        self.snapshots = [x.clone() for x in values]
        # The oracle sees private values, before the candidate can mutate them.
        self.expected = self.oracle(*self.snapshots)
        for actual, values in zip(self.inputs, self.snapshots): actual.copy_(values)
        self.output.fill_(float('nan'))

    def restore(self):
        for actual, original in zip(self.inputs, self.original): actual.copy_(original)
        self.output.copy_(self.output_original)



def gamma(count):
    # Standard accumulation bound: n rounded operations, FP32 unit roundoff.
    u = 2.0 ** -24
    if count * u >= 1:
        raise ValueError('Accumulation bound is undefined for this dimension')
    return count * u / (1.0 - count * u)


def chain_bounds(a, b, c):
    """FP64 oracle encloses FP32 dots, mandatory intermediate FP16 and final FP16.

    FP16 products are exactly representable in FP32. gamma(2*K) bounds any
    multiplication/addition ordering in the first K-term dot. Converting both
    endpoints to FP16 encloses all allowed intermediate rounding choices. The
    intermediate uncertainty propagates through abs(C), then gamma(2*N) bounds
    the second dot. This bound depends on input values/dimensions, never errors
    observed from the baseline or candidate. FP64 roundoff is dominated by the
    deliberately doubled operation counts in the FP32 bounds.
    """
    if any(x.dtype != torch.float16 for x in (a, b, c)):
        raise ValueError('Declared chained matmul inputs are FP16')
    x, y, z = [v.double() for v in (a, b, c)]
    center = x @ y.T
    radius = gamma(2 * a.shape[1]) * (x.abs() @ y.abs().T)
    lo_h = (center - radius).half().double()
    hi_h = (center + radius).half().double()
    if not bool(torch.isfinite(lo_h).all() and torch.isfinite(hi_h).all()):
        raise ValueError('Intermediate half overflow outside the finite task domain')
    midpoint = (lo_h + hi_h) * .5
    uncertainty = (hi_h - lo_h) * .5
    center_o = midpoint @ z
    radius_o = uncertainty @ z.abs()
    radius_o += gamma(2 * b.shape[0]) * (torch.maximum(lo_h.abs(), hi_h.abs()) @ z.abs())
    return (center_o - radius_o).half(), (center_o + radius_o).half()


class ChainCheck(OutputCheck):
    def __init__(self, a, b, c, output, *, exact=False):
        self.exact = exact
        super().__init__([a, b, c], output, self.reference)

    def reference(self, a, b, c):
        expected = (a.float() @ b.float().T).half().float() @ c.float()
        self.lower, self.upper = chain_bounds(a, b, c)
        return expected.half()

    def compare_output(self, output):
        if self.exact:
            torch.testing.assert_close(output, self.expected, atol=0.0, rtol=0.0)
        elif not bool(((output >= self.lower) & (output <= self.upper)).all()):
            raise AssertionError('Output outside the two-GEMM rounding interval')

    def fresh(self, output):
        if output is not self.output:
            raise ValueError('Unexpected replay output')
        a, b, c = self.original
        rows = ((torch.arange(a.shape[0], device=a.device) % 5) - 2).to(a.dtype)[:, None] * .125
        cols = ((torch.arange(c.shape[1], device=c.device) % 7) - 3).to(c.dtype)[None, :] * .125
        self.replace([-a.flip(1) + rows, b.flip(1), -c.flip(1) + cols])


def prepare(context, module):
    return ChainCheck(context['a'], context['b'], context['c_mat'], context['triton_result_buffer'])
