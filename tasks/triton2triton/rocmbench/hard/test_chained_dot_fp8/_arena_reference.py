"""Scaled FP8 original gate and bounded FP16 two-stage products; protected replay."""
import torch


class NumericalMismatch(AssertionError):
    pass


def equal_bytes(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

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
    center = x @ y.transpose(-1, -2)
    radius = gamma(2 * a.shape[-1]) * (x.abs() @ y.abs().transpose(-1, -2))
    lo_h = (center - radius).half().double()
    hi_h = (center + radius).half().double()
    if not bool(torch.isfinite(lo_h).all() and torch.isfinite(hi_h).all()):
        raise ValueError('Intermediate half overflow outside the finite task domain')
    midpoint = (lo_h + hi_h) * .5
    uncertainty = (hi_h - lo_h) * .5
    center_o = midpoint @ z
    radius_o = uncertainty @ z.abs()
    radius_o += gamma(2 * b.shape[-2]) * (torch.maximum(lo_h.abs(), hi_h.abs()) @ z.abs())
    return (center_o - radius_o).half(), (center_o + radius_o).half()



class DotCheck:
    def __init__(self, q, k, v, scales=(1., 1., 1., 1., 1., 1.), *, strict=False):
        self.inputs = [q, k, v]
        self.original = [x.clone() for x in self.inputs]
        self.strides = [x.stride() for x in self.inputs]
        self.snapshots = self.original
        self.scales, self.strict = scales, strict
        self.fp8 = q.dtype == torch.float8_e4m3fnuz
        if q.dtype not in (torch.float16, torch.float8_e4m3fnuz) or any(x.dtype != q.dtype for x in self.inputs):
            raise ValueError('Expected homogeneous declared FP16/FP8 inputs')
        self.poisoned_outputs = []
        self.reference()

    def reference(self):
        q, k, v = self.snapshots
        q_desc, k_desc, v_desc, s_sc, s_desc, o_sc = self.scales
        if self.fp8:
            # Keep the existing scaled FP8 reference and absolute gate.
            s = (q.float() * q_desc) @ (k.float().transpose(-1, -2) * k_desc)
            s = (s * s_sc).to(q.dtype).float() * s_desc
            out = s @ (v.float().transpose(-1, -2) * v_desc)
            self.expected = (out * o_sc).to(q.dtype)
            self.bounds = None
        else:
            self.expected = (q @ k.transpose(-1, -2)) @ v.transpose(-1, -2)
            self.bounds = chain_bounds(q, k, v.transpose(-1, -2))

    def check_inputs(self):
        for actual, expected, stride in zip(self.inputs, self.snapshots, self.strides):
            if (actual.shape, actual.dtype, actual.device, actual.stride()) != (expected.shape, expected.dtype, expected.device, stride):
                raise ValueError('Read-only input metadata changed')
            if not equal_bytes(actual, expected): raise ValueError('Read-only input was modified')

    def __call__(self, output):
        if not isinstance(output, torch.Tensor): raise TypeError('Expected returned chained-dot tensor')
        if (output.shape, output.dtype, output.device) != (self.expected.shape, self.expected.dtype, self.expected.device):
            raise ValueError('Output metadata violates the contract')
        self.check_inputs()
        if any(output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in self.inputs):
            raise ValueError('Output aliases a read-only input')
        if not bool(torch.isfinite(output.float()).all()): raise ValueError('Nonfinite output')
        try:
            if self.fp8 or self.strict:
                torch.testing.assert_close(output.float(), self.expected.float(), atol=1e-2, rtol=0)
            elif not bool(((output >= self.bounds[0]) & (output <= self.bounds[1])).all()):
                raise AssertionError('Output outside the two-stage FP16 rounding interval')
        except AssertionError as exc: raise NumericalMismatch(str(exc)) from exc

    def fresh(self, output):
        self.poisoned_outputs.append((output, output.clone()))
        q, k, v = self.original
        if self.fp8:
            # Exact FP8 sign flip preserves range with the unchanged scale
            # constants, avoiding unsupported byte arithmetic/overflow controls.
            values = [(-q.float()).to(q.dtype), k.clone(), v.clone()]
        else:
            rows = ((torch.arange(q.shape[1], device=q.device) % 5) - 2).to(q.dtype)[None, :, None] * .125
            values = [-q.flip(-1) + rows, k.flip(-1), v.clone()]
        self.snapshots = values
        self.reference()
        for actual, values in zip(self.inputs, self.snapshots): actual.copy_(values)
        output.fill_(float('nan'))

    def restore(self):
        for actual, original in zip(self.inputs, self.original): actual.copy_(original)
        for output, original in self.poisoned_outputs: output.copy_(original)


def prepare(context, module):
    scales = tuple(context[k] for k in ('q_desc_py', 'k_desc_py', 'v_desc_py', 's_sc_py', 's_desc_py', 'o_sc_py'))
    return DotCheck(context['q_for_kernel'], context['k_for_kernel'], context['v_for_kernel_call'], scales)
