"""Protected FP32 GEMM oracle and exact unsupported-multi-CTA checks."""
import torch


class NumericalMismatch(AssertionError):
    pass


def equal_bytes(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class GemmCheck:
    def __init__(self, a, b, output):
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError('Invalid GEMM dimensions')
        if a.dtype != torch.float16 or b.dtype != torch.float16:
            raise ValueError('Inputs must be float16')
        if output.dtype not in (torch.float16, torch.float32):
            raise ValueError('Output must be float16 or float32')
        if output.shape != (a.shape[0], b.shape[1]) or a.device != b.device or output.device != a.device:
            raise ValueError('Output shape/device mismatch')
        self.inputs = [a, b]
        self.original = [a.clone(), b.clone()]
        self.strides = [a.stride(), b.stride()]
        self.snapshots = self.original
        self.output = output
        self.output_original = output.clone()
        self.output_stride = output.stride()
        self._reference()

    def _reference(self):
        # Original correctness uses float32 operands/reference even for fp16 C.
        self.expected = self.snapshots[0].float() @ self.snapshots[1].float()

    def check_inputs(self):
        for actual, snapshot, stride in zip(self.inputs, self.snapshots, self.strides):
            if (actual.shape, actual.dtype, actual.device, actual.stride()) != (snapshot.shape, snapshot.dtype, snapshot.device, stride):
                raise ValueError('Read-only input metadata changed')
            if not equal_bytes(actual, snapshot):
                raise ValueError('Read-only input was modified')

    def __call__(self, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError('Expected output tensor')
        if output is not self.output:
            raise ValueError('Unexpected public output buffer')
        if (output.shape, output.dtype, output.device, output.stride()) != (self.output_original.shape, self.output_original.dtype, self.output_original.device, self.output_stride):
            raise ValueError('Output metadata changed')
        self.check_inputs()
        if any(output.untyped_storage().data_ptr() == a.untyped_storage().data_ptr() for a in self.inputs):
            raise ValueError('Output aliases a read-only input')
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Nonfinite output')
        try:
            torch.testing.assert_close(output, self.expected, atol=1e-3, rtol=1e-2, check_dtype=False)
        except AssertionError as exc:
            raise NumericalMismatch(str(exc)) from exc

    def fresh(self, output):
        if output is not self.output:
            raise ValueError('Unexpected replay output')
        a, b = self.original
        rows = ((torch.arange(a.shape[0], device=a.device) % 5) - 2).to(a.dtype)[:, None] * 0.125
        cols = ((torch.arange(b.shape[1], device=b.device) % 7) - 3).to(b.dtype)[None, :] * 0.125
        self.snapshots = [-a.flip(1) + rows, b.flip(0) + cols]
        self._reference()
        for dest, values in zip(self.inputs, self.snapshots): dest.copy_(values)
        output.fill_(float('nan'))

    def restore(self):
        for dest, original in zip(self.inputs, self.original): dest.copy_(original)
        self.output.copy_(self.output_original)


def rejected_cta_arch(num_ctas):
    if num_ctas <= 1:
        return None
    from triton.runtime import driver
    target = driver.active.get_current_target()
    if target.backend != 'hip':
        return None
    from triton.backends.amd.compiler import amd
    return None if amd.supports_multi_cta_launch(target.arch) else target.arch


def check_cta_rejection(launch, check, arch):
    # Require the real candidate launch to reach the backend option rejection.
    try:
        launch()
    except ValueError as exc:
        if str(exc) != f'num_ctas > 1 not supported on {arch}':
            raise
    else:
        raise AssertionError('Unsupported multi-CTA launch unexpectedly succeeded')
    check.check_inputs()
    if not equal_bytes(check.output, check.output_original):
        raise ValueError('Rejected launch modified its output')


def prepare(context, module):
    return GemmCheck(context['a_host'], context['b_host'], context['c_buffer'])
