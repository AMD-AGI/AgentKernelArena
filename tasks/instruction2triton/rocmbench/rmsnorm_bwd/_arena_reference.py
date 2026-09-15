"""Independent RMS backward outputs; no extra reduced-gradient scoring surrogate."""
import torch


class NumericalMismatch(AssertionError):
    pass


def equal_bytes(a, b):
    return torch.equal(a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8))


class FrozenInputs:
    def __init__(self, inputs):
        self.inputs = list(inputs)
        self.original = [x.detach().clone() for x in inputs]
        self.snapshots = self.original
        self.strides = [x.stride() for x in inputs]

    def check(self):
        for actual, snapshot, stride in zip(self.inputs, self.snapshots, self.strides):
            if (actual.shape, actual.dtype, actual.device, actual.stride()) != (snapshot.shape, snapshot.dtype, snapshot.device, stride):
                raise ValueError('Read-only input metadata changed')
            if not equal_bytes(actual, snapshot): raise ValueError('Read-only input was modified')

    def restore(self):
        with torch.no_grad():
            for actual, original in zip(self.inputs, self.original): actual.copy_(original)


class BackwardCheck(FrozenInputs):
    def __init__(self, context):
        super().__init__([context[k] for k in ('x', 'g', 'grad_output', 'rsigma_buffer')])
        self.outputs = [context['dx_bench'], context['dg_tmp_bench']]
        self.output_original = [x.clone() for x in self.outputs]
        self.output_strides = [x.stride() for x in self.outputs]
        self.centered, self.eps = context['ZERO_CENTERED_GAMMA'], context['eps']
        self.atol, self.rtol = (1e-3, 1e-2) if self.inputs[0].dtype in (torch.float16, torch.bfloat16) else (1e-5, 1e-5)
        x, g, go, r = self.original
        # rsigma is an input to the timed backward, produced by protected forward.
        torch.testing.assert_close(r, torch.rsqrt(x.float().square().mean(-1) + self.eps), atol=1e-5, rtol=1e-5)
        self.reference()

    def reference(self):
        x, g, go, r = [x.float() for x in self.snapshots]
        r = r[:, None]
        gamma = g + (1 if self.centered else 0)
        grad_sum = (go * x * gamma).mean(-1, keepdim=True)
        dx = go * r * gamma - (r * r * r) * x * grad_sum
        # The public kernel writes every per-row contribution, not its row sum.
        dg_tmp = (go * x) * r
        self.expected = [dx.to(self.outputs[0].dtype), dg_tmp]

    def __call__(self, launch_result):
        # A Triton launch returns a compiled-kernel handle; its actual declared
        # outputs are the two preallocated buffers bound to this timed callable.
        self.check()
        for output, expected, original, stride in zip(self.outputs, self.expected, self.output_original, self.output_strides):
            if (output.shape, output.dtype, output.device, output.stride()) != (expected.shape, expected.dtype, expected.device, stride):
                raise ValueError('Backward output metadata violates the contract')
            if any(output.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in self.inputs):
                raise ValueError('Backward output aliases a read-only input')
            if not bool(torch.isfinite(output).all()): raise ValueError('Nonfinite backward output')
            try:
                torch.testing.assert_close(output, expected, atol=self.atol, rtol=self.rtol)
            except AssertionError as exc: raise NumericalMismatch(str(exc)) from exc

    def fresh(self, launch_result):
        x, g, go, r = self.original
        rows = ((torch.arange(x.shape[0], device=x.device) % 5) - 2).to(x.dtype)[:, None] * .125
        cols = ((torch.arange(x.shape[1], device=x.device) % 7) - 3).to(g.dtype) * .125
        fresh_x = -x.flip(1) + rows
        fresh_g = -g.flip(-1) + cols
        fresh_go = go.flip(1) * .5 + cols[None, :]
        fresh_r = torch.rsqrt(fresh_x.float().square().mean(-1) + self.eps)
        self.snapshots = [fresh_x, fresh_g, fresh_go, fresh_r]
        self.reference()
        for actual, values in zip(self.inputs, self.snapshots): actual.copy_(values)
        for output in self.outputs: output.fill_(float('nan'))

    def restore(self):
        super().restore()
        for output, original in zip(self.outputs, self.output_original): output.copy_(original)


def prepare(context, module):
    return BackwardCheck(context)
