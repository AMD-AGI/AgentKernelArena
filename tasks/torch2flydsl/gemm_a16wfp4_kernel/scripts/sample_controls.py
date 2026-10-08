"""Observe every Event-measured output against a deterministic input stream.

Sample zero uses the declared input. Later samples have independent BF16
operands of the same shape. Input preparation, output capture, and reference
checks are outside the timed device interval for both scored roles.
"""


class MeasuredInputStream:
    def __init__(self, left, right, *, seed, case_index, samples, output_shape, output_dtype):
        if samples < 1:
            raise ValueError("At least one measured sample is required")
        self.left, self.right = left, right
        self.original_left, self.original_right = left.clone(), right.clone()
        self.output_shape = output_shape
        self.output_dtype = output_dtype
        self.output_device = left.device
        self.seed = seed
        self.case_index = case_index
        self.samples = samples
        self.prepared = 0
        self.reference_prepared = 0
        self.outputs = []
        self.measuring = True
        self.prepared_left = None
        self.prepared_right = None

    def _set_input(self, index):
        import torch

        if index == 0:
            self.left.copy_(self.original_left)
            self.right.copy_(self.original_right)
            return
        generator = torch.Generator(device=self.left.device)
        generator.manual_seed(self.seed + 10_000 * (self.case_index + 1) + index)
        self.left.copy_(torch.randn(self.left.shape, generator=generator,
                                    device=self.left.device, dtype=self.left.dtype))
        self.right.copy_(torch.randn(self.right.shape, generator=generator,
                                     device=self.right.device, dtype=self.right.dtype))

    def prepare(self):
        # TimedRun.rerun invokes this callback too. Preserve the explicit
        # changed inputs used by the post-timing replay check.
        if not self.measuring:
            return
        if self.prepared >= self.samples:
            raise AssertionError("More input preparations than scored samples")
        self._set_input(self.prepared)
        self.prepared_left = self.left.clone()
        self.prepared_right = self.right.clone()
        self.prepared += 1

    def observe(self, output):
        # This runs after end-event synchronization, before the next prepare.
        import torch

        if not isinstance(output, torch.Tensor):
            raise AssertionError("Measured output must be a Tensor")
        if (output.shape != self.output_shape or output.dtype != self.output_dtype
                or output.device != self.output_device):
            raise AssertionError("Measured output violates device/output contract")
        if (not torch.equal(self.left.contiguous().view(torch.uint8),
                            self.prepared_left.contiguous().view(torch.uint8))
                or not torch.equal(self.right.contiguous().view(torch.uint8),
                                    self.prepared_right.contiguous().view(torch.uint8))):
            raise AssertionError("Operator modified a read-only measured input")
        self.outputs.append(output.detach().to("cpu", copy=True))

    def validate(self, reference, compare):
        self.measuring = False
        if self.prepared != self.samples or len(self.outputs) != self.samples:
            raise AssertionError("Reported samples lack matching inputs or outputs")
        for index, actual in enumerate(self.outputs):
            self._set_input(index)
            expected = reference()
            compare(actual, expected.cpu())
        self.outputs.clear()

    def prepare_reference(self):
        if self.reference_prepared >= self.samples:
            raise AssertionError("More reference preparations than scored samples")
        self._set_input(self.reference_prepared)
        self.reference_prepared += 1
