"""Distinct scored inputs and read-only output capture for the a4w4 Event run.

Only the operator call is timed. Input preparation and oracle checks run outside
the Event interval for both baseline and candidate. Sample zero is the original
declared input; every later sample has a reproducible, independently generated
BF16 activation and weight tensor of the same shape.
"""


class MeasuredInputStream:
    def __init__(self, a, w, *, seed, case_index, samples):
        if samples < 1:
            raise ValueError("At least one measured sample is required")
        self.a, self.w = a, w
        self.output_shape = (a.shape[0], w.shape[0])
        self.output_dtype = a.dtype
        self.output_device = a.device
        self.original_a, self.original_w = a.clone(), w.clone()
        self.seed = seed
        self.case_index = case_index
        self.samples = samples
        self.prepared = 0
        self.outputs = []
        self.measuring = True
        self.prepared_a = None
        self.prepared_w = None

    def _set_input(self, index):
        import torch

        if index == 0:
            self.a.copy_(self.original_a)
            self.w.copy_(self.original_w)
            return
        generator = torch.Generator(device=self.a.device)
        generator.manual_seed(self.seed + 10_000 * (self.case_index + 1) + index)
        self.a.copy_(torch.randn(self.a.shape, generator=generator,
                                 device=self.a.device, dtype=self.a.dtype))
        self.w.copy_(torch.randn(self.w.shape, generator=generator,
                                 device=self.w.device, dtype=self.w.dtype))

    def prepare(self):
        # TimedRun.rerun invokes the same preparation callback; leave the
        # validator's explicitly perturbed operands intact for that replay.
        if not self.measuring:
            return
        if self.prepared >= self.samples:
            raise AssertionError("More input preparations than scored samples")
        self._set_input(self.prepared)
        # Retain only this sample's pristine operands. The observer checks that
        # the candidate did not alter them before regeneration can erase it.
        self.prepared_a = self.a.clone()
        self.prepared_w = self.w.clone()
        self.prepared += 1

    def observe(self, output):
        # The callback runs after the end event. It observes the output and
        # verifies that the operator left the prepared operands unchanged.
        # Copying the output to host storage prevents a later sample from
        # overwriting the bytes belonging to this exact measured invocation.
        import torch

        if not isinstance(output, torch.Tensor):
            raise AssertionError("Measured output must be a Tensor")
        if (output.shape != self.output_shape or output.dtype != self.output_dtype
                or output.device != self.output_device):
            raise AssertionError("Measured output violates BF16 device contract")
        if not torch.equal(self.a, self.prepared_a) or not torch.equal(self.w, self.prepared_w):
            raise AssertionError("Operator modified a read-only measured input")
        self.outputs.append(output.detach().to("cpu", copy=True))

    def validate(self, reference, compare):
        """Check every reported output, then restore the final sample inputs."""
        self.measuring = False
        if self.prepared != self.samples or len(self.outputs) != self.samples:
            raise AssertionError("Reported samples lack matching inputs or outputs")
        last_expected = None
        for index, actual in enumerate(self.outputs):
            self._set_input(index)
            last_expected = reference()
            compare(actual, last_expected.cpu())
        return last_expected
