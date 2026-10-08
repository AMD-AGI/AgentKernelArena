"""Observe every measured Event output for the declared BF16 GEMM cases.

Preparation and numerical checks occur outside the Event interval. Sample zero
uses the original declared inputs; later samples use distinct seeded operands
of the same shape and valid dtype. Only one pristine input pair is held live.
"""


class MeasuredInputStream:
    def __init__(self, a, b, *, seed, case_index, samples):
        if samples < 1:
            raise ValueError("At least one measured sample is required")
        self.a, self.b = a, b
        self.original_a, self.original_b = a.clone(), b.clone()
        self.seed, self.case_index, self.samples = seed, case_index, samples
        self.prepared = 0
        self.outputs = []
        self.measuring = True
        self.prepared_a = self.prepared_b = None
        self.output_shape = (a.shape[0], b.shape[0])

    def _set_input(self, index):
        import torch

        if index == 0:
            self.a.copy_(self.original_a)
            self.b.copy_(self.original_b)
            return
        generator = torch.Generator(device=self.a.device)
        generator.manual_seed(self.seed + 10_000 * (self.case_index + 1) + index)
        self.a.copy_(torch.rand(self.a.shape, generator=generator,
                                device=self.a.device, dtype=self.a.dtype))
        self.b.copy_(torch.rand(self.b.shape, generator=generator,
                                device=self.b.device, dtype=self.b.dtype))

    def prepare(self):
        # The canonical eager rerun invokes prepare_fn again. Leave the
        # validator's explicitly perturbed operands intact for that replay.
        if not self.measuring:
            return
        if self.prepared >= self.samples:
            raise AssertionError("More preparations than reported samples")
        self._set_input(self.prepared)
        self.prepared_a = self.a.clone()
        self.prepared_b = self.b.clone()
        self.prepared += 1

    def observe(self, output):
        import torch

        if not isinstance(output, torch.Tensor):
            raise AssertionError("Measured GEMM output must be a Tensor")
        if (output.shape != self.output_shape or output.dtype != self.a.dtype
                or output.device != self.a.device):
            raise AssertionError("Measured GEMM output violates the BF16 device contract")
        if not torch.equal(self.a, self.prepared_a) or not torch.equal(self.b, self.prepared_b):
            raise AssertionError("Operator modified a read-only measured input")
        # A later call may reuse output storage. Keep this completed sample's
        # actual bytes without retaining a graph or a second GPU output.
        self.outputs.append(output.detach().to("cpu", copy=True))

    def validate(self, reference, compare):
        self.measuring = False
        if self.prepared != self.samples or len(self.outputs) != self.samples:
            raise AssertionError("Reported samples lack matching measured outputs")
        last_expected = None
        for index, actual in enumerate(self.outputs):
            self._set_input(index)
            last_expected = reference()
            compare(actual, last_expected.cpu())
        return last_expected

    def restore(self):
        self.measuring = False
        self.a.copy_(self.original_a)
        self.b.copy_(self.original_b)
