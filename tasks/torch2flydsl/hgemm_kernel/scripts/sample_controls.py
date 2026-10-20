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
        self.reference_prepared = 0
        self.checked = 0
        self.reference = None
        self.compare = None
        self.last_expected = None
        self.expected_cpu = None
        self.measuring = True
        self.prepared_a = self.prepared_b = None
        self.output_shape = (a.shape[0], b.shape[0])

    def bind(self, reference, compare):
        if self.prepared or self.checked:
            raise AssertionError("Measured reference must be bound before sampling")
        self.reference, self.compare = reference, compare

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
        if self.reference is None or self.compare is None:
            raise AssertionError("Measured reference must be bound before sampling")
        self.last_expected = self.reference()
        self.expected_cpu = self.last_expected.detach().to("cpu", copy=True)
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
        if self.reference is None or self.compare is None or self.checked + 1 != self.prepared:
            raise AssertionError("Measured sample lacks its bound reference or preparation")
        # The Event has ended. Read these bytes before later output reuse.
        actual = output.detach().to("cpu", copy=True)
        self.compare(actual, self.expected_cpu)
        self.expected_cpu = None
        self.checked += 1

    def validate(self, reference, compare):
        self.measuring = False
        if (reference is not self.reference or compare is not self.compare
                or self.prepared != self.samples or self.checked != self.samples):
            raise AssertionError("Reported samples lack matching checked outputs")
        return self.last_expected

    def prepare_reference(self):
        if self.reference_prepared >= self.samples:
            raise AssertionError("More diagnostic reference samples than scored samples")
        self._set_input(self.reference_prepared)
        self.reference_prepared += 1

    def restore(self):
        self.measuring = False
        self.a.copy_(self.original_a)
        self.b.copy_(self.original_b)
