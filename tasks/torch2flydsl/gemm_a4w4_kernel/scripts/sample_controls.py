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
        self.reference_prepared = 0
        self.checked = 0
        self.reference = None
        self.compare = None
        self.last_expected = None
        self.expected_cpu = None
        self.measuring = True
        self.prepared_a = None
        self.prepared_w = None

    def bind(self, reference, compare):
        if self.prepared or self.checked:
            raise AssertionError("Measured reference must be bound before sampling")
        self.reference, self.compare = reference, compare

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
        if self.reference is None or self.compare is None:
            raise AssertionError("Measured reference must be bound before sampling")
        # Compute this one oracle before its start Event. The output observer
        # only reads completed values and never launches reference GPU work.
        self.last_expected = self.reference()
        self.expected_cpu = self.last_expected.detach().to("cpu", copy=True)
        self.prepared += 1

    def observe(self, output):
        # The callback runs after the end event. Check this exact output before
        # a later call can reuse its storage, retaining only the last oracle.
        import torch

        if not isinstance(output, torch.Tensor):
            raise AssertionError("Measured output must be a Tensor")
        if (output.shape != self.output_shape or output.dtype != self.output_dtype
                or output.device != self.output_device):
            raise AssertionError("Measured output violates BF16 device contract")
        if not torch.equal(self.a, self.prepared_a) or not torch.equal(self.w, self.prepared_w):
            raise AssertionError("Operator modified a read-only measured input")
        if self.reference is None or self.compare is None or self.checked + 1 != self.prepared:
            raise AssertionError("Measured sample lacks its bound reference or preparation")
        actual = output.detach().to("cpu", copy=True)
        self.compare(actual, self.expected_cpu)
        self.expected_cpu = None
        self.checked += 1

    def validate(self, reference, compare):
        """Confirm that every reported Event sample was checked."""
        self.measuring = False
        if (reference is not self.reference or compare is not self.compare
                or self.prepared != self.samples or self.checked != self.samples):
            raise AssertionError("Reported samples lack matching checked inputs or outputs")
        return self.last_expected

    def prepare_reference(self):
        if self.reference_prepared >= self.samples:
            raise AssertionError("More diagnostic reference samples than scored samples")
        self._set_input(self.reference_prepared)
        self.reference_prepared += 1
