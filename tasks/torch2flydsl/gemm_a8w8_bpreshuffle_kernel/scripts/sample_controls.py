"""Prepared quantized operands and actual Event outputs for every scored sample.

Preparation and protected FP32 oracle checks remain outside Event intervals.
The first sample is the original declared input; later samples use separate
seeded BF16 draws that are independently quantized and preshuffled.
"""


class MeasuredQuantizedStream:
    def __init__(self, inputs, *, seed, case_index, samples, make_inputs,
                 quantize, checked_preshuffle, check_unchanged):
        if samples < 1:
            raise ValueError("At least one measured sample is required")
        self.inputs = inputs
        self.originals = tuple(value.clone() for value in inputs)
        self.seed, self.case_index, self.samples = seed, case_index, samples
        self.make_inputs = make_inputs
        self.quantize = quantize
        self.checked_preshuffle = checked_preshuffle
        self.check_unchanged = check_unchanged
        self.prepared = 0
        self.prepared_inputs = None
        self.outputs = []
        self.measuring = True
        self.output_shape = (inputs[0].shape[0], inputs[1].shape[0])
        self.output_device = inputs[0].device

    def _set_input(self, index):
        if index == 0:
            for target, original in zip(self.inputs, self.originals):
                target.copy_(original)
            return
        x, weight = self.inputs[:2]
        alternate_x, alternate_weight = self.make_inputs(
            x.shape[0], weight.shape[0], x.shape[1], device=x.device,
            seed=self.seed + 10_000 * (self.case_index + 1) + index,
        )
        xq, x_scale = self.quantize(alternate_x)
        wq, w_scale = self.quantize(alternate_weight)
        packed_wq = self.checked_preshuffle(wq)
        sources = (alternate_x, alternate_weight, xq, wq, packed_wq,
                   x_scale, w_scale)
        for target, source in zip(self.inputs, sources):
            if (target.shape != source.shape or target.dtype != source.dtype
                    or target.device != source.device):
                raise AssertionError("Measured quantized input changed its declared contract")
            target.copy_(source)

    def prepare(self):
        # The eager TimedRun.rerun uses prepare_fn again. Leave its explicit
        # changed-input replay intact once measurement has finished.
        if not self.measuring:
            return
        if self.prepared >= self.samples:
            raise AssertionError("More preparations than reported samples")
        self._set_input(self.prepared)
        self.prepared_inputs = tuple(value.clone() for value in self.inputs)
        self.prepared += 1

    def observe(self, output):
        import torch

        if not isinstance(output, torch.Tensor):
            raise AssertionError("Measured quantized GEMM output must be a Tensor")
        if (output.shape != self.output_shape or output.dtype != torch.bfloat16
                or output.device != self.output_device):
            raise AssertionError("Measured quantized GEMM output violates its contract")
        self.check_unchanged(self.inputs, self.prepared_inputs)
        # Retain the full returned bytes. This is necessary for the original
        # per-element numerical gate; a checksum would not establish it.
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
        for target, original in zip(self.inputs, self.originals):
            target.copy_(original)


class RawReferenceStream:
    """Give the diagnostic PyTorch timing the identical raw BF16 sample stream."""

    def __init__(self, x, weight, *, seed, case_index, samples, make_inputs):
        self.x, self.weight = x, weight
        self.original_x, self.original_weight = x.clone(), weight.clone()
        self.seed, self.case_index, self.samples = seed, case_index, samples
        self.make_inputs = make_inputs
        self.prepared = 0

    def prepare(self):
        if self.prepared >= self.samples:
            raise AssertionError("More reference preparations than samples")
        if self.prepared == 0:
            self.x.copy_(self.original_x)
            self.weight.copy_(self.original_weight)
        else:
            x, weight = self.make_inputs(
                self.x.shape[0], self.weight.shape[0], self.x.shape[1],
                device=self.x.device,
                seed=self.seed + 10_000 * (self.case_index + 1) + self.prepared,
            )
            self.x.copy_(x)
            self.weight.copy_(weight)
        self.prepared += 1

    def restore(self):
        self.x.copy_(self.original_x)
        self.weight.copy_(self.original_weight)
