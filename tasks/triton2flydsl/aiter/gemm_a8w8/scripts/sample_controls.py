"""Fresh scaled-GEMM operands and checks around each measured invocation.

Preparation and oracle/immutability checks are outside device timing. The
existing graph helper captures one logical call when prepare_fn is provided,
so a replay cannot amortize a cached accumulator across identical invocations.
"""
from .replay_checks import allclose_output, require_unchanged


def fresh_quantized_operands(x, w, *, seed):
    """Draw independent FP32 operands and quantize with the original row scales."""
    import torch

    quantized = []
    for index, value in enumerate((x, w)):
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed + index)
        raw = torch.randn(value.shape, generator=generator,
                          device=value.device, dtype=torch.float32)
        scale = raw.abs().amax(dim=1, keepdim=True) / torch.finfo(value.dtype).max
        quantized.append(((raw / scale).to(value.dtype), scale))
    (left, left_scale), (right, right_scale) = quantized
    return left, right, left_scale, right_scale.T.contiguous()


class MeasuredScaledGemmInputs:
    def __init__(self, x, w, x_scale, w_scale, bias, *, seed, case_index,
                 samples, output_dtype, reference):
        if samples < 1:
            raise ValueError("At least one measured sample is required")
        self.x, self.w = x, w
        self.x_scale, self.w_scale = x_scale, w_scale
        self.inputs = tuple(value for value in (x, w, x_scale, w_scale, bias)
                            if value is not None)
        self.originals = tuple(value.clone() for value in self.inputs)
        self.seed = seed + 10_000 * (case_index + 1)
        self.samples = samples
        self.output_dtype = output_dtype
        self.reference = reference
        self.checked = 0
        self.preparations = 0
        self.prepared_sample_index = None
        self.measured_draws = []
        self.measuring = True
        self.pending = False
        self.prepared_index = None
        self.prepared_inputs = None
        self.expected_cpu = None

    def _replace_operands(self, index, *, left=True, right=True):
        fresh_x, fresh_w, fresh_x_scale, fresh_w_scale = fresh_quantized_operands(
            self.x, self.w, seed=self.seed + 2 * index)
        if left:
            self.x.copy_(fresh_x)
            self.x_scale.copy_(fresh_x_scale)
        if right:
            self.w.copy_(fresh_w)
            self.w_scale.copy_(fresh_w_scale)

    def prepare(self):
        # Advance on every preparation, including capture/calibration. The
        # external original-input warmups remain in the unchanged harness.
        # Even the first scored invocation must see a new operand pair.
        if not self.measuring:
            return
        if self.checked >= self.samples or self.pending:
            raise AssertionError("Unexpected input preparation after scored samples")
        if self.prepared_inputs is not None:
            require_unchanged(self.inputs, self.prepared_inputs)
        self._replace_operands(self.preparations)
        self.prepared_inputs = tuple(value.clone() for value in self.inputs)
        self.prepared_index = self.preparations
        self.prepared_sample_index = self.checked
        self.preparations += 1
        self.expected_cpu = self.reference().detach().to("cpu", copy=True)

    def before_sample(self, calls):
        if (calls != 1 or self.pending or self.prepared_sample_index != self.checked
                or self.expected_cpu is None):
            raise AssertionError("Each measured sample needs one fresh checked GEMM call")
        require_unchanged(self.inputs, self.prepared_inputs)
        self.pending = True

    def _compare(self, output, expected_cpu):
        import torch

        if (not isinstance(output, torch.Tensor)
                or output.shape != (self.x.shape[0], self.w.shape[0])
                or output.dtype != self.output_dtype or output.device != self.x.device):
            raise AssertionError("Measured scaled GEMM output contract mismatch")
        allclose_output(output.detach().to("cpu", copy=True), expected_cpu,
                        atol=0.02, rtol=0.01)

    def observe(self, output):
        if (not self.pending or self.prepared_sample_index != self.checked
                or self.checked >= self.samples):
            raise AssertionError("Measured output lacks its own prepared inputs")
        require_unchanged(self.inputs, self.prepared_inputs)
        self._compare(output, self.expected_cpu)
        import hashlib
        import json
        import torch
        digest = hashlib.sha256()
        for index, value in enumerate(self.prepared_inputs):
            digest.update(json.dumps([index, list(value.shape), str(value.dtype)]).encode())
            digest.update(value.detach().contiguous().view(torch.uint8).to("cpu").numpy().tobytes())
        self.measured_draws.append({
            "prepare_index": self.prepared_index,
            "operand_seeds": [self.seed + 2 * self.prepared_index,
                              self.seed + 2 * self.prepared_index + 1],
            "input_sha256": digest.hexdigest(),
        })
        self.expected_cpu = None
        self.pending = False
        self.checked += 1

    def finish(self):
        self.measuring = False
        if self.pending or self.checked != self.samples:
            raise AssertionError("Not every reported scaled GEMM sample was checked")
        indices = [record["prepare_index"] for record in self.measured_draws]
        if indices != list(range(indices[0], indices[0] + self.samples)):
            raise AssertionError("Measured GEMM preparation indices are not consecutive")
        if len({record["input_sha256"] for record in self.measured_draws}) != self.samples:
            raise AssertionError("Measured GEMM samples reused identical operands")
        require_unchanged(self.inputs, self.prepared_inputs)

    def validate_replays(self, timed):
        if self.measuring or self.checked != self.samples or not timed.bound:
            raise AssertionError("Measured GEMM evidence is incomplete")
        require_unchanged(self.inputs, self.prepared_inputs)
        self._compare(timed.outputs, self.reference().detach().to("cpu", copy=True))
        measured_inputs = tuple(value.clone() for value in self.inputs)
        try:
            # Each independent operand control starts from the last measured
            # draw. A scale-only replay cannot detect an old FP32 accumulator.
            for index, (left, right) in enumerate(((True, False), (False, True),
                                                    (True, True)), 1):
                for value, original in zip(self.inputs, measured_inputs):
                    value.copy_(original)
                self._replace_operands(self.preparations + index, left=left, right=right)
                changed_inputs = tuple(value.clone() for value in self.inputs)
                expected_cpu = self.reference().detach().to("cpu", copy=True)
                timed.outputs.fill_(float("nan"))
                actual = timed.rerun()
                require_unchanged(self.inputs, changed_inputs)
                self._compare(actual, expected_cpu)
        finally:
            for value, original in zip(self.inputs, measured_inputs):
                value.copy_(original)
        return {"timed_output_correctness": "PASS", "replay_correctness": "PASS",
                "replay_inputs_perturbed": True, "replay_output_poisoned": True,
                "replay_operand_controls_checked": 3,
                "measured_sample_outputs_checked": self.checked,
                "measured_input_stream": "independent_quantized_x_w_every_preparation",
                "measured_input_draws": self.measured_draws,
                "measured_input_distinct_draws": len({record["input_sha256"] for record in self.measured_draws}),
                "measured_calls_per_sample": 1}

    def restore(self):
        for value, original in zip(self.inputs, self.originals):
            value.copy_(original)
