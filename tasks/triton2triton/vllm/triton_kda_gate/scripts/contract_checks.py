"""Protected output, read-only input, and actual timed-replay checks.

Copied within each task package so an isolated workspace needs no Arena imports.
All comparisons, snapshots, perturbations and poisoning run outside timing.
"""
import torch


class ContractFailure(RuntimeError):
    pass


class NumericalMismatch(ContractFailure):
    pass


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in tensors(item)]
    raise ContractFailure('Expected a tensor or complete tensor tuple')


class InputSnapshot:
    def __init__(self, inputs):
        self.entries = [(name, x, x.detach().clone(), x.shape, x.stride(), x.dtype, x.device)
                        for name, x in inputs.items() if x is not None]

    def check(self):
        for name, x, saved, shape, stride, dtype, device in self.entries:
            if (x.shape != shape or x.stride() != stride or x.dtype != dtype or
                    x.device != device or not torch.equal(x, saved)):
                raise ContractFailure(f'Read-only input changed: {name}')

    def restore(self):
        for _, x, saved, *_ in self.entries:
            x.copy_(saved)


class MeasuredOutputReset:
    """Poison the captured output before, and outside, each measured replay."""

    def __init__(self):
        self.output = None
        self.enabled = True
        self.resets_for_output = 0

    def bind_output(self, output):
        if output is not self.output:
            self.output = output
            self.resets_for_output = 0

    def __call__(self):
        if self.enabled and self.output is not None:
            for output in tensors(self.output):
                output.fill_(float('nan'))
            self.resets_for_output += 1


class MeasuredInputStream:
    """Bind one distinct, precomputed input and oracle to each reported replay.

    Warmup and capture repeatedly use sample zero. Only a successful measured
    observation advances the stream. Preparation and input guards run before
    the start Event; the observer reads only the returned output.
    """

    def __init__(self, live_g, live_A, variants, expected, output_reset):
        if not variants or len(variants) != len(expected):
            raise ContractFailure('Incomplete measured input stream')
        self.live_g, self.live_A = live_g, live_A
        self.variants, self.expected = variants, expected
        self.output_reset = output_reset
        self.sample_index = 0
        self.current_index = None
        self.preparations = 0
        self.changed_replay = False

    def check_current(self):
        if self.current_index is None:
            return
        g, A = self.variants[self.current_index]
        if not torch.equal(self.live_g, g) or not torch.equal(self.live_A, A):
            raise ContractFailure('Measured input changed before next preparation')

    def prepare(self):
        if self.changed_replay:
            return  # Preserve the perturbed inputs and previous valid output.
        self.check_current()
        if self.sample_index >= len(self.variants):
            raise ContractFailure('Measured input stream exhausted')
        g, A = self.variants[self.sample_index]
        self.live_g.copy_(g)
        self.live_A.copy_(A)
        self.current_index = self.sample_index
        self.preparations += 1
        self.output_reset()

    def observe(self, outputs, *, atol, rtol):
        if self.current_index != self.sample_index or self.sample_index >= len(self.expected):
            raise ContractFailure('Measured output has no prepared input and oracle')
        check_outputs(outputs, self.expected[self.sample_index], atol=atol, rtol=rtol,
                      inputs=(self.live_g, self.live_A))
        self.sample_index += 1


def check_outputs(actual, expected, *, atol, rtol, inputs=()):
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise ContractFailure('Missing or extra output tuple members')
        for got, ref in zip(actual, expected):
            check_outputs(got, ref, atol=atol, rtol=rtol, inputs=inputs)
        return
    if not isinstance(actual, torch.Tensor):
        raise ContractFailure('Output is not a tensor')
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise ContractFailure('Output shape, dtype or device differs from contract')
    if any(actual.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
           for x in inputs if x is not None):
        raise ContractFailure('Output aliases a read-only input')
    if not torch.isfinite(actual).all():
        raise ContractFailure('Output contains nonfinite values')
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        delta = (actual.float() - expected.float()).abs().max().item()
        raise NumericalMismatch(f'Output numerical mismatch: max_abs_error={delta}')


def comparator_control(expected, *, atol, rtol):
    # A dense wrong answer must fail at the very same task gate. This is not
    # a claim that comparing a reference with itself validates its mathematics.
    bad = tuple(x + 100 for x in expected) if isinstance(expected, tuple) else expected + 100
    try:
        check_outputs(bad, expected, atol=atol, rtol=rtol)
    except NumericalMismatch:
        return 'rejected'
    raise ContractFailure('Comparator accepted deliberately wrong output')


def observe_measured_samples(timed, readonly, reference, *, atol, rtol):
    """Check each reported sample after its end event, outside device timing."""
    expected = reference()
    timed.sample_checks = 0
    def check_sample(outputs):
        check_outputs(outputs, expected, atol=atol, rtol=rtol)
        timed.sample_checks += 1
    timed.after_sample = check_sample


def validate_timed(timed, readonly, reference, perturb, *, atol, rtol, expected_samples,
                   output_reset=None, input_stream=None):
    inputs = [entry[1] for entry in readonly.entries]
    try:
        if not timed.bound:
            raise ContractFailure('No observable measured invocation')
        if timed.sample_checks != expected_samples:
            raise ContractFailure('Reported samples were not all checked')
        if input_stream is None:
            readonly.check()
            expected_last = reference()
        else:
            if input_stream.sample_index != expected_samples:
                raise ContractFailure('Measured input stream was not fully checked')
            input_stream.check_current()
            expected_last = input_stream.expected[-1]
        check_outputs(timed.outputs, expected_last, atol=atol, rtol=rtol, inputs=inputs)
        before_changed = InputSnapshot({name: x for name, x, *_ in readonly.entries})
        perturb()
        if not any(not torch.equal(x, saved) for _, x, saved, *_ in before_changed.entries):
            raise ContractFailure('Replay perturbation did not change input values')
        changed = InputSnapshot({name: x for name, x, *_ in readonly.entries})
        expected = reference()
        try:
            check_outputs(timed.outputs, expected, atol=atol, rtol=rtol, inputs=inputs)
        except NumericalMismatch:
            pass
        else:
            raise ContractFailure('Changed input did not change the expected output')
        # An ordinary changed-input call starts with the previous valid output.
        # Poisoning here would force a cache-on-valid-output implementation to
        # compute only for this control, hiding its incorrect normal behavior.
        if output_reset is not None:
            output_reset.enabled = False
        if input_stream is not None:
            input_stream.changed_replay = True
        try:
            replayed = timed.rerun()
        finally:
            if output_reset is not None:
                output_reset.enabled = True
            if input_stream is not None:
                input_stream.changed_replay = False
        changed.check()
        check_outputs(replayed, expected, atol=atol, rtol=rtol, inputs=inputs)
    finally:
        readonly.restore()
    readonly.check()
    return {'timed_output_correctness': 'PASS', 'replay_correctness': 'PASS',
            'readonly_inputs': 'PASS', 'replay_inputs_changed': True,
            'input_state_restored': True,
            'changed_input_replay_unpoisoned': True,
            'output_coverage': 'all_output_tensors_and_elements',
            'measured_samples_checked': timed.sample_checks}
