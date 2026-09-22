"""Protected checks around the canonical timer; no extra work inside timing."""
import torch
from _aka_benchmark import TimedRun


class PristineInputs:
    """Keep independent oracle inputs and verify every read-only tensor byte."""
    def __init__(self, tensors):
        self.live = dict(tensors)
        self.saved = {name: value.detach().clone() for name, value in tensors.items()}
        self.layout = {name: (value.shape, value.stride(), value.dtype, value.device)
                       for name, value in tensors.items()}

    def check(self, expected=None):
        expected = self.saved if expected is None else expected
        for name, value in self.live.items():
            assert (value.shape, value.stride(), value.dtype, value.device) == self.layout[name], (
                'Read-only input layout changed', name)
            assert torch.equal(value.detach().clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8),
                               expected[name].clone(memory_format=torch.contiguous_format).reshape(-1).view(torch.uint8)), (
                'Read-only input changed', name)

    def restore(self, values=None):
        values = self.saved if values is None else values
        with torch.no_grad():
            for name, value in self.live.items():
                value.copy_(values[name])


def assert_output_contract(actual, expected):
    """Check all public outputs, including optional fields and FP8 finiteness."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), 'Missing tensor output'
        assert actual.shape == expected.shape, 'Wrong output shape'
        assert actual.dtype == expected.dtype, 'Wrong output dtype'
        assert actual.device == expected.device, 'Wrong output device'
        if actual.is_floating_point():
            assert torch.isfinite(actual.float()).all(), 'Nonfinite output'
            assert torch.isfinite(expected.float()).all(), 'Nonfinite oracle'
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)) and len(actual) == len(expected), 'Wrong output structure'
        for a, e in zip(actual, expected):
            assert_output_contract(a, e)
    else:
        assert actual is expected, 'Wrong optional output'


def _poison(outputs, readonly):
    if isinstance(outputs, torch.Tensor):
        # Some operators legitimately return a read-only input view. Perturbed
        # input + full oracle comparison still checks that output without
        # overwriting the input through its alias.
        if any(torch._C._overlaps(outputs, x) for x in readonly):
            return
        with torch.no_grad():
            if outputs.is_floating_point():
                outputs.fill_(float('nan'))
            else:
                outputs.fill_(torch.iinfo(outputs.dtype).min)
    elif isinstance(outputs, (tuple, list)):
        for output in outputs:
            _poison(output, readonly)


def checked_call(fn, *, inputs, reference, check):
    guard = PristineInputs(inputs)
    try:
        expected = reference(guard.saved)  # before any candidate call
        actual = fn()
        guard.check()
        assert_output_contract(actual, expected)
        check(actual, expected)
        return actual
    finally:
        guard.restore()


def checked_benchmark(benchmark, fn, *, inputs, reference, check, perturb, **options):
    """Check the original measured output and the exact bound graph replay.

    Reference evaluation, input copies and poison operations are outside the
    canonical timing call. Warmups, samples, allocation boundaries and adaptive
    graph repetition remain entirely controlled by the original timer/options.
    """
    guard = PristineInputs(inputs)
    try:
        expected = reference(guard.saved)
        timed = TimedRun()
        ms, metadata = benchmark(fn, timed_run=timed, **options)
        assert timed.bound, 'Timer did not expose its actual invocation'
        guard.check()
        assert_output_contract(timed.outputs, expected)
        check(timed.outputs, expected)

        changed = perturb(dict(guard.saved))
        assert set(changed) == set(guard.saved), 'Perturbation changed input contract'
        changed_expected = reference(changed)  # private inputs, before replay
        guard.restore(changed)
        _poison(timed.outputs, guard.live.values())
        replayed = timed.rerun()
        guard.check(changed)
        assert_output_contract(replayed, changed_expected)
        check(replayed, changed_expected)
        metadata.update(benchmark_original_output_checked=True,
                        benchmark_replay_checked=True,
                        benchmark_readonly_inputs_checked=True)
        return ms, metadata
    finally:
        guard.restore()
