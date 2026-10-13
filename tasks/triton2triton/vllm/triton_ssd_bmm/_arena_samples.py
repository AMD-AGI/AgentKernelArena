"""One complete public call and independent oracle per reported device sample."""
import torch


def _tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [x for part in value for x in _tensors(part)]
    raise AssertionError('Expected complete tensor outputs')


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return type(value)(_clone(x) for x in value)


def _bytes(value):
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


class Samples:
    def __init__(self, inputs, varied, reference, compare, count, *, atol, rtol):
        self.inputs, self.varied = inputs, varied
        self.reference, self.compare = reference, compare
        self.count, self.atol, self.rtol = count, atol, rtol
        self.original = {key: x.clone() for key, x in inputs.items()}
        self.current = {key: x.clone() for key, x in inputs.items()}
        self.metadata = {key: (x.shape, x.stride(), x.dtype, x.device, x.data_ptr())
                         for key, x in inputs.items()}
        self.expected = _clone(reference())
        self.sample = self.prepared = 0
        self.output = None
        self.resets = 0
        self.control = False

    def check_inputs(self):
        for key, x in self.inputs.items():
            if ((x.shape, x.stride(), x.dtype, x.device, x.data_ptr()) != self.metadata[key]
                    or not torch.equal(_bytes(x), _bytes(self.current[key]))):
                raise AssertionError(f'Measured input changed: {key}')

    def draw(self, index):
        self.check_inputs()
        for attempt in range(8):
            for position, key in enumerate(self.varied):
                x = self.inputs[key]
                generator = torch.Generator(device=x.device).manual_seed(
                    910003 + index * 101 + attempt * 100003 + position * 1009)
                x.copy_(torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator))
            expected = _clone(self.reference())
            wanted, previous = _tensors(expected), _tensors(self.expected)
            if len(wanted) != len(previous) or any(not torch.isfinite(x).all() for x in wanted):
                raise AssertionError('Invalid independent measured oracle')
            if any(not torch.allclose(x, old, atol=self.atol, rtol=self.rtol)
                   for x, old in zip(wanted, previous)):
                self.expected = expected
                self.current = {key: x.clone() for key, x in self.inputs.items()}
                return
        raise AssertionError('New measured inputs have no distinguishable answer')

    def bind(self, output):
        if output is not self.output:
            self.output, self.resets = output, 0
        return output

    def poison(self):
        if self.output is None:
            return
        outputs, expected = _tensors(self.output), _tensors(self.expected)
        if len(outputs) != len(expected):
            raise AssertionError('Missing measured output')
        for x, ref in zip(outputs, expected):
            if (x.shape, x.dtype, x.device) != (ref.shape, ref.dtype, ref.device):
                raise AssertionError('Measured output metadata mismatch')
            if any(torch._C._overlaps(x, value) for value in self.inputs.values()):
                raise AssertionError('Measured output aliases read-only input')
            x.fill_(float('nan'))
        self.resets += 1

    def prepare(self):
        if self.control:
            return
        self.check_inputs()
        if self.sample >= self.count:
            raise AssertionError('Measured input stream exhausted')
        if self.prepared != self.sample:
            self.draw(self.sample)
            self.prepared = self.sample
        self.poison()

    def observe(self, output):
        if self.prepared != self.sample:
            raise AssertionError('Sample has no prepared independent oracle')
        # Read-only after the end Event; oracle work and refills precede timing.
        self.compare(output, self.expected)
        self.sample += 1

    def restore(self):
        for key, x in self.inputs.items():
            x.copy_(self.original[key])
        self.current = self.original
        self.check_inputs()


def measure(benchmark, timed, call, inputs, varied, reference, compare, *,
            warmup, repetition, atol, rtol):
    stream = Samples(inputs, varied, reference, compare, repetition, atol=atol, rtol=rtol)
    timed.after_sample = stream.observe
    try:
        ms, metadata = benchmark(lambda: stream.bind(call()), warmup=warmup,
                                 repetition=repetition, max_graph_repeats=1,
                                 prepare_fn=stream.prepare, timed_run=timed)
        graph = (metadata.get('benchmark_method') == 'cuda_graph'
                 and metadata.get('benchmark_timed_run_kind') == 'captured_graph')
        event = (metadata.get('benchmark_method') == 'cuda_event_fallback'
                 and metadata.get('benchmark_timed_run_kind') == 'eager_callable'
                 and bool(metadata.get('benchmark_fallback_reason')))
        if (not timed.bound or not (graph or event)
                or metadata.get('benchmark_effective_repeats') != 1
                or stream.sample != repetition or stream.output is not timed.outputs
                or (graph and stream.resets < repetition)):
            raise AssertionError('Every one-call reported sample must be observed')
        stream.check_inputs()
        compare(timed.outputs, stream.expected)
        stream.poison()
        stream.control = True
        compare(timed.rerun(), stream.expected)
        stream.check_inputs()
        # Keep the last valid output for the changed-input replay: poisoning
        # here would conceal a kernel that only computes when it sees NaN.
        stream.draw(repetition)
        compare(timed.rerun(), stream.expected)
        stream.check_inputs()
        return ms, {**metadata, 'timed_output_checked': True,
                    'timed_output_correctness': 'PASS', 'replay_correctness': 'PASS',
                    'readonly_inputs': 'PASS', 'replay_inputs_changed': True,
                    'input_state_restored': True, 'changed_input_replay_unpoisoned': True,
                    'measured_samples_checked': stream.sample,
                    'measured_input_variants': stream.sample,
                    'output_coverage': 'all_output_tensors_and_elements'}
    finally:
        stream.restore()
