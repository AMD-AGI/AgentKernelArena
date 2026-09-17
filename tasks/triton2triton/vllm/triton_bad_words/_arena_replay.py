"""Protected task-owned checks for original calls and the actual measured graph.

No reference work or tensor snapshots execute inside the measured callable.
The one unscored probe discovers wrapper arguments and is restored before timing.
"""
import inspect


def tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


def clone(value, *, cpu=False):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone() if cpu else value.detach().clone()
    if isinstance(value, (tuple, list)):
        return type(value)(clone(v, cpu=cpu) for v in value)
    return value


def restore(values, saved):
    for value, before in zip(tensors(values), tensors(saved)):
        value.copy_(before)


def unchanged(args, saved, mutable):
    import torch
    writable = {id(t) for i in mutable for t in tensors(args[i])}
    for value, before in zip(tensors(args), tensors(saved)):
        if id(value) not in writable and (value.shape != before.shape or value.dtype != before.dtype or
                value.device != before.device or not torch.equal(value, before)):
            raise AssertionError('Candidate modified a read-only input')


def to_device(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(v, device) for v in value)
    return value


def compare(actual, expected, *, atol=0.0, rtol=0.0):
    import torch
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise AssertionError('Output count/container violates the operator contract')
        for a, e in zip(actual, expected):
            compare(a, e, atol=atol, rtol=rtol)
        return
    if not isinstance(actual, torch.Tensor) or (actual.shape != expected.shape or
            actual.dtype != expected.dtype or actual.device != expected.device):
        raise AssertionError('Output shape/dtype/device violates the operator contract')
    if actual.is_floating_point():
        if torch.isnan(actual).any() or not torch.equal(torch.isposinf(actual), torch.isposinf(expected)) or not torch.equal(torch.isneginf(actual), torch.isneginf(expected)):
            raise AssertionError('Output NaN/infinity mask violates the reference')
        finite = torch.isfinite(expected)
        torch.testing.assert_close(actual[finite], expected[finite], atol=atol, rtol=rtol)
    elif not torch.equal(actual, expected):
        raise AssertionError('Output differs from the exact integer reference')


def expected(harness, contract, args):
    device = next(tensors(args)).device
    return to_device(contract.reference(harness, clone(args, cpu=True)), device)


def check(harness, contract, output, answer, args):
    if hasattr(contract, 'check'):
        contract.check(harness, output, answer, args)
    else:
        compare(output, answer, atol=contract.ATOL, rtol=contract.RTOL)


class Recorder:
    def __init__(self, harness, contract):
        self.harness, self.contract = harness, contract
        self.mode = 'correctness'
        self.args = self.result = self.saved = None

    def wrap(self, function):
        signature = inspect.signature(function)
        def call(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = tuple(bound.arguments.values())
            self.args = values
            if self.mode == 'correctness':
                saved = clone(values)
                answer = expected(self.harness, self.contract, values)
                result = function(*args, **kwargs)
                unchanged(values, saved, self.contract.MUTABLE)
                check(self.harness, self.contract, self.contract.observe(result, values), answer, values)
            else:
                if self.mode == 'probe':
                    self.saved = clone(values)
                result = function(*args, **kwargs)
            self.result = result
            return result
        return call

    def benchmark(self, benchmark, fn, **options):
        contract = self.contract
        self.args = self.result = self.saved = None
        prepare = options.get('prepare_fn')
        if prepare is not None:
            prepare()
        direct = hasattr(contract, 'direct')
        try:
            self.mode = 'probe'
            if direct:
                self.args, self.result = contract.direct(inspect.getclosurevars(fn).nonlocals)
                self.saved = clone(self.args)
            fn()
            if self.args is None or self.saved is None:
                raise AssertionError('Timed callable did not invoke the declared operator')
            restore(self.args, self.saved)
            answer = expected(self.harness, contract, self.args)
            self.mode = 'timing'
            def measured():
                fn()
                return contract.observe(self.result, self.args)
            timed = self.harness._TimedRun()
            ms, metadata = benchmark(measured, timed_run=timed, **options)
            unchanged(self.args, self.saved, contract.MUTABLE)
            check(self.harness, contract, timed.outputs, answer, self.args)
            # Restore pristine mutable buffers before checking a new input on
            # the exact graph. Existing preparation still runs inside rerun().
            restore(self.args, self.saved)
            contract.fresh(self.args)
            fresh_saved = clone(self.args)
            answer = expected(self.harness, contract, self.args)
            output = timed.rerun()
            unchanged(self.args, fresh_saved, contract.MUTABLE)
            check(self.harness, contract, output, answer, self.args)
            return ms, {**metadata, 'timed_output_checked': True,
                        'replay_input_control_checked': True, 'input_state_restored': True}
        finally:
            if self.args is not None and self.saved is not None:
                restore(self.args, self.saved)
            self.mode = 'performance'


def install(harness, contract):
    recorder = Recorder(harness, contract)
    loader, correctness, performance = harness.load_module, harness.run_correctness, harness.run_performance
    def load():
        module = loader()
        function = getattr(module, contract.FUNCTION)
        setattr(module, contract.FUNCTION, recorder.wrap(function))
        return module
    def checked_correctness(*, case_index=None):
        recorder.mode = 'correctness'
        try:
            if case_index == contract.CONTROL_INDEX:
                contract.controls(harness, getattr(load(), contract.FUNCTION), 'cuda')
                return True, None
            result = correctness(case_index=case_index)
            if case_index is None and result[0]:
                contract.controls(harness, getattr(load(), contract.FUNCTION), 'cuda')
            return result
        except Exception as exc:
            return False, f'{type(exc).__name__}: {exc}'
    def checked_performance():
        recorder.mode = 'performance'
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kw: recorder.benchmark(benchmark, fn, **kw)
        try:
            return performance()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark
            recorder.mode = 'correctness'
    harness.load_module = load
    harness.run_correctness = checked_correctness
    harness.run_performance = checked_performance
