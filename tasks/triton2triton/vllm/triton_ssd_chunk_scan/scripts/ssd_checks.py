"""Protected SSD output, read-only input, and actual timed-replay checks.

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


def validate_timed(timed, readonly, reference, perturb, *, atol, rtol):
    if not timed.bound:
        raise ContractFailure('No observable measured invocation')
    inputs = [entry[1] for entry in readonly.entries]
    named = {name: value for name, value, *_ in readonly.entries}
    scored_scan = {'cb', 'x', 'dt', 'dA_cumsum', 'C', 'states', 'seq_idx'} <= set(named)
    readonly.check()
    expected = reference()
    if scored_scan:
        from scripts.semantic_controls import inter_chunk_reference, require_intra_signal
        without_intra = inter_chunk_reference(
            named['x'], named['dA_cumsum'], named['C'], named['states'],
            named['seq_idx'], named['dt'].shape[-1])
        require_intra_signal(expected, without_intra)
    check_outputs(timed.outputs, expected, atol=atol, rtol=rtol, inputs=inputs)
    try:
        perturb()
        if scored_scan:
            # Preserve the historical measured input, then make the intra term
            # observable when checking the exact same graph outside timing.
            named['cb'].mul_(32.0)
        if not any(not torch.equal(x, saved) for _, x, saved, *_ in readonly.entries):
            raise ContractFailure('Replay perturbation did not change input values')
        changed = InputSnapshot({name: x for name, x, *_ in readonly.entries})
        expected = reference()
        if scored_scan:
            from scripts.semantic_controls import inter_chunk_reference, require_intra_signal
            without_intra = inter_chunk_reference(
                named['x'], named['dA_cumsum'], named['C'], named['states'],
                named['seq_idx'], named['dt'].shape[-1])
            require_intra_signal(expected, without_intra)
        for output in tensors(timed.outputs):
            output.fill_(float('nan'))
        replayed = timed.rerun()
        changed.check()
        check_outputs(replayed, expected, atol=atol, rtol=rtol, inputs=inputs)
    finally:
        readonly.restore()
    return {'timed_output_correctness': 'PASS', 'replay_correctness': 'PASS',
            'readonly_inputs': 'PASS', 'replay_inputs_changed': True,
            'strong_signal_replay_checked': scored_scan,
            'output_coverage': 'all_output_tensors_and_elements'}
