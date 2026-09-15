"""Validate weighted expert gather, disabled routes, and exact timed output."""
from contextlib import contextmanager
import inspect


def snapshots(values):
    return tuple(value.clone() for value in values)


def unchanged(values, saved):
    import torch
    for value, original in zip(values, saved):
        if not torch.equal(value, original):
            raise AssertionError('Expert gather modified a read-only input')


def reference(harness, inputs, output):
    data, ids, weights, indices = (value.cpu() for value in inputs)
    expected = harness.reference_gather(data, ids, weights, indices, *output.shape)
    return expected.to(dtype=output.dtype, device=output.device)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError('Expert gather output shape/dtype/device is invalid')
    if not torch.isfinite(value).all():
        raise AssertionError('Expert gather output must be finite')
    torch.testing.assert_close(value.float(), expected.float(), atol=5e-2, rtol=5e-2)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = module.ep_gather
        patched.append((module, original))
        diagnosed = False

        def verify(data, ids, weights, indices, output):
            inputs = (data, ids, weights, indices)
            pristine = snapshots(inputs)
            expected = reference(harness, pristine, output)
            try:
                result = original(data, ids, weights, indices, output)
                unchanged(inputs, pristine)
                check_output(output, expected)
                return result
            finally:
                for value, saved in zip(inputs, pristine):
                    value.copy_(saved)

        def checked(input_tensor, recv_topk_ids, recv_topk_weight, input_index, output_tensor):
            nonlocal diagnosed
            import torch
            result = verify(input_tensor, recv_topk_ids, recv_topk_weight, input_index, output_tensor)
            if not diagnosed:
                # Unscored: >1024 tokens exercises the grid-stride token loop;
                # 2048 columns exercises the second hidden block. Disabled
                # routes have deliberately invalid indices and must not read.
                device = input_tensor.device
                data = ((torch.arange(13*2048, device=device).reshape(13,2048)%17)/8+.25).to(input_tensor.dtype)
                ids = torch.tensor([[0,-1,2],[1,2,3],[-1,-1,-1]], device=device, dtype=recv_topk_ids.dtype)
                ids = ids[torch.arange(1025, device=device)%3].clone()
                weights = torch.tensor([.5,-.75,1.25], device=device, dtype=recv_topk_weight.dtype).expand(1025,3).clone()
                indices = (torch.arange(1025*3, device=device).reshape(1025,3)%13).to(input_index.dtype)
                indices[ids<0] = -999
                output = torch.full((2050,2048), float('nan'), device=device, dtype=output_tensor.dtype)[::2]
                verify(data, ids, weights, indices, output)
                diagnosed = True
            return result

        module.ep_gather = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            module.ep_gather = original


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    inputs = tuple(state[name] for name in ('input_tensor','topk_ids','topk_weight','input_index'))
    output = state['output_tensor']
    pristine = snapshots(inputs)
    saved_output = output.clone()
    expected = reference(harness, pristine, output)

    def measured():
        fn()
        return output

    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        inputs[0].mul_(-.5).add_(.25)
        inputs[1].add_(1).remainder_(8)
        inputs[1][::2,0] = -1
        inputs[2].mul_(1.25).add_(.5)
        inputs[3].add_(7).remainder_(inputs[0].shape[0])
        replay_inputs = snapshots(inputs)
        expected_replay = reference(harness, replay_inputs, output)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_output(replayed, expected_replay)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        for value, saved in zip(inputs, pristine):
            value.copy_(saved)
        output.copy_(saved_output)


def install(harness):
    correctness, performance = harness.run_correctness, harness.run_performance

    def checked_correctness(*args, **kwargs):
        with checked_modules(harness):
            return correctness(*args, **kwargs)

    def checked_performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(harness, benchmark, fn, **kwargs)
        try:
            return performance()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_correctness, harness.run_performance = checked_correctness, checked_performance
