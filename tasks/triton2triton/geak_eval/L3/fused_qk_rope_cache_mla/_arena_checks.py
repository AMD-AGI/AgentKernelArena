"""Preserve MLA's complete output contract and its prepared measured replay."""
import inspect
from _aka_benchmark import TimedRun


def snapshot(inp):
    import torch
    return {key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in inp.items()}


def unchanged(inp, saved):
    import torch
    for key, original in saved.items():
        if isinstance(original, torch.Tensor):
            value = inp[key]
            if (value.shape != original.shape or value.dtype != original.dtype or
                    value.device != original.device or not torch.equal(
                        value.contiguous().reshape(-1).view(torch.uint8),
                        original.contiguous().reshape(-1).view(torch.uint8))):
                raise AssertionError(f'MLA modified read-only input {key}')


def restore(inp, saved):
    import torch
    for key, original in saved.items():
        if isinstance(original, torch.Tensor):
            inp[key].copy_(original)


def check_output(values, expected, inp):
    import torch
    if not isinstance(values, (tuple, list)) or len(values) != 5:
        raise AssertionError('MLA must return all four outputs and the updated cache')
    for index, (value, reference) in enumerate(zip(values, expected)):
        dtype = inp['cache_dtype_actual'] if index == 4 and inp['cache_dtype'] == torch.uint8 else reference.dtype
        if not isinstance(value, torch.Tensor) or (value.shape != reference.shape or
                value.dtype != dtype or value.device != reference.device):
            raise AssertionError(f'MLA output {index} shape/dtype/device is invalid')
        if index == 4 and inp['cache_dtype'] == torch.uint8:
            value = value.to(inp['dtype'])
            reference = reference.view(dtype).to(inp['dtype'])
        # Keep the original gate, including its FP8->BF16 cache comparison and
        # full-cache check. assert_close rejects NaN; no new finite-only rule.
        torch.testing.assert_close(value, reference, atol=1e-1, rtol=1e-1)


def checked_benchmark(harness, benchmark, fn, **options):
    import torch
    state = inspect.getclosurevars(fn).nonlocals
    inp, cache = state['inp'], state['kv_cache_clone']
    pristine = snapshot(inp)
    cache_before = cache.clone()
    expected = harness._run_reference(pristine)
    prepare = options['prepare_fn']
    replay_only = False

    def measured():
        return (*fn(), cache)

    def prepared():
        prepare()
        if replay_only:
            # Keep the original scored reset. Poison only after that reset in
            # the unscored exact replay; a skipped cache write must remain NaN.
            slots = inp['slot_mapping'].long()
            poison = torch.full((slots.numel(), *cache.shape[1:]), float('nan'),
                                dtype=torch.float32, device=cache.device).to(cache.dtype)
            cache.view(torch.uint8).index_copy_(0, slots, poison.view(torch.uint8))

    try:
        timed = TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed,
                                 **{**options, 'prepare_fn': prepared})
        unchanged(inp, pristine)
        check_output(timed.outputs, expected, pristine)
        inp['q_nope'].mul_(-.5)
        inp['q_pe'].neg_()
        inp['k_lora'].mul_(-.75)
        inp['k_pe'].mul_(.5)
        inp['k_scale'].mul_(.75)
        inp['positions'].add_(1).remainder_(inp['freqs'].shape[0])
        inp['slot_mapping'].copy_(inp['slot_mapping'].flip(0))
        replay_inputs = snapshot(inp)
        replay_expected = harness._run_reference(replay_inputs)
        for output in timed.outputs[:4]:
            output.fill_(float('nan'))
        replay_only = True
        replayed = timed.rerun()
        unchanged(inp, replay_inputs)
        check_output(replayed, replay_expected, replay_inputs)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True,
                    'prepared_cache_replay_checked': True,
                    'source_buffers_unchanged': True}
    finally:
        restore(inp, pristine)
        cache.copy_(cache_before)


def install(harness):
    if getattr(harness, '_arena_mla_checks_installed', False):
        return
    run = harness._run_kernel
    benchmark = harness.benchmark_cuda_graph_or_events

    def checked_run(inp):
        pristine = snapshot(inp)
        expected = harness._run_reference(pristine)
        try:
            actual = run(inp)
            unchanged(inp, pristine)
            check_output(actual, expected, pristine)
            return actual
        finally:
            restore(inp, pristine)

    harness._run_kernel = checked_run
    harness.benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(harness, benchmark, fn, **kwargs)
    harness._arena_mla_checks_installed = True
