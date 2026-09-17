"""Apply the original MXFP4 comparison to the actual timed/replayed outputs."""
from contextlib import contextmanager
import inspect


def snapshots(inputs):
    return {key: inputs[key].clone() for key in ('x', 'sorted_ids', 'num_valid_ids')}


def unchanged(inputs, pristine):
    import torch
    for key, saved in pristine.items():
        if not torch.equal(inputs[key], saved):
            raise AssertionError(f'Quant-sort modified read-only input {key}')


def reference(harness, inputs):
    packed, sorted_scales, scales = harness.run_fused_dynamic_mxfp4_quant_moe_sort_ref(
        inputs['x'], inputs['sorted_ids'], inputs['token_num'], inputs['topk'],
        harness._fp4x2, None, inputs['num_valid_ids'], inputs['block_size_M'])
    # Preserve the original round-trip convention, including native scales.
    _, native_scales = harness.dynamic_mxfp4_quant(inputs['x'])
    native_scales = native_scales[:scales.shape[0], :scales.shape[1]]
    return packed, sorted_scales, scales, native_scales


def check_output(harness, outputs, expected, valid_rows):
    import torch
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise AssertionError('Quant-sort must return both packed data and sorted scales')
    packed, sorted_scales = outputs
    ref_packed, ref_sorted, ref_scales, native_scales = expected
    for actual, ref, dtype in ((packed, ref_packed, harness._fp4x2),
                               (sorted_scales, ref_sorted, harness._fp8_e8m0)):
        if not isinstance(actual, torch.Tensor) or (actual.shape != ref.shape or
                actual.dtype != dtype or actual.device != ref.device):
            raise AssertionError('Quant-sort output shape/dtype/device is invalid')
    # Keep the original valid-row slice: padding outside it is not specified.
    torch.testing.assert_close(ref_sorted[:valid_rows].view(torch.uint8),
                               sorted_scales[:valid_rows].view(torch.uint8), atol=0.1, rtol=0.1)
    ref_float = harness.convert_mxfp4_to_fp32(ref_packed.view(torch.uint8),
                                           ref_scales.view(torch.uint8))
    actual_float = harness.convert_mxfp4_to_fp32(packed.view(torch.uint8),
                                              native_scales.view(torch.uint8))
    torch.testing.assert_close(ref_float, actual_float, atol=0.1, rtol=0.1)


@contextmanager
def checked_correctness(harness):
    original = harness.fused_dynamic_mxfp4_quant_moe_sort

    def checked(x, sorted_ids, num_valid_ids, token_num, topk, block_size=32, **kwargs):
        inputs = dict(x=x, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
                      token_num=token_num, topk=topk, block_size_M=block_size)
        pristine = snapshots(inputs)
        expected = reference(harness, {**inputs, **pristine})
        valid_rows = int(pristine['num_valid_ids'][0].item())
        outputs = original(x, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
                           token_num=token_num, topk=topk, block_size=block_size, **kwargs)
        unchanged(inputs, pristine)
        check_output(harness, outputs, expected, valid_rows)
        return outputs

    harness.fused_dynamic_mxfp4_quant_moe_sort = checked
    try:
        yield
    finally:
        harness.fused_dynamic_mxfp4_quant_moe_sort = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    import torch
    from _aka_benchmark import TimedRun
    inputs = inspect.getclosurevars(fn).nonlocals['inp']
    pristine = snapshots(inputs)
    expected = reference(harness, {**inputs, **pristine})
    valid_rows = int(pristine['num_valid_ids'][0].item())
    original = harness.fused_dynamic_mxfp4_quant_moe_sort
    captured = None

    def collect(*args, **kw):
        nonlocal captured
        captured = original(*args, **kw)
        return captured

    def measured():
        nonlocal captured
        captured = None
        fn()  # Keep the original full wrapper invocation/allocation boundary.
        if captured is None:
            raise AssertionError('Benchmark did not invoke the declared quant-sort candidate')
        return captured

    harness.fused_dynamic_mxfp4_quant_moe_sort = collect
    try:
        timed = TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        unchanged(inputs, pristine)
        check_output(harness, timed.outputs, expected, valid_rows)
        # No new scored cases or measurements. Change data and legal row mapping
        # in the buffers referenced by the already captured GPU graph.
        inputs['x'].mul_(-4)
        ids = inputs['sorted_ids']
        token = ((ids & 0xffffff) + 1).remainder(inputs['token_num'])
        expert = ((ids >> 24) + 1).remainder(inputs['topk'])
        ids.copy_((expert << 24) | token)
        replay_pristine = snapshots(inputs)
        replay_expected = reference(harness, {**inputs, **replay_pristine})
        for output in timed.outputs:
            output.view(torch.uint8).fill_(255)
        replayed = timed.rerun()
        unchanged(inputs, replay_pristine)
        check_output(harness, replayed, replay_expected, valid_rows)
        metadata.update(timed_output_checked=True, perturbed_input_replay_checked=True,
                        source_buffers_unchanged=True)
        return ms, metadata
    finally:
        harness.fused_dynamic_mxfp4_quant_moe_sort = original
        for key, saved in pristine.items():
            inputs[key].copy_(saved)
