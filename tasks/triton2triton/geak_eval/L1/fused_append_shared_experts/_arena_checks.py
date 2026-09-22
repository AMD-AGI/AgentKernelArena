"""Validate both expert-routing outputs from the exact measured graph."""
import inspect


def check_output(outputs, topk_ids, topk_weights, cfg, reference):
    import torch

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise AssertionError('Expert append must return both IDs and weights')
    out_ids, out_weights = outputs
    if not all(isinstance(value, torch.Tensor) for value in outputs):
        raise AssertionError('Expert append outputs must be tensors')
    if not torch.isfinite(out_weights).all():
        raise AssertionError('Expert append weights contain nonfinite values')
    ref_ids, ref_weights = reference(topk_ids, topk_weights, cfg['S'], cfg['scale_factor'], cfg['N'])
    torch.testing.assert_close(out_ids, ref_ids, atol=0, rtol=0)
    torch.testing.assert_close(out_weights, ref_weights, atol=1e-6, rtol=1e-5)


def checked_benchmark(harness, benchmark, fn, **kwargs):
    from _aka_benchmark import TimedRun

    inputs = inspect.getclosurevars(fn).nonlocals
    topk_ids, topk_weights, cfg = (inputs[key] for key in ('topk_ids', 'topk_weights', 'cfg'))
    original = harness.fused_append_shared_experts
    observed = []

    def observe(*args, **call_kwargs):
        outputs = original(*args, **call_kwargs)
        observed[:] = [outputs]
        return outputs

    def measured():
        fn()
        return observed[0]

    harness.fused_append_shared_experts = observe
    try:
        timed = TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        check_output(timed.outputs, topk_ids, topk_weights, cfg, harness.reference_fused_append)
        topk_ids.add_(1).remainder_(cfg['N'])
        topk_weights.mul_(0.5).add_(0.125)
        timed.outputs[0].fill_(-1)
        timed.outputs[1].fill_(float('nan'))
        check_output(timed.rerun(), topk_ids, topk_weights, cfg, harness.reference_fused_append)
        metadata.update(timed_output_checked=True, perturbed_input_replay_checked=True)
        return ms, metadata
    finally:
        harness.fused_append_shared_experts = original
