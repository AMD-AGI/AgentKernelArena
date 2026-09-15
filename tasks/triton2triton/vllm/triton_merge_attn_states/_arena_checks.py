"""Check the in-place merge, optional LSE, and the actual timed invocation."""
from contextlib import contextmanager
import inspect

SYMBOL = "merge_attn_states"


def snapshots(inputs):
    return tuple(value.clone() for value in inputs)


def unchanged(inputs, saved):
    import torch
    for value, original in zip(inputs, saved):
        if not torch.equal(value, original):
            raise AssertionError("Merge modified a read-only attention state")


def reference(inputs):
    """Independent FP64 CPU logaddexp oracle; +inf denotes an empty FA2 state."""
    import torch
    p, pl, s, sl = inputs
    pl, sl = pl.double().cpu(), sl.double().cpu()
    pl = torch.where(torch.isposinf(pl), -torch.inf, pl)
    sl = torch.where(torch.isposinf(sl), -torch.inf, sl)
    lse = torch.logaddexp(pl, sl)
    pw = (pl - lse).exp().T.unsqueeze(-1)
    sw = (sl - lse).exp().T.unsqueeze(-1)
    result = p.double().cpu() * pw + s.double().cpu() * sw
    return result.to(device=p.device, dtype=p.dtype), lse.to(device=p.device, dtype=torch.float32)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError("Merge output shape/dtype/device violates the contract")
    if not torch.isfinite(value).all():
        raise AssertionError("Merge output must be finite for these attention states")
    torch.testing.assert_close(value, expected, atol=1e-2, rtol=1e-2)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = getattr(module, SYMBOL)
        patched.append((module, original))
        diagnosed = False

        def verify(output, inputs, output_lse=None):
            pristine = snapshots(inputs)
            expected, expected_lse = reference(pristine)
            returned = original(output, *inputs, output_lse=output_lse)
            if returned is not None:
                raise AssertionError("merge_attn_states must write its supplied buffers and return None")
            unchanged(inputs, pristine)
            check_output(output, expected)
            if output_lse is not None:
                check_output(output_lse, expected_lse)

        def checked(output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse=None):
            nonlocal diagnosed
            import torch
            verify(output, (prefix_output, prefix_lse, suffix_output, suffix_lse), output_lse)
            if not diagnosed:
                # Unscored optional-output, masked-tail and empty-partition
                # controls. Each query has at least one nonempty partition.
                p = torch.arange(102, device=output.device, dtype=torch.float32).reshape(3, 2, 17).to(output.dtype) / 17
                s = -p / 2 + 1
                pl = torch.tensor([[-torch.inf, torch.inf, 0.], [0.7, 1000., -1000.]], device=output.device)
                sl = torch.tensor([[1.1, 0., -torch.inf], [torch.inf, -1000., 1000.]], device=output.device)
                for with_lse in (False, True):
                    out = torch.full_like(p, torch.nan)
                    lse = torch.full_like(pl, torch.nan) if with_lse else None
                    verify(out, (p, pl, s, sl), lse)
                diagnosed = True

        setattr(module, SYMBOL, checked)
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            setattr(module, SYMBOL, original)


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    inputs = tuple(state[name] for name in ("prefix_output", "prefix_lse", "suffix_output", "suffix_lse"))
    output = state["output"]
    pristine = snapshots(inputs)
    expected, _ = reference(pristine)

    def measured():
        fn()
        return output

    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        inputs[0].neg_()
        inputs[2].mul_(0.5)
        inputs[1].add_(0.7)
        inputs[3].sub_(0.4)
        replay_inputs = snapshots(inputs)
        replay_expected, _ = reference(replay_inputs)
        output.fill_(float("nan"))
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_output(replayed, replay_expected)
        return ms, {**metadata, "timed_output_checked": True,
                    "perturbed_input_replay_checked": True, "source_buffers_unchanged": True}
    finally:
        for value, original in zip(inputs, pristine):
            value.copy_(original)


def install(harness):
    correctness = harness.run_correctness
    performance = harness.run_performance

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

    harness.run_correctness = checked_correctness
    harness.run_performance = checked_performance
