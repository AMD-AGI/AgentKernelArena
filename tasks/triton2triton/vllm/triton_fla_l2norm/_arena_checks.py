"""Check L2 output contracts, epsilon/tail diagnostics and actual measured replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'l2norm_fwd'
TOL = 0.0001
MAIN_KERNEL_WIDTHS = (255, 256, 257, 512, 513, 1024, 1025, 2048, 2049,
                      4096, 4097, 8192, 8193, 16384)
WIDE_KERNEL_WIDTHS = (16385, 32768, 32769)


def unchanged(g, pristine):
    import torch
    if not torch.equal(g, pristine):
        raise AssertionError('L2 normalization modified its read-only input')


def reference(harness, x, eps):
    return harness.reference(x.cpu(), eps).to(device=x.device, dtype=x.dtype)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('L2 normalization output shape/dtype/device is invalid')
    if not torch.isfinite(output).all():
        raise AssertionError('L2 normalization output must be finite')
    torch.testing.assert_close(output, expected, atol=TOL, rtol=TOL)


def diagnostic_input(x):
    # Rank/row/feature tails plus a zero and a near-zero row make epsilon
    # observable. These extra checks do not replace the five scored inputs.
    result = x[:6, :min(17, x.shape[-1])].contiguous().clone()
    result[0].zero_()
    result[1].mul_(1e-4)
    return result.reshape(2, 3, -1)


def independent_reference(x, eps):
    """Use a separate CPU norm operator for added width diagnostics."""
    import torch
    cpu = x.float().cpu()
    length = torch.linalg.vector_norm(cpu, dim=-1, keepdim=True)
    return (cpu / torch.sqrt(length.square() + eps)).to(device=x.device, dtype=x.dtype)


def main_width_input(x, width):
    import torch
    features = torch.linspace(-0.75, 0.5, steps=width, device=x.device, dtype=x.dtype)
    rows = torch.stack((features, features.flip(0) * 1e-4,
                        torch.zeros_like(features)))
    rows[2, 0] = 0.75
    rows[2, -1] = -0.25
    return rows


def _triton_runtime():
    from triton import knobs
    from triton.compiler import compiler
    from triton.runtime.jit import JITFunction
    from triton.runtime.autotuner import Autotuner, Heuristics
    return knobs.runtime, compiler, JITFunction, (Autotuner, Heuristics)


def _declared_kernel(module, jit_type, wrappers):
    outer = getattr(module, 'l2norm_fwd_kernel', None)
    current = outer
    chain = []
    seen = set()
    while isinstance(current, wrappers) and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.fn
    if not isinstance(current, jit_type):
        raise AssertionError('l2norm_fwd_kernel must be a genuine Triton JITFunction')
    chain.append(current)
    return current, chain


@contextmanager
def named_kernel_dispatches(module, *, suppress=False):
    """Observe compiled launches of the declared JIT, not compile-only run calls."""
    runtime, compiler, jit_type, wrappers = _triton_runtime()
    declared, chain = _declared_kernel(module, jit_type, wrappers)
    calls = [0]
    if suppress:
        patched = []
        try:
            for obj in chain:
                original = getattr(obj, 'run', None)
                if not callable(original):
                    raise AssertionError('Declared L2 Triton kernel has no run method')
                own = vars(obj).get('run') if hasattr(obj, '__dict__') else None
                had_own = hasattr(obj, '__dict__') and 'run' in vars(obj)

                def skip(*args, **kwargs):
                    calls[0] += 1
                    # A skipped launch must not pass by reusing an output.
                    import torch
                    if len(args) > 1 and isinstance(args[1], torch.Tensor):
                        args[1].fill_(float('nan'))
                    return None

                setattr(obj, 'run', skip)
                patched.append((obj, had_own, own))
            yield calls
        finally:
            for obj, had_own, own in reversed(patched):
                if had_own:
                    setattr(obj, 'run', own)
                else:
                    delattr(obj, 'run')
        return

    run_code = jit_type.run.__code__
    previous_enter, previous_exit = runtime.launch_enter_hook, runtime.launch_exit_hook

    def enter(metadata):
        if previous_enter is not None:
            previous_enter(metadata)

    def leave(metadata):
        frame = inspect.currentframe().f_back
        try:
            while frame is not None:
                compiled = None
                if frame.f_code is run_code and not frame.f_locals.get('warmup', True):
                    compiled = frame.f_locals.get('kernel')
                elif frame.f_globals is vars(compiler):
                    compiled = frame.f_locals.get('self')
                if (isinstance(compiled, compiler.CompiledKernel) and
                        getattr(compiled.src, 'fn', None) is declared):
                    calls[0] += 1
                    break
                frame = frame.f_back
        finally:
            del frame
        if previous_exit is not None:
            previous_exit(metadata)

    runtime.launch_enter_hook, runtime.launch_exit_hook = enter, leave
    try:
        yield calls
        if runtime.launch_enter_hook is not enter or runtime.launch_exit_hook is not leave:
            raise AssertionError('L2 candidate changed the Triton launch audit')
    finally:
        runtime.launch_enter_hook, runtime.launch_exit_hook = previous_enter, previous_exit


def require_kernel_contribution(module, wrapper, x, eps, expected):
    """A correct alternate implementation must not hide an inert named kernel."""
    probe = x.clone()
    with named_kernel_dispatches(module, suppress=True) as calls:
        try:
            output = wrapper(probe, eps=eps)
        except Exception:
            output = None
    if not calls[0]:
        raise AssertionError('L2 wrapper never launched the declared kernel')
    unchanged(probe, x)
    try:
        check_output(output, expected)
    except (AssertionError, TypeError):
        return
    raise AssertionError('L2 output is independent of the declared kernel')


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))
        diagnosed_geometry = False

        def checked(x, eps=1e-6):
            nonlocal diagnosed_geometry
            import torch
            pristine = x.clone()
            expected = reference(harness, pristine, eps)
            diagnostic = diagnostic_input(pristine)
            saved = diagnostic.clone()
            diagnostic_eps = 1e-3
            diagnostic_expected = reference(harness, saved, diagnostic_eps)
            with named_kernel_dispatches(module) as calls:
                diagnostic_output = original(diagnostic, eps=diagnostic_eps)
            if not calls[0]:
                raise AssertionError('L2 diagnostic bypassed the declared kernel')
            check_output(diagnostic_output, diagnostic_expected)
            unchanged(diagnostic, saved)
            tail = pristine[:7, :65].contiguous().clone()
            tail[0].zero_()
            tail[1].mul_(1e-3)
            tail_saved = tail.clone()
            tail_eps = 2e-4
            with named_kernel_dispatches(module) as calls:
                tail_output = original(tail, eps=tail_eps)
            if not calls[0]:
                raise AssertionError('L2 tail bypassed the declared kernel')
            check_output(tail_output, reference(harness, tail_saved, tail_eps))
            unchanged(tail, tail_saved)
            if not diagnosed_geometry:
                # Cover the main JIT's BD=256 and larger power-of-two dispatches,
                # with widths on both sides of feature-tail boundaries. These
                # correctness-only probes leave the five scored inputs intact.
                for width in MAIN_KERNEL_WIDTHS:
                    main = main_width_input(x, width)
                    main_saved = main.clone()
                    main_eps = 3e-4
                    main_expected = independent_reference(main_saved, main_eps)
                    with named_kernel_dispatches(module) as calls:
                        main_output = original(main, eps=main_eps)
                    if not calls[0]:
                        raise AssertionError('L2 main-width control bypassed the declared kernel')
                    check_output(main_output, main_expected)
                    unchanged(main, main_saved)
                    require_kernel_contribution(module, original, main_saved,
                                                main_eps, main_expected)
                # The FP32 single-block cap is 16,384 features. Cross the cap,
                # then check a complete second tile and third/tail tile.
                for width in WIDE_KERNEL_WIDTHS:
                    wide = torch.linspace(-0.5, 0.75, steps=width, device=x.device,
                                          dtype=x.dtype).repeat(2, 1)
                    wide[0].zero_()
                    wide_saved = wide.clone()
                    wide_eps = 4e-4
                    check_output(original(wide, eps=wide_eps),
                                 reference(harness, wide_saved, wide_eps))
                    unchanged(wide, wide_saved)
                diagnosed_geometry = True
            with named_kernel_dispatches(module) as calls:
                output = original(x, eps=eps)
            if not calls[0]:
                raise AssertionError('Scored L2 wrapper bypassed the declared kernel')
            unchanged(x, pristine)
            check_output(output, expected)
            require_kernel_contribution(module, original, pristine, eps, expected)
            return output

        setattr(module, SYMBOL, checked)
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = load_original
        for module, original in reversed(patched):
            setattr(module, SYMBOL, original)


def checked_benchmark(harness, benchmark, fn, **options):
    c = inspect.getclosurevars(fn).nonlocals
    module, args, kwargs = (c[name] for name in ('mod', 'args', 'kwargs'))
    g, = args
    eps = kwargs.get('eps', 1e-6)
    pristine = g.clone()
    expected = reference(harness, pristine, eps)
    original = getattr(module, SYMBOL)
    captured = None
    named_calls_last = [0]

    def collect(*args, **kwargs):
        nonlocal captured
        captured = original(*args, **kwargs)
        return captured

    def measured():
        nonlocal captured
        captured = None
        before = named_calls[0]
        fn()
        named_calls_last[0] = named_calls[0] - before
        if captured is None:
            raise AssertionError('Benchmark did not invoke the L2 candidate')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        with named_kernel_dispatches(module) as named_calls:
            timed = harness._TimedRun()
            checked_samples = [0]
            def check_sample(output):
                if not named_calls_last[0]:
                    raise AssertionError('Measured L2 wrapper bypassed the declared kernel')
                check_output(output, expected)
                checked_samples[0] += 1
            timed.after_sample = check_sample
            # Each reported sample must execute and validate the full wrapper.
            options = {**options, 'use_cuda_graph': False,
                       'fallback_reason': 'validate_each_public_invocation'}
            ms, metadata = benchmark(measured, timed_run=timed, **options)
            if not timed.bound or checked_samples[0] != options['repetition']:
                raise AssertionError('Reported sample outputs were not all checked')
            unchanged(g, pristine)
            check_output(timed.outputs, expected)
            g.mul_(-1).add_(0.25)
            replay_pristine = g.clone()
            replay_expected = reference(harness, replay_pristine, eps)
            timed.outputs.fill_(float('nan'))
            replayed = timed.rerun()
            if not named_calls_last[0]:
                raise AssertionError('Perturbed L2 replay bypassed the declared kernel')
            unchanged(g, replay_pristine)
            check_output(replayed, replay_expected)
            return ms, {**metadata, 'timed_output_checked': True,
                        'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True,
                        'measured_samples_checked': checked_samples[0],
                        'named_kernel_dispatches_checked': checked_samples[0], }
    finally:
        setattr(module, SYMBOL, original)
        g.copy_(pristine)


def install(harness):
    correctness_original = harness.run_correctness
    performance_original = harness.run_performance

    def correctness(*args, **kwargs):
        with checked_modules(harness):
            return correctness_original(*args, **kwargs)

    def performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(harness, benchmark, fn, **kwargs)
        try:
            return performance_original()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_correctness = correctness
    harness.run_performance = performance
