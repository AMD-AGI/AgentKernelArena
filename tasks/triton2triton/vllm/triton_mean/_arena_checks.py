"""Check mean output contracts, optional arguments and actual measured replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'mean_dim'
TOL = 0.01

# Only allocation, same-device conversion/copy and views belong in the host
# wrapper. The reduction itself must execute the declared Triton kernel.
PREPARATION_OPS = frozenset({
    'aten::empty', 'aten::empty_like', 'aten::empty_strided',
    'aten::new_empty', 'aten::new_empty_strided',
    'aten::zeros', 'aten::zeros_like', 'aten::new_zeros',
    'aten::ones', 'aten::ones_like', 'aten::new_ones',
    'aten::full', 'aten::full_like', 'aten::new_full',
    'aten::zero_', 'aten::fill_', 'aten::copy_', 'aten::clone',
    'aten::view', 'aten::_unsafe_view', 'aten::reshape', 'aten::alias',
    'aten::detach', 'aten::as_strided', 'aten::transpose', 'aten::t',
    'aten::permute', 'aten::unsqueeze', 'aten::squeeze', 'aten::slice',
    'aten::select', 'aten::expand', 'aten::narrow', 'aten::contiguous',
    'aten::to', 'aten::_to_copy',
})


@contextmanager
def candidate_preparation_only():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class PreparationOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = func._schema.name
            if name not in PREPARATION_OPS:
                raise AssertionError(
                    f'Mean wrapper issued non-preparation PyTorch operation {name}; '
                    'operator computation must use Triton')
            if name in {'aten::_to_copy', 'aten::to'} and args:
                device = kwargs.get('device')
                if device is not None and torch.device(device).type != args[0].device.type:
                    raise AssertionError('Mean wrapper cannot move operator data to another device')
            return func(*args, **kwargs)

    with PreparationOnly():
        yield


def _triton_runtime():
    from triton import knobs
    from triton.compiler import compiler
    from triton.runtime.jit import JITFunction
    from triton.runtime.autotuner import Autotuner, Heuristics
    return knobs.runtime, compiler, JITFunction, (Autotuner, Heuristics)


def checked_candidate_call(module, fn, *args, **kwargs):
    """Audit this actual invocation, including graph capture, without GPU work.

    Triton's launcher hook is paired with the real runtime stack/object identity:
    a same-named Python class, unused JIT declaration, or compile-only warmup is
    not launch evidence. Numerical checks still establish output correctness.
    This is an execution-contract check, not a Python security sandbox.
    """
    runtime, compiler, jit_type, wrappers = _triton_runtime()
    declared = getattr(module, 'mean_kernel', None)
    seen = set()
    while isinstance(declared, wrappers) and id(declared) not in seen:
        seen.add(id(declared))
        declared = declared.fn
    if not isinstance(declared, jit_type):
        raise AssertionError('mean_kernel must be a genuine Triton JITFunction')
    run_code = jit_type.run.__code__
    launches = 0
    previous_enter, previous_exit = runtime.launch_enter_hook, runtime.launch_exit_hook

    def enter(metadata):
        if previous_enter is not None:
            previous_enter(metadata)

    def leave(metadata):
        nonlocal launches
        # Hooks are invoked by the compiled launcher while JITFunction.run or
        # CompiledKernel's cached runner is still on the stack.
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
                    launches += 1
                    break
                frame = frame.f_back
        finally:
            del frame
        if previous_exit is not None:
            previous_exit(metadata)

    runtime.launch_enter_hook, runtime.launch_exit_hook = enter, leave
    try:
        with candidate_preparation_only():
            result = fn(*args, **kwargs)
        if runtime.launch_enter_hook is not enter or runtime.launch_exit_hook is not leave:
            raise AssertionError('Mean candidate changed the runtime launch audit')
        if launches == 0:
            raise AssertionError('No genuine declared mean_kernel Triton launch in mean_dim call')
        return result
    finally:
        runtime.launch_enter_hook, runtime.launch_exit_hook = previous_enter, previous_exit


def unchanged(g, pristine):
    import torch
    if not torch.equal(g, pristine):
        raise AssertionError('Mean reduction modified its read-only input')


def reference(x, dim, keepdim=False, dtype=None):
    import torch
    if dtype is None:
        dtype = x.dtype if x.is_floating_point() else torch.float32
    # Match the wrapper's declared conversion before FP32 accumulation.
    return x.to(dtype).float().cpu().mean(dim=dim, keepdim=keepdim).to(device=x.device, dtype=dtype)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('Mean reduction output shape/dtype/device is invalid')
    if not torch.isfinite(output).all():
        raise AssertionError('Mean reduction output must be finite')
    torch.testing.assert_close(output, expected, atol=TOL, rtol=TOL)


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(x, dim, keepdim=False, dtype=None):
            import torch
            pristine = x.clone()
            expected = reference(pristine, dim, keepdim, dtype)
            diagnostic = torch.arange(30, device=x.device, dtype=x.dtype).reshape(2, 3, 5)
            saved = diagnostic.clone()
            wanted = reference(saved, -1, keepdim=True, dtype=torch.float32)
            check_output(checked_candidate_call(module, original, diagnostic, -1,
                         keepdim=True, dtype=torch.float32), wanted)
            unchanged(diagnostic, saved)
            output = checked_candidate_call(module, original, x, dim, keepdim=keepdim, dtype=dtype)
            unchanged(x, pristine)
            check_output(output, expected)
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
    module, g, dim = (c[name] for name in ('mod', 'x', 'dim'))
    pristine = g.clone()
    expected = reference(pristine, dim)
    original = getattr(module, SYMBOL)
    captured = None

    def collect(*args, **kwargs):
        nonlocal captured
        captured = checked_candidate_call(module, original, *args, **kwargs)
        return captured

    def measured():
        nonlocal captured
        captured = None
        fn()
        if captured is None:
            raise AssertionError('Benchmark did not invoke the mean candidate')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(g, pristine)
        check_output(timed.outputs, expected)
        g.mul_(-1).add_(4)
        replay_pristine = g.clone()
        replay_expected = reference(replay_pristine, dim)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(g, replay_pristine)
        check_output(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True,
                    'triton_wrapper_dispatch_checked': True}
    finally:
        setattr(module, SYMBOL, original)
        g.copy_(pristine)


def install(harness):
    correctness_original = harness.run_correctness
    performance_original = harness.run_performance
    load_original = harness.load_module

    def load():
        # Also reject eager Torch computation during candidate initialization;
        # the independent reference is evaluated after this context exits.
        with candidate_preparation_only():
            return load_original()

    harness.load_module = load

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
