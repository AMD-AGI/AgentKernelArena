"""Protected full-output, immutable-input and actual measured replay contract."""
from contextlib import contextmanager
import inspect

SYMBOL = 'unified_attention_2d'
KERNEL = 'kernel_unified_attention_2d'
OUTPUT_KEYS = (3,)
PERTURB_KEYS = (0, 1, 2)

def expected_outputs(h, a):
    return (h.reference_attention(a[0], a[1], a[2], a[4], a[5], a[6], a[7], a[1].shape[1],
                                  sliding_window=a.get('sliding_window', 0), softcap=a.get('softcap', 0.0)),)

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


def _device_identity(device):
    import torch
    device = torch.device(device)
    index = device.index
    if device.type == 'cuda' and index is None:
        # An unindexed CUDA target means the current device, which need not
        # be the source tensor's device. ROCm also uses this CUDA interface.
        index = torch.cuda.current_device()
    elif device.type == 'cpu':
        # PyTorch accepts cpu:0, but CPU tensors have an unindexed device.
        index = None
    return device.type, index


@contextmanager
def candidate_preparation_only():
    from torch.utils._python_dispatch import TorchDispatchMode

    class PreparationOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = func._schema.name
            if name not in PREPARATION_OPS:
                raise AssertionError(
                    f'Attention wrapper issued non-preparation PyTorch operation {name}; '
                    'operator computation must use Triton')
            if name in {'aten::_to_copy', 'aten::to'} and args:
                device = kwargs.get('device')
                if device is not None and _device_identity(device) != _device_identity(args[0].device):
                    raise AssertionError('Attention wrapper cannot move operator data to another device')
            if name == 'aten::copy_' and len(args) >= 2:
                if _device_identity(args[0].device) != _device_identity(args[1].device):
                    raise AssertionError('Attention wrapper cannot move operator data to another device')
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
    declared = getattr(module, KERNEL, None)
    seen = set()
    while isinstance(declared, wrappers) and id(declared) not in seen:
        seen.add(id(declared))
        declared = declared.fn
    if not isinstance(declared, jit_type):
        raise AssertionError(f'{KERNEL} must be a genuine Triton JITFunction')
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
            raise AssertionError('Attention candidate changed the runtime launch audit')
        if launches == 0:
            raise AssertionError(f'No genuine declared {KERNEL} Triton launch in {SYMBOL} call')
        return result
    finally:
        runtime.launch_enter_hook, runtime.launch_exit_hook = previous_enter, previous_exit



def _tensor_bytes(value):
    # Byte equality also covers shape-only uninitialized tensors containing NaN.
    import torch
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


class CallPlan:
    def __init__(self, harness, args, kwargs):
        import torch
        self.harness = harness
        self.values = dict(enumerate(args)) | kwargs
        self.saved = {k: v.clone() for k, v in self.values.items() if isinstance(v, torch.Tensor)}
        self.metadata = {k: (v.shape, v.stride(), v.dtype, v.device, v.data_ptr())
                         for k, v in self.values.items() if isinstance(v, torch.Tensor)}
        self.expected = expected_outputs(harness, self.values | self.saved)

    def outputs(self, result):
        if OUTPUT_KEYS:
            return tuple(self.values[k] for k in OUTPUT_KEYS)
        if not isinstance(result, (tuple, list)) or len(result) != len(self.expected):
            raise AssertionError('Attention must return every declared output')
        return tuple(result)

    def unchanged(self):
        import torch
        for key, original in self.saved.items():
            value = self.values[key]
            if (value.shape, value.stride(), value.dtype, value.device, value.data_ptr()) != self.metadata[key]:
                raise AssertionError('Attention changed caller tensor metadata/storage')
            if key not in OUTPUT_KEYS and not torch.equal(_tensor_bytes(value), _tensor_bytes(original)):
                raise AssertionError('Attention modified a read-only input')

    def check(self, outputs, expected=None):
        import torch
        expected = self.expected if expected is None else expected
        if not isinstance(outputs, (tuple, list)) or len(outputs) != len(expected):
            raise AssertionError('Attention omitted a declared output')
        for index, (actual, wanted) in enumerate(zip(outputs, expected)):
            if not isinstance(actual, torch.Tensor) or (actual.shape != wanted.shape or
                    actual.dtype != wanted.dtype or actual.device != wanted.device):
                raise AssertionError(f'Attention output {index} shape/dtype/device mismatch')
            for key in self.saved:
                if key not in OUTPUT_KEYS and torch._C._overlaps(actual, self.values[key]):
                    raise AssertionError('Attention output aliases a read-only input')
            for previous in outputs[:index]:
                if torch._C._overlaps(actual, previous):
                    raise AssertionError('Distinct attention outputs share storage')
            # assert_close rejects NaN and requires matching signed infinities;
            # ordinary finite cases therefore reject unwritten poison everywhere.
            torch.testing.assert_close(actual, wanted, atol=0.01, rtol=0.01, equal_nan=False)

    def poison(self, outputs):
        for output in outputs:
            output.fill_(float('nan'))

    def perturb(self):
        for key in PERTURB_KEYS:
            self.values[key].mul_(-0.75).add_(0.3125)
        # New source snapshot is the reference input and read-only replay guard.
        self.saved = {k: self.values[k].clone() for k in self.saved}
        return expected_outputs(self.harness, self.values | self.saved)

    def restore(self, originals):
        for key, value in originals.items():
            self.values[key].copy_(value)


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(*args, **kwargs):
            plan = CallPlan(harness, args, kwargs)
            if OUTPUT_KEYS:
                plan.poison(plan.outputs(None))
            result = checked_candidate_call(module, original, *args, **kwargs)
            plan.unchanged()
            plan.check(plan.outputs(result))
            return result

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
    module = inspect.getclosurevars(fn).nonlocals['mod']
    original = getattr(module, SYMBOL)
    plans = []
    captured = None

    def describe(*args, **kwargs):
        plans.append(CallPlan(harness, args, kwargs))

    # Only collect the protected invocation's arguments; no candidate launch.
    setattr(module, SYMBOL, describe)
    try:
        fn()
    finally:
        setattr(module, SYMBOL, original)
    if len(plans) != 1:
        raise AssertionError('Benchmark must invoke exactly one public attention wrapper')
    plan = plans[0]
    originals = dict(plan.saved)

    def collect(*args, **kwargs):
        nonlocal captured
        result = checked_candidate_call(module, original, *args, **kwargs)
        captured = plan.outputs(result)
        return result

    def measured():
        nonlocal captured
        captured = None
        fn()
        if captured is None:
            raise AssertionError('Benchmark omitted the attention wrapper')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        plan.unchanged()
        plan.check(timed.outputs)
        replay_expected = plan.perturb()
        plan.poison(timed.outputs)
        replayed = timed.rerun()
        plan.unchanged()
        plan.check(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True,
                    'triton_wrapper_dispatch_checked': True}
    finally:
        setattr(module, SYMBOL, original)
        plan.restore(originals)


def install(harness):
    correctness_original = harness.run_correctness
    performance_original = harness.run_performance
    load_original = harness.load_module

    loaded = None

    def load():
        nonlocal loaded
        # Keep one candidate module (and its compiled Triton kernels) alive
        # throughout this action. Collecting a prior case's module during a
        # later graph capture may call HIP unload_module, invalidating capture.
        if loaded is None:
            with candidate_preparation_only():
                loaded = load_original()
        return loaded

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
    install_controls(harness)


CONTRACT_CASES = {'ragged_permuted_window_softcap': {'batch': 2,
                                    'query_heads': 8,
                                    'kv_heads': 2,
                                    'head_dim': 64,
                                    'seed': 927,
                                    'dtypes': {'data': 'float16',
                                               'statistics': 'float32',
                                               'routing': 'int32'},
                                    'contract_case': 'ragged_permuted_window_softcap',
                                    'sequence_lengths': [19, 7],
                                    'query_lengths': [3, 1],
                                    'block_size': 16,
                                    'sliding_window': 12,
                                    'softcap': 2.0,
                                    'input_shapes': {'q': [4, 8, 64],
                                                     'key_cache': [8, 16, 2, 64],
                                                     'value_cache': [8, 16, 2, 64],
                                                     'block_table': [2, 2],
                                                     'cu_seqlens_q': [3],
                                                     'seqused_k': [2]},
                                    'output_shapes': {'output': [4, 8, 64]},
                                    'page_mapping': 'reversed physical page order'}}


def control_inputs(h,case,device="cuda"):
    import torch
    c=CONTRACT_CASES[case]
    torch.manual_seed(c["seed"])
    generated=h.make_test_data(2,2,19,8,2,64,16,device,torch.float16)
    q,kc,vc=generated[:3]
    out,pages,starts,lengths,scale=generated[3:]
    pages.copy_((kc.shape[0]-1-torch.arange(pages.numel(),device=device).reshape_as(pages)).int())
    starts.copy_(torch.tensor([0,3,4],device=device,dtype=torch.int32))
    lengths.copy_(torch.tensor(c['sequence_lengths'],device=device,dtype=torch.int32))
    kwargs={'sliding_window':c['sliding_window'],'softcap':c['softcap']}
    return (q,kc,vc,out,pages,starts,lengths,scale),kwargs


def install_controls(harness):
    harness.CONTRACT_CASES = CONTRACT_CASES

    def correctness(case):
        with checked_modules(harness):
            module = harness.load_module()
            args, kwargs = control_inputs(harness, case)
            getattr(module, SYMBOL)(*args, **kwargs)
        # These branch controls are correctness-only. Observe their actual
        # captured replay without adding a row to the official score domain.
        mod = harness.load_module()
        def fn():
            getattr(mod, SYMBOL)(*args, **kwargs)
        ms, metadata = checked_benchmark(harness, harness._benchmark_cuda_graph_or_events, fn,
                    warmup=harness.WARMUP_ITERATIONS, repetition=harness.BENCHMARK_ITERATIONS)
        return True, {'unscored_control': True, 'unscored_diagnostic_time_ms': ms, **metadata}

    harness.run_contract_correctness = correctness
