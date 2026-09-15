"""Protected full-output, immutable-input and actual measured replay contract."""
from contextlib import contextmanager
import inspect

SYMBOL = 'decode_att_m_fwd'
KERNEL = '_fwd_kernel_stage1'
OUTPUT_KEYS = (3,)
PERTURB_KEYS = (0, 1, 2)

def expected_outputs(h, a):
    cap = a.get('logit_cap', 0.0)
    if cap == 0:
        result = h.reference_stage1(a[0], a[1], a[2], a[4], a[5], a[6], a[7], a[8])
        result[~output_write_mask(a)] = a[3][~output_write_mask(a)]
        return (result,)
    # Independent FP32 attention with cap applied to logits before softmax.
    import torch
    q, k, v, pages, lengths, splits, scale, page_size = (a[i] for i in (0, 1, 2, 4, 5, 6, 7, 8))
    result = a[3].clone()
    for b in range(q.shape[0]):
        length = int(lengths[b])
        split_size = (length + splits - 1) // splits
        for head in range(q.shape[1]):
            kv_head = head // (q.shape[1] // k.shape[1])
            for part in range(splits):
                begin, end = part * split_size, min((part + 1) * split_size, length)
                if begin >= end:
                    continue
                positions = torch.arange(begin, end, device=q.device)
                locations = pages[b, positions // page_size].long() * page_size + positions % page_size
                logits = k[locations, kv_head].float() @ q[b, head].float() * scale
                logits = cap * torch.tanh(logits / cap)
                result[b, head, part, :-1] = torch.softmax(logits, dim=0) @ v[locations, kv_head].float()
                result[b, head, part, -1] = torch.logsumexp(logits, dim=0)
    return (result,)


def output_write_mask(a):
    import torch
    splits = a[6]
    lengths = a[5]
    split_size = (lengths + splits - 1) // splits
    active = torch.arange(splits, device=lengths.device)[None, :] * split_size[:, None] < lengths[:, None]
    return active[:, None, :, None].expand_as(a[3])

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
            inactive = ~output_write_mask(self.values)
            if not torch.equal(_tensor_bytes(actual[inactive]), _tensor_bytes(self.saved[3][inactive])):
                raise AssertionError('Attention modified inactive caller-owned split slots')

    def poison(self, outputs):
        # Inactive split slots are caller-owned state: the upstream kernel
        # intentionally does not write them. Poison every required output byte.
        outputs[0].masked_fill_(output_write_mask(self.values), float('nan'))

    def perturb(self):
        for key in PERTURB_KEYS:
            self.values[key].mul_(-0.75).add_(0.3125)
        self.values[4].copy_(self.values[4].roll(1, dims=1))
        self.values[5].copy_(self.values[5].flip(0))
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

    def load():
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
    install_controls(harness)


CONTRACT_CASES = {'ragged_permuted_pages': {'contract_case': 'ragged_permuted_pages',
                           'batch': 2,
                           'query_heads': 8,
                           'kv_heads': 2,
                           'head_dim': 64,
                           'max_seq': 63,
                           'num_splits': 4,
                           'page_size': 16,
                           'sequence_lengths': [63, 37],
                           'page_mapping': 'reversed physical page order',
                           'logit_cap': 0.0,
                           'input_shapes': {'q': [2, 8, 64],
                                            'k_buffer': [128, 2, 64],
                                            'v_buffer': [128, 2, 64],
                                            'req_to_tokens': [2, 64],
                                            'b_seqlen': [2]},
                           'output_shapes': {'att_out': [2, 8, 4, 65]},
                           'dtypes': {'data': 'float16',
                                      'partial_or_statistics': 'float32',
                                      'routing': 'int32'},
                           'seed': 814},
 'ragged_permuted_capped': {'contract_case': 'ragged_permuted_capped',
                            'batch': 2,
                            'query_heads': 8,
                            'kv_heads': 2,
                            'head_dim': 64,
                            'max_seq': 63,
                            'num_splits': 4,
                            'page_size': 16,
                            'sequence_lengths': [63, 37],
                            'page_mapping': 'reversed physical page order',
                            'logit_cap': 1.5,
                            'input_shapes': {'q': [2, 8, 64],
                                             'k_buffer': [128, 2, 64],
                                             'v_buffer': [128, 2, 64],
                                             'req_to_tokens': [2, 64],
                                             'b_seqlen': [2]},
                            'output_shapes': {'att_out': [2, 8, 4, 65]},
                            'dtypes': {'data': 'float16',
                                       'partial_or_statistics': 'float32',
                                       'routing': 'int32'},
                            'seed': 814},
 'inactive_ragged_pages': {'contract_case': 'inactive_ragged_pages',
                           'batch': 2,
                           'query_heads': 8,
                           'kv_heads': 2,
                           'head_dim': 64,
                           'max_seq': 63,
                           'num_splits': 4,
                           'page_size': 16,
                           'sequence_lengths': [2, 37],
                           'page_mapping': 'reversed physical page order',
                           'logit_cap': 1.5,
                           'input_shapes': {'q': [2, 8, 64],
                                            'k_buffer': [128, 2, 64],
                                            'v_buffer': [128, 2, 64],
                                            'req_to_tokens': [2, 64],
                                            'b_seqlen': [2]},
                           'output_shapes': {'att_out': [2, 8, 4, 65]},
                           'dtypes': {'data': 'float16',
                                      'partial_or_statistics': 'float32',
                                      'routing': 'int32'},
                           'seed': 814,
                           'initial_partial_value': 23.5}}


def control_inputs(h, case, device='cuda'):
    import torch
    c = CONTRACT_CASES[case]
    q, k, v, out, pages, lengths, scale = h.make_inputs(c['batch'], c['query_heads'],
             c['kv_heads'], c['head_dim'], c['max_seq'], c['num_splits'], c['page_size'], device, torch.float16)
    torch.manual_seed(c['seed'])
    for data in (q, k, v):
        data.copy_(torch.randn_like(data))
    lengths.copy_(torch.tensor(c['sequence_lengths'], device=device, dtype=torch.int32))
    out.fill_(c.get('initial_partial_value', 0.0))
    total_pages = k.shape[0] // c['page_size']
    # The kernel indexes the table by logical page number, not token number.
    # Every entry is a valid physical page, with non-identity routing per request.
    for b in range(c['batch']):
        pages[b].copy_((total_pages - 1 - (torch.arange(pages.shape[1], device=device) + b) % total_pages).int())
    return (q, k, v, out, pages, lengths, c['num_splits'], scale, c['page_size']), {'logit_cap':c['logit_cap']}


def install_controls(harness):
    harness.CONTRACT_CASES = CONTRACT_CASES

    def correctness(case):
        with checked_modules(harness):
            module = harness.load_module()
            args, kwargs = control_inputs(harness, case)
            getattr(module, SYMBOL)(*args, **kwargs)
        return True, None

    def performance():
        rows = []
        for case in CONTRACT_CASES:
            mod = harness.load_module()
            args, kwargs = control_inputs(harness, case)
            def fn():
                getattr(mod, SYMBOL)(*args, **kwargs)
            ms, metadata = checked_benchmark(harness, harness._benchmark_cuda_graph_or_events, fn,
                        warmup=harness.WARMUP_ITERATIONS, repetition=harness.BENCHMARK_ITERATIONS)
            rows.append({'test_case_id':case, 'execution_time_ms':ms, **metadata, 'params':CONTRACT_CASES[case]})
        return rows

    harness.run_contract_correctness = correctness
    harness.run_contract_performance = performance
