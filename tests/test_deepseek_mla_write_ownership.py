"""Actual native-wrapper AST cleanup tests with full out and split scratch.

The native host wrapper executes unchanged. CUDA contexts and Triton writes are
simulated on CPU; this proves ownership/lifetime behavior, not GPU performance.
"""
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

TASK = Path(__file__).resolve().parents[1] / 'tasks/headkernel/deepseek-v4-pro__unified_paged_attention_decode'
sys.path.insert(0, str(TASK / 'ut'))
import paired_reference as pair
from native_write_ownership import track_native_allocations, owned_outputs
from snapshots import raw_storage

NATIVE_PATH = TASK / 'source/native.py'
NATIVE = ast.parse(NATIVE_PATH.read_text())
NODE = next(node for node in NATIVE.body if isinstance(node, ast.FunctionDef)
            and node.name == '_sparse_attn_v4_paged_decode_triton')
RUNNER_PATH = TASK / 'scripts/task_runner.py'
RUNNER = ast.parse(RUNNER_PATH.read_text())
RUNNER_NODES = [node for node in RUNNER.body if isinstance(node, ast.FunctionDef)
                and node.name in ('engage_specialization', 'verify_after_snapshot')]


class Metadata:
    is_cuda = True
    device = torch.device('cpu')
    def __init__(self, shape, strides, dtype=torch.bfloat16):
        self.shape, self.strides, self.dtype = shape, strides, dtype
    def stride(self, axis): return self.strides[axis]
    def new_empty(self, size, dtype): return torch.empty(size, dtype=dtype)


def native_fixture(mode, events):
    allocations, writes, scopes = [], [], []
    class Factory:
        def __getattr__(self, name): return getattr(torch, name)
        def empty(self, *args, **kwargs):
            result = torch.empty(*args, **kwargs)
            role = 'acc_partial' if result.ndim == 4 else 'm_partial'
            allocations.append((role, result)); events.append(('allocate', role))
            return result
        def empty_like(self, tensor, *args, **kwargs):
            if isinstance(tensor, Metadata): result = torch.empty(tensor.shape, dtype=tensor.dtype); role = 'out'
            else: result = torch.empty_like(tensor, *args, **kwargs); role = 'l_partial'
            allocations.append((role, result)); events.append(('allocate', role))
            return result
    class Kernel:
        def __init__(self, reduce=False): self.reduce = reduce
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                owner = tracked.allocation_tracker.active
                assert owner is not None, 'first kernel must already have an owner'
                # All four written allocations exist and are owned before the
                # first split launch, including out which reduce writes later.
                assert len(owner.tensors) == 4
                if owner not in scopes: scopes.append(owner)
                if self.reduce:
                    args[5].fill_(7); writes.append(args[5]); events.append(('write', 'out'))
                    if mode == 'first_invoke_failure' or (mode == 'capture_invoke_failure' and len(allocations) == 16):
                        raise RuntimeError('native failed after writing out and private scratch')
                else:
                    for index, role in zip((5, 6, 7), ('m_partial', 'l_partial', 'acc_partial')):
                        args[index].fill_(31 + index); writes.append(args[index]); events.append(('write', role))
            return call
    # Replay simulates kernels directly rather than invoking the Python
    # wrapper; captured allocations and the ownership set must remain stable.
    state = {'replay': False}
    factory = Factory()
    namespace = {'torch': factory, 'triton': SimpleNamespace(next_power_of_2=lambda x: 1 << (x - 1).bit_length()),
        '_FP8_DTYPE': torch.float8_e4m3fn, '_FP8_GROUP_SIZE': 64, 'LOG2E': 1.4426950408889634,
        '_kernel_config': lambda h: (16, 4, 2), '_cu_count': lambda: 304,
        '_paged_decode_split_kernel': Kernel(), '_paged_decode_reduce_kernel': Kernel(reduce=True)}
    exec(compile(ast.Module(body=[NODE], type_ignores=[]), str(NATIVE_PATH), 'exec'), namespace)
    tracked = track_native_allocations(namespace[NODE.name])
    values = {'q': Metadata((64, 16, 512), (32768, 512, 1)),
        'unified_kv': Metadata((136856, 512), (512, 1)),
        'kv_indices': torch.empty(16384, dtype=torch.int32), 'kv_indptr': torch.empty(65, dtype=torch.int32),
        'attn_sink': torch.empty(16, dtype=torch.float32), 'softmax_scale': 0.04419417382415922,
        'block_h': 16, 'kv_splits': 4, 'block_k': 16}
    def leaves(value):
        if torch.is_tensor(value): return [value]
        if isinstance(value, (list, tuple)): return sum((leaves(v) for v in value), [])
        if isinstance(value, dict): return sum((leaves(v) for v in value.values()), [])
        return []
    api = {'leaves': leaves, 'restore_storages': lambda *args: None,
        'storage_snapshots': lambda *args: {}, 'assert_immutable_inputs': lambda *args: None,
        'cpu_clone': lambda value: value.clone(), 'invoke': lambda selected, inputs: selected(**inputs),
        'runtime_abi': lambda *args: ({}, {}), 'observe_case': lambda *args: None,
        'compare': lambda *args: None, 'clear_owned': pair.clear_owned, 'owned_outputs': owned_outputs}
    return tracked, values, api, allocations, writes, scopes, state, namespace, factory


@pytest.fixture
def fake_cuda(monkeypatch):
    events = []; state = {'capturing': False, 'timing': False}; streams = []
    class Stream:
        def __init__(self): self.ident = len(streams); streams.append(self)
        def wait_stream(self, other): pass
        def synchronize(self): pass
    @contextmanager
    def stream_context(stream): yield
    @contextmanager
    def graph_context(graph, stream):
        state['capturing'] = True
        try: yield
        finally: state['capturing'] = False
    monkeypatch.setattr(torch.cuda, 'Stream', Stream)
    monkeypatch.setattr(torch.cuda, 'current_stream', Stream)
    monkeypatch.setattr(torch.cuda, 'stream', stream_context)
    monkeypatch.setattr(torch.cuda, 'graph', graph_context)
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', lambda: SimpleNamespace())
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    return events, state


def assert_all_written_scrubbed(allocations, writes):
    pointers = {value.untyped_storage().data_ptr() for value in writes}
    observed = [(name, value) for name, value in allocations if value.untyped_storage().data_ptr() in pointers]
    assert observed
    assert all(bool((raw_storage(value) == 0xAA).all()) for _, value in observed)
    return observed


def test_real_wrapper_successful_setup_clears_all_scratch_and_retains_capture_owner(fake_cuda, monkeypatch):
    events, state = fake_cuda
    function, inputs, api, allocations, writes, scopes, namespace, env, factory = native_fixture('success', events)
    graph, output, stream, owner = pair.capture_native_graph(api, {}, function, inputs, {})
    observed = assert_all_written_scrubbed(allocations, writes)
    assert len(observed) == 16  # three warmups and one capture, four storages each
    assert sum(value.untyped_storage().nbytes() for name, value in observed if name != 'out') == 4 * 8421376
    assert len(owner.tensors) == 4
    assert sum(value.untyped_storage().nbytes() for value in owner.tensors) == 9469952
    assert env['torch'].original is factory
    assert torch.empty is not env['torch'].empty  # no global torch patch
    assert hashlib.sha256(NATIVE_PATH.read_bytes()).hexdigest() == 'fd1e927877cdefcf8cabcbc6b0e04ef39ea61cd5967a4fbfd6c6e67463f2ccc5'
    pair.clear_owned(owner); owner.release()


@pytest.mark.parametrize('entry', ['capture', 'calibration', 'oracle'])
def test_real_wrapper_first_failure_scrubs_out_and_every_unreturned_partial(fake_cuda, entry):
    events, _ = fake_cuda
    function, inputs, api, allocations, writes, scopes, state, env, factory = native_fixture('first_invoke_failure', events)
    namespace = dict(api)
    exec(compile(ast.Module(body=RUNNER_NODES, type_ignores=[]), str(RUNNER_PATH), 'exec'), namespace)
    with pytest.raises(RuntimeError, match='native failed'):
        if entry == 'capture': pair.capture_native_graph(api, {}, function, inputs, {})
        elif entry == 'calibration': namespace['engage_specialization'](function, inputs, None, 0.02, 'reference', scrub_output=True)
        else: namespace['verify_after_snapshot'](torch.zeros(1), inputs, {}, function, inputs, 0.02)
    observed = assert_all_written_scrubbed(allocations, writes)
    assert len(observed) == 4
    assert sum(value.untyped_storage().nbytes() for name, value in observed if name != 'out') == 8421376
    assert function.allocation_tracker.active is None
    assert events[:4] == [('allocate', 'out'), ('allocate', 'm_partial'), ('allocate', 'l_partial'), ('allocate', 'acc_partial')]


def test_real_graph_written_buffers_are_reset_and_scrubbed_outside_timer_on_replay_failure(fake_cuda, monkeypatch):
    events, execution = fake_cuda
    function, inputs, api, allocations, writes, scopes, state, env, factory = native_fixture('success', events)
    graph, output, stream, owner = pair.capture_native_graph(api, {}, function, inputs, {})
    captured = list(owner.tensors)
    original_clear = owner.clear
    def clear():
        assert not execution['capturing'] and not execution['timing']
        events.append(('full_clear', len(owner.tensors))); original_clear()
    owner.clear = clear
    def replay():
        assert all(bool((raw_storage(value) == 0xAA).all()) for value in captured)
        for value in captured: value.fill_(57)
        raise RuntimeError('captured native replay failed after writes')
    graph.replay = replay
    def timed(call):
        execution['timing'] = True
        try: return call()
        finally: execution['timing'] = False
    monkeypatch.setattr(pair, 'device_time', timed)
    oracle = pair.PairedOracle(api, {}, {'warmup_iterations': 0, 'benchmark_iterations': 100, 'method': 'cuda_graph'},
        inputs, torch.zeros(1), inputs, output, graph, owner, None, 0.02)
    with pytest.raises(RuntimeError, match='captured native replay failed'):
        oracle.verify({})
    assert all(bool((raw_storage(value) == 0xAA).all()) for value in captured)
    assert ('full_clear', 4) in events and oracle.clears == 1
    assert len(allocations) == 16  # replay allocated no new native tensor
    owner.release()


def test_unowned_native_entry_fails_before_allocation(fake_cuda):
    events, _ = fake_cuda
    function, inputs, api, allocations, writes, scopes, state, env, factory = native_fixture('success', events)
    with pytest.raises(RuntimeError, match='scope before invoking'):
        function(**inputs)
    assert allocations == writes == []


def test_actual_capture_failure_unwinds_capture_before_scrubbing_new_allocations(fake_cuda, monkeypatch):
    from native_write_ownership import WrittenStorages
    events, state = fake_cuda
    function, inputs, api, allocations, writes, scopes, _, env, factory = native_fixture('capture_invoke_failure', events)
    original_clear = WrittenStorages.clear
    def clear(owner):
        assert not state['capturing'], 'scrub must not become part of CUDA capture'
        return original_clear(owner)
    monkeypatch.setattr(WrittenStorages, 'clear', clear)
    with pytest.raises(RuntimeError, match='native failed'):
        pair.capture_native_graph(api, {}, function, inputs, {})
    assert len(assert_all_written_scrubbed(allocations, writes)) == 16
    assert function.allocation_tracker.active is None


def test_both_legs_keep_identical_native_allocations_and_capture_setup(fake_cuda):
    events, _ = fake_cuda
    candidate = native_fixture('success', events)
    reference = native_fixture('success', events)
    cfn, cin, capi, callocations = candidate[:4]
    rfn, rin, rapi, rallocations = reference[:4]
    cgraph, cout, cstream, cowner = pair.capture_native_graph(capi, {}, cfn, cin, {})
    rgraph, rout, rstream, rowner = pair.capture_native_graph(rapi, {}, rfn, rin, {})
    def descriptors(allocations):
        return [(role, tuple(value.shape), tuple(value.stride()), str(value.dtype), value.untyped_storage().nbytes())
                for role, value in allocations]
    assert len(callocations) == len(rallocations) == 16
    assert descriptors(callocations) == descriptors(rallocations)
    pair.assert_disjoint_legs(capi, cin, cowner, rin, rowner)
    for owner in (cowner, rowner): pair.clear_owned(owner); owner.release()


def test_all_110_reference_replays_scrub_full_real_wrapper_storage(fake_cuda, monkeypatch):
    events, execution = fake_cuda
    function, inputs, api, allocations, writes, scopes, state, env, factory = native_fixture('success', events)
    graph, output, stream, owner = pair.capture_native_graph(api, {}, function, inputs, {})
    captured = tuple(owner.tensors); calls = []; measured = []
    original_clear = owner.clear
    def clear():
        assert not execution['capturing'] and not execution['timing']
        original_clear()
    owner.clear = clear
    def replay():
        assert all(bool((raw_storage(value) == 0xAA).all()) for value in captured)
        for value in captured: value.fill_(73)
        calls.append(True)
    graph.replay = replay
    def timed(call):
        execution['timing'] = True
        try: call(); measured.append(True); return 1.0
        finally: execution['timing'] = False
    monkeypatch.setattr(pair, 'device_time', timed)
    policy = {'warmup_iterations': 10, 'benchmark_iterations': 100, 'method': 'cuda_graph'}
    oracle = pair.PairedOracle(api, {}, policy, inputs, torch.zeros_like(output), inputs,
                              output, graph, owner, None, 0.02)
    for _ in range(110):
        oracle.verify({})
        assert all(bool((raw_storage(value) == 0xAA).all()) for value in captured)
    assert len(calls) == oracle.index == oracle.clears == 110
    assert len(measured) == len(oracle.reference_row()['samples_ms']) == 100
    assert len(allocations) == 16
    owner.release()
