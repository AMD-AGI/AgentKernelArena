"""CPU control-plane regressions; the pinned GPU pilot proves HIP semantics."""
import ast
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASKS = [ROOT / 'tasks/headkernel' / name for name in (
    'glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle',
    'glm-5.3-flash__gemm_a16w16_bf16_cijk')]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def timing():
    return load(TASKS[0] / 'ut/hip_graph_timing.py', 'glm_timing_test')


class Function:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


class HipGraph:
    """Emulate the pinned C ABI with two independent root-to-sink branches."""
    def __init__(self, fail=None, remove_edge=False):
        self.nodes = {1, 2, 3, 4}
        self.edges = {(1, 3), (2, 4)}
        self.events = {}
        self.fail = fail
        self.remove_edge = remove_edge
        for name in ['hipGraphGetNodes', 'hipGraphGetRootNodes', 'hipGraphGetEdges',
                     'hipGraphAddEventRecordNode', 'hipGraphAddDependencies']:
            setattr(self, name, Function(lambda *args, name=name: self.call(name, *args)))
        self.hipGetErrorString = Function(lambda code: b'injected HIP failure')

    @staticmethod
    def count(pointer, value):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_size_t))[0] = value

    def call(self, name, *args):
        if name == self.fail:
            return 7
        if name in ['hipGraphGetNodes', 'hipGraphGetRootNodes']:
            graph, array, count = args
            assert graph.value == 77
            nodes = sorted(self.nodes if name == 'hipGraphGetNodes' else
                           self.nodes - {target for _, target in self.edges})
            self.count(count, len(nodes))
            if array is not None:
                for index, node in enumerate(nodes):
                    array[index] = node
        elif name == 'hipGraphGetEdges':
            graph, sources, targets, count = args
            self.count(count, len(self.edges))
            if sources is not None:
                for index, (source, target) in enumerate(sorted(self.edges)):
                    sources[index], targets[index] = source, target
        elif name == 'hipGraphAddEventRecordNode':
            output, graph, dependencies, count, event = args
            node = max(self.nodes) + 1
            self.nodes.add(node)
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = node
            self.events[node] = event.value
            self.edges.update((dependencies[index], node) for index in range(count))
        elif name == 'hipGraphAddDependencies':
            graph, sources, targets, count = args
            self.edges.update((sources[index], targets[index]) for index in range(count))
            if self.remove_edge:
                self.edges.remove((1, 3))
        return 0


def use_hip(monkeypatch, timing, tmp_path, **options):
    library = tmp_path / 'libamdhip64.so'
    library.touch()
    hip = HipGraph(**options)
    monkeypatch.setattr(timing, 'Path', lambda path: library)
    monkeypatch.setattr(timing.ctypes, 'CDLL', lambda path: hip)
    return hip


def test_native_nodes_bracket_all_branches_without_changing_original_graph(timing, monkeypatch, tmp_path):
    hip = use_hip(monkeypatch, timing, tmp_path)
    owner, proof = timing._add_timing_nodes(SimpleNamespace(raw_cuda_graph=lambda: 77),
                                           SimpleNamespace(cuda_event=901), SimpleNamespace(cuda_event=902))
    assert owner is hip
    assert hip.nodes == {1, 2, 3, 4, 5, 6}
    assert hip.edges == {(1, 3), (2, 4), (5, 1), (5, 2), (3, 6), (4, 6)}
    assert hip.events == {5: 901, 6: 902}
    assert proof['original_nodes_and_edges_preserved']
    assert proof['original_root_count'] == proof['original_sink_count'] == 2
    assert proof['native_event_record_nodes'] == 2


@pytest.mark.parametrize('failure', ['hipGraphGetNodes', 'hipGraphAddEventRecordNode', 'hipGraphAddDependencies'])
def test_hip_failure_has_no_timing_fallback(timing, monkeypatch, tmp_path, failure):
    use_hip(monkeypatch, timing, tmp_path, fail=failure)
    with pytest.raises(RuntimeError, match=failure + ' returned 7'):
        timing._add_timing_nodes(SimpleNamespace(raw_cuda_graph=lambda: 77),
                                 SimpleNamespace(cuda_event=901), SimpleNamespace(cuda_event=902))


def test_graph_edge_loss_rejects(timing, monkeypatch, tmp_path):
    use_hip(monkeypatch, timing, tmp_path, remove_edge=True)
    with pytest.raises(RuntimeError, match='graph dependencies changed'):
        timing._add_timing_nodes(SimpleNamespace(raw_cuda_graph=lambda: 77),
                                 SimpleNamespace(cuda_event=901), SimpleNamespace(cuda_event=902))


class Context:
    def __init__(self, enter=lambda: None, leave=lambda: None):
        self.enter, self.leave = enter, leave

    def __enter__(self):
        self.enter()

    def __exit__(self, *args):
        self.leave()


class FakeCuda:
    def __init__(self):
        self.clock = 0.0
        self.events = []
        self.capturing = None
        self.trace = []
        self.stale = False
        self.bad_interval = False
        self.coarse_anchor = False

    def Stream(self):
        return SimpleNamespace(wait_stream=lambda other: self.trace.append('wait'),
                               synchronize=lambda: self.trace.append('stream_sync'))

    def current_stream(self):
        return self.Stream()

    def stream(self, stream):
        return Context()

    def Event(self, **kwargs):
        assert kwargs == {'enable_timing': True}
        cuda = self
        class Event:
            timestamp = 0.0

            def record(self):
                assert cuda.capturing is None, 'Python Event.record must not enter capture'
                cuda.clock += 0.1
                self.timestamp = cuda.clock

            def synchronize(self):
                cuda.trace.append('event_sync')

            def elapsed_time(self, other):
                elapsed = other.timestamp - self.timestamp
                return round(elapsed) if cuda.coarse_anchor and self is cuda.events[0] else elapsed
        event = Event()
        self.events.append(event)
        return event

    def CUDAGraph(self, *, keep_graph):
        assert keep_graph
        cuda = self
        class Graph:
            def __init__(self):
                self.calls = []
                self.instantiated = False

            def instantiate(self):
                assert hasattr(self, 'begin')
                self.instantiated = True
                cuda.trace.append('instantiate')

            def replay(self):
                assert self.instantiated
                cuda.clock += 10  # Host preparation outside the event interval.
                start = cuda.clock
                for call in self.calls:
                    call()
                    cuda.clock += 0.005
                if not cuda.stale:
                    self.begin.timestamp = start
                    self.end.timestamp = start - 0.001 if cuda.bad_interval else cuda.clock
                cuda.trace.append('replay')
        return Graph()

    def graph(self, graph):
        return Context(lambda: setattr(self, 'capturing', graph), lambda: setattr(self, 'capturing', None))


def make_timer(timing, monkeypatch, calls=1):
    cuda = FakeCuda()
    state = {'input': 0, 'output': None, 'kernel_calls': 0}
    def kernel():
        state['kernel_calls'] += 1
        state['output'] = state['input'] * 2
    def invoke():
        if cuda.capturing is None:
            kernel()
        else:
            cuda.capturing.calls.append(kernel)
    def add_nodes(graph, begin, end):
        graph.begin, graph.end = begin, end
        return object(), {'native_event_record_nodes': 2, 'original_node_count': len(graph.calls)}
    monkeypatch.setattr(timing, '_add_timing_nodes', add_nodes)
    timer = timing.capture_timed_graph(SimpleNamespace(cuda=cuda), invoke, calls)
    assert state['kernel_calls'] == 3  # Original eager capture warmups.
    return timer, cuda, state


def test_capture_reuses_events_and_retains_kernel_multiplicity(timing, monkeypatch):
    timer, cuda, state = make_timer(timing, monkeypatch, calls=4)
    for _ in range(3):
        assert timer.measure(timer.replay) == pytest.approx(0.020)
    assert state['kernel_calls'] == 3 + 3 * 4
    assert len(cuda.events) == 3  # Anchor and two materialized event owners.
    assert timer.proof()['timestamp_updates_verified'] == 3
    assert cuda.trace.index('instantiate') < cuda.trace.index('replay')


@pytest.mark.parametrize('stale,bad_interval,message', [(True, False, 'stale event'), (False, True, 'invalid graph device')])
def test_stale_or_invalid_device_timestamps_reject(timing, monkeypatch, stale, bad_interval, message):
    timer, cuda, state = make_timer(timing, monkeypatch)
    cuda.stale, cuda.bad_interval = stale, bad_interval
    with pytest.raises(RuntimeError, match=message):
        timer.measure(timer.replay)


def test_short_interval_uses_direct_events_not_rounded_anchor_difference(timing, monkeypatch):
    timer, cuda, state = make_timer(timing, monkeypatch)
    cuda.coarse_anchor = True
    assert timer.measure(timer.replay) == pytest.approx(0.005)
    assert timer._previous[0] == timer._previous[1]


@pytest.mark.parametrize('count', [0, 2])
def test_timer_requires_exactly_one_replay(timing, monkeypatch, count):
    timer, cuda, state = make_timer(timing, monkeypatch)
    with pytest.raises(RuntimeError, match='exactly one graph replay'):
        timer.measure(lambda: [timer.replay() for _ in range(count)])


def test_canonical_loop_keeps_all_resets_oracles_warmups_and_samples(timing, monkeypatch):
    timer, cuda, state = make_timer(timing, monkeypatch)
    contract = load(TASKS[0] / 'ut/evaluation_contract.py', 'glm_timing_contract')
    observed = {'reset': 0, 'initialize': 0, 'verify': 0}
    case = {'case_id': 'cpu-timer-contract', 'calls_per_sample': 1}
    def reset(seed):
        observed['reset'] += 1
        state['input'] = seed
        return seed * 2
    def initialize():
        observed['initialize'] += 1
        state['output'] = None
    def verify(expected):
        assert state['output'] == expected
        observed['verify'] += 1
    row = contract.checked_replays(case, {'warmup_iterations': 10, 'benchmark_iterations': 100, 'method': 'cuda_graph'},
                                   reset_inputs=reset, initialize_outputs=initialize, replay=timer.replay,
                                   measure=timer.measure, verify=verify, observe=lambda: case, seed=700)
    assert observed == {'reset': 110, 'initialize': 110, 'verify': 110}
    assert len(row['samples_ms']) == 100
    assert state['kernel_calls'] == 113
    assert timer.proof()['timestamp_updates_verified'] == 110


def test_task_local_copies_are_identical_and_all_four_paths_use_them():
    assert (TASKS[0] / 'ut/hip_graph_timing.py').read_bytes() == (TASKS[1] / 'ut/hip_graph_timing.py').read_bytes()
    for task in TASKS:
        for filename in ['task_runner.py', 'production_comparison.py']:
            tree = ast.parse((task / 'scripts' / filename).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            assert sum(isinstance(node.func, ast.Name) and node.func.id == 'capture_timed_graph' for node in calls) == 1
            assert not any(isinstance(node.func, ast.Attribute) and node.func.attr == 'Event' for node in calls)
            checked = next(node for node in calls if getattr(node.func, 'id', getattr(node.func, 'attr', '')) == 'checked_replays')
            callbacks = {kw.arg: ast.unparse(kw.value) for kw in checked.keywords}
            assert callbacks['replay'] == 'timing.replay' and callbacks['measure'] == 'timing.measure'
