"""CPU checks for HIP graph topology and per-replay timing evidence."""

import importlib.util
import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest


TASK = Path(__file__).resolve().parents[1] / "tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8"
SPEC = importlib.util.spec_from_file_location("quant_graph_timer_test", TASK / "ut/graph_timer.py")
TIMER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIMER)


class FakeHip:
    """Exercise the ctypes argument layout without loading a GPU runtime."""
    def __init__(self, nodes, edges):
        self.nodes = list(nodes)
        self.edges = set(edges)
        self.added_events = {}

    def fill_nodes(self, values, array, count):
        count._obj.value = len(values)
        if array is not None:
            for index, value in enumerate(values):
                array[index] = value
        return 0

    def hipGraphGetNodes(self, raw, array, count):
        assert raw.value == 123
        return self.fill_nodes(self.nodes, array, count)

    def hipGraphGetRootNodes(self, raw, array, count):
        roots = sorted(set(self.nodes) - {target for _, target in self.edges})
        return self.fill_nodes(roots, array, count)

    def hipGraphGetEdges(self, raw, sources, targets, count):
        count._obj.value = len(self.edges)
        if sources is not None:
            for index, (source, target) in enumerate(sorted(self.edges)):
                sources[index], targets[index] = source, target
        return 0

    def hipGraphAddEventRecordNode(self, node, raw, dependencies, count, event):
        new = max(self.nodes) + 1
        node._obj.value = new
        self.nodes.append(new)
        self.added_events[event.value] = new
        if dependencies is not None:
            self.edges.update((dependencies[i], new) for i in range(count))
        return 0

    def hipGraphAddDependencies(self, raw, sources, targets, count):
        self.edges.update((sources[i], targets[i]) for i in range(count))
        return 0

    def hipGetErrorString(self, status):
        return b"synthetic API error"


def graph_and_events():
    return SimpleNamespace(raw_cuda_graph=lambda: 123), SimpleNamespace(cuda_event=501), SimpleNamespace(cuda_event=502)


@pytest.mark.parametrize("nodes,edges", [([10], []), ([10, 20, 30, 40], [(10, 30), (20, 40)])])
def test_native_nodes_bracket_all_branches_and_preserve_captured_work(monkeypatch, nodes, edges):
    hip = FakeHip(nodes, edges)
    monkeypatch.setattr(TIMER, "hip_library", lambda: hip)
    graph, begin, end = graph_and_events()
    runtime, proof = TIMER.add_timing_nodes(graph, begin, end)
    assert runtime is hip
    start, stop = hip.added_events[501], hip.added_events[502]
    roots = set(nodes) - {target for _, target in edges}
    sinks = set(nodes) - {source for source, _ in edges}
    assert hip.edges == set(edges) | {(start, root) for root in roots} | {(sink, stop) for sink in sinks}
    assert set(hip.nodes) == set(nodes) | {start, stop}
    assert proof["final_node_count"] == len(nodes) + 2
    assert proof["final_edge_count"] == len(edges) + len(roots) + len(sinks)


def test_changed_original_dependencies_are_rejected(monkeypatch):
    hip = FakeHip([10, 20, 30, 40], [(10, 30), (20, 40)])
    original = hip.hipGraphAddDependencies

    def corrupt(*args):
        code = original(*args)
        hip.edges.remove((10, 30))
        return code

    hip.hipGraphAddDependencies = corrupt
    monkeypatch.setattr(TIMER, "hip_library", lambda: hip)
    with pytest.raises(ValueError, match="captured work dependencies"):
        TIMER.add_timing_nodes(*graph_and_events())


@pytest.mark.parametrize("error", ["empty_graph", "null_graph", "uninitialized_event", "same_event", "api_failure"])
def test_missing_graph_or_event_support_fails_without_fallback(monkeypatch, error):
    hip = FakeHip([10], [])
    graph, begin, end = graph_and_events()
    if error == "empty_graph":
        hip.nodes = []
    elif error == "null_graph":
        graph.raw_cuda_graph = lambda: 0
    elif error == "uninitialized_event":
        begin.cuda_event = 0
    elif error == "same_event":
        end.cuda_event = begin.cuda_event
    else:
        hip.hipGraphAddEventRecordNode = lambda *args: 99
    monkeypatch.setattr(TIMER, "hip_library", lambda: hip)
    with pytest.raises((ValueError, RuntimeError)):
        TIMER.add_timing_nodes(graph, begin, end)


class Clock:
    def __init__(self):
        self.now, self.records, self.syncs = 0.0, 0, 0
        self.events = []

    def event(self, *, enable_timing):
        assert enable_timing
        clock = self

        class Event:
            def record(self):
                clock.now += 1
                clock.records += 1
                self.stamp = clock.now

            def synchronize(self):
                clock.syncs += 1

            def elapsed_time(self, other):
                return other.stamp - self.stamp

        event = Event()
        self.events.append(event)
        return event

    def synchronize(self):
        self.syncs += 1


def topology():
    return {"api": "hipGraphAddEventRecordNode", "original_node_count": 1, "original_edge_count": 0,
            "original_root_count": 1, "original_sink_count": 1, "final_node_count": 3, "final_edge_count": 2,
            "original_nodes_and_edges_preserved": True, "all_original_roots_depend_on_begin": True,
            "end_depends_on_all_original_sinks": True}


def test_persistent_events_advance_for_every_sample_and_instantiate_after_insertion(monkeypatch):
    clock, operations = Clock(), []
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=clock.event, synchronize=clock.synchronize))
    timer = TIMER.NativeGraphTimer(torch)

    def replay():
        clock.now += 10
        timer.begin.stamp, timer.end.stamp = clock.now, clock.now + 0.25

    def attach(*args):
        operations.append("insert")
        return "runtime_owner", topology()

    monkeypatch.setattr(TIMER, "add_timing_nodes", attach)
    graph = SimpleNamespace(instantiate=lambda: operations.append("instantiate"), replay=replay)
    timer.attach(graph)
    samples = [timer.replay() for _ in range(101)]
    assert operations == ["insert", "instantiate"]
    assert timer.hip == "runtime_owner"
    assert clock.records == 3  # No external event recording occurs around replay submission.
    assert samples == [0.25] * 101 and clock.syncs == 102
    TIMER.validate_timing_proof(timer.proof(), samples[1:])
    graph.replay = lambda: None
    with pytest.raises(ValueError, match="stale"):
        timer.replay()
    assert len(timer.replays) == 101


def valid_proof():
    return {"method": TIMER.METHOD, "topology": topology(), "initial_timestamps_ms": [0.0, 0.1],
            "instantiated_after_event_insertion": True,
            "replays": [{"begin_ms_from_anchor": i + 1.0, "end_ms_from_anchor": i + 1.25,
                         "elapsed_ms": 0.25} for i in range(101)]}


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(method="external_event_bracket"),
    lambda p: p.update(instantiated_after_event_insertion=False),
    lambda p: p["topology"].update(final_node_count=2),
    lambda p: p["topology"].update(final_edge_count=1),
    lambda p: p["topology"].update(all_original_roots_depend_on_begin=False),
    lambda p: p["replays"].pop(),
    lambda p: p["replays"].append(p["replays"][-1]),
    lambda p: p["replays"][1].update(begin_ms_from_anchor=1.0),
    lambda p: p["replays"][1].update(end_ms_from_anchor=1.25),
    lambda p: p["replays"][1].update(end_ms_from_anchor=0.0),
    lambda p: p["replays"][1].update(elapsed_ms=0.0),
    lambda p: p["replays"][1].update(elapsed_ms=float("nan")),
    lambda p: p["replays"][1].update(elapsed_ms=0.125),
])
def test_malformed_or_replayed_event_evidence_rejected(mutate):
    proof = valid_proof()
    TIMER.validate_timing_proof(proof, [0.25] * 100)
    mutate(proof)
    with pytest.raises(ValueError):
        TIMER.validate_timing_proof(proof, [0.25] * 100)


def test_anchor_rounding_does_not_replace_direct_positive_event_interval():
    stamp = {"begin_ms_from_anchor": 1_000_000.0, "end_ms_from_anchor": 1_000_000.0, "elapsed_ms": 0.02}
    assert TIMER.validate_replay(stamp, (999_999.0, 999_999.0)) == (1_000_000.0, 1_000_000.0)
    stamp["elapsed_ms"] = 0.0
    with pytest.raises(ValueError):
        TIMER.validate_replay(stamp, (999_999.0, 999_999.0))


def lifecycle_timer(monkeypatch):
    """Track owner release independently from the graph's retained lifetime."""
    log, events = [], []
    state = {'stamp': 0, 'reset': False}

    class Event:
        def __init__(self):
            self.name = len(events)
            events.append(weakref.ref(self))

        def record(self):
            state['stamp'] += 1
            self.stamp = state['stamp']

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return other.stamp - self.stamp

        def __del__(self):
            log.append(('event_released', self.name, state['reset']))

    def event(*, enable_timing):
        assert enable_timing
        return Event()

    def sync():
        log.append('synchronize')

    def reset():
        assert all(owner() is not None for owner in events)
        assert log[-1] == 'synchronize'
        state['reset'] = True
        log.append('reset_while_events_live')

    torch = SimpleNamespace(cuda=SimpleNamespace(Event=event, synchronize=sync))
    timer = TIMER.NativeGraphTimer(torch)
    graph = SimpleNamespace(instantiate=lambda: log.append('instantiate'), reset=reset)
    monkeypatch.setattr(TIMER, 'add_timing_nodes', lambda *args: (object(), topology()))
    return timer, graph, torch, log, events, state


def test_close_resets_graph_before_event_owners_even_with_external_graph_reference(monkeypatch):
    timer, graph, _, log, events, state = lifecycle_timer(monkeypatch)
    timer.attach(graph)
    timer.close()
    assert timer.graph is None and timer.closed and not timer.instantiated
    assert state['reset'] and all(owner() is not None for owner in events)
    timer.close()
    assert log.count('reset_while_events_live') == log.count('synchronize') == 1
    del timer
    gc.collect()
    assert graph is not None  # The caller's local graph still outlives the timer.
    assert all(owner() is None for owner in events)
    releases = [row for row in log if isinstance(row, tuple)]
    assert len(releases) == 3 and all(row[2] is True for row in releases)


@pytest.mark.parametrize('stage', ['event_insertion', 'instantiation'])
def test_partial_attachment_still_owns_graph_for_exception_cleanup(monkeypatch, stage):
    timer, graph, _, log, events, _ = lifecycle_timer(monkeypatch)

    def fail(*args):
        assert timer.graph is graph and all(owner() is not None for owner in events)
        raise RuntimeError(stage)

    if stage == 'event_insertion':
        monkeypatch.setattr(TIMER, 'add_timing_nodes', fail)
    else:
        graph.instantiate = fail
    with pytest.raises(RuntimeError, match=stage):
        try:
            timer.attach(graph)
        finally:
            timer.close()
    assert timer.closed and timer.graph is None
    assert log[-2:] == ['synchronize', 'reset_while_events_live']


def test_sync_failure_still_resets_graph_and_is_not_suppressed(monkeypatch):
    timer, graph, torch, log, _, state = lifecycle_timer(monkeypatch)
    timer.attach(graph)

    def fail_sync():
        log.append('synchronize')
        raise RuntimeError('device synchronization failed')

    torch.cuda.synchronize = fail_sync
    with pytest.raises(RuntimeError, match='device synchronization failed'):
        timer.close()
    assert state['reset'] and timer.closed and timer.graph is None


def test_reset_failure_retains_owners_and_does_not_claim_closed(monkeypatch):
    timer, graph, _, log, events, state = lifecycle_timer(monkeypatch)
    timer.attach(graph)

    def fail_reset():
        assert all(owner() is not None for owner in events)
        raise RuntimeError('graph reset failed')

    graph.reset = fail_reset
    with pytest.raises(RuntimeError, match='graph reset failed'):
        timer.close()
    assert timer.graph is graph and not timer.closed and not state['reset']
    assert all(owner() is not None for owner in events)
    assert log.count('synchronize') == 1  # No implicit cleanup retry.


def test_closed_timer_cannot_be_reused(monkeypatch):
    timer, graph, _, log, _, _ = lifecycle_timer(monkeypatch)
    timer.close()
    assert timer.closed and log == []
    with pytest.raises(ValueError, match='closed'):
        timer.attach(graph)
    with pytest.raises(ValueError, match='instantiated'):
        timer.replay()
    with pytest.raises(ValueError, match='instantiated'):
        timer.proof()
