"""Protected timing for the pinned ROCm runtime, with events inside each graph.

Python event bracketing includes host enqueue stalls. PyTorch external events
are unavailable on this ROCm build, and ordinary captured events retain stale
timestamps. Native HIP event-record nodes provide fresh device timestamps.
The original graph and its output/allocation ownership are preserved.
"""
import ctypes
import math
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _add_timing_nodes(graph, begin, end):
    library_path = Path('/opt/rocm/lib/libamdhip64.so').resolve(strict=True)
    hip = ctypes.CDLL(str(library_path))
    pointer = ctypes.c_void_p
    pointer_array = ctypes.POINTER(pointer)
    size_pointer = ctypes.POINTER(ctypes.c_size_t)
    declarations = {
        'hipGraphGetNodes': [pointer, pointer_array, size_pointer],
        'hipGraphGetRootNodes': [pointer, pointer_array, size_pointer],
        'hipGraphGetEdges': [pointer, pointer_array, pointer_array, size_pointer],
        'hipGraphAddEventRecordNode': [pointer_array, pointer, pointer_array, ctypes.c_size_t, pointer],
        'hipGraphAddDependencies': [pointer, pointer_array, pointer_array, ctypes.c_size_t],
    }
    for name, arguments in declarations.items():
        function = getattr(hip, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    hip.hipGetErrorString.restype = ctypes.c_char_p
    def checked(name, *arguments):
        result = getattr(hip, name)(*arguments)
        if result:
            message = hip.hipGetErrorString(result)
            raise RuntimeError(name + ' returned ' + str(result) + ': ' + str(message))
    raw = pointer(graph.raw_cuda_graph())
    _require(raw.value, 'empty raw graph handle')
    def nodes(name):
        count = ctypes.c_size_t()
        checked(name, raw, None, ctypes.byref(count))
        _require(0 < count.value <= 10000, 'unexpected graph node count')
        array = (pointer * count.value)()
        checked(name, raw, array, ctypes.byref(count))
        return [array[index] for index in range(count.value)]
    def edges():
        count = ctypes.c_size_t()
        checked('hipGraphGetEdges', raw, None, None, ctypes.byref(count))
        _require(count.value <= 100000, 'unexpected graph edge count')
        sources, destinations = (pointer * count.value)(), (pointer * count.value)()
        if count.value:
            checked('hipGraphGetEdges', raw, sources, destinations, ctypes.byref(count))
        return [(sources[index], destinations[index]) for index in range(count.value)]
    original_nodes = nodes('hipGraphGetNodes')
    original_roots = nodes('hipGraphGetRootNodes')
    original_edges = edges()
    original_sinks = sorted(set(original_nodes) - {source for source, _ in original_edges})
    _require(original_sinks and set(original_roots).issubset(original_nodes), 'invalid graph roots or sinks')
    begin_node = pointer()
    end_node = pointer()
    begin_event, end_event = pointer(begin.cuda_event), pointer(end.cuda_event)
    _require(begin_event.value and end_event.value and begin_event.value != end_event.value, 'invalid timing events')
    checked('hipGraphAddEventRecordNode', ctypes.byref(begin_node), raw, None, 0, begin_event)
    sources = (pointer * len(original_roots))(*([begin_node.value] * len(original_roots)))
    destinations = (pointer * len(original_roots))(*original_roots)
    checked('hipGraphAddDependencies', raw, sources, destinations, len(original_roots))
    dependencies = (pointer * len(original_sinks))(*original_sinks)
    checked('hipGraphAddEventRecordNode', ctypes.byref(end_node), raw, dependencies, len(original_sinks), end_event)
    final_nodes = nodes('hipGraphGetNodes')
    final_edges = edges()
    expected_edges = set(original_edges) | {(begin_node.value, root) for root in original_roots} | {
        (sink, end_node.value) for sink in original_sinks}
    _require(set(final_nodes) == set(original_nodes) | {begin_node.value, end_node.value}, 'graph nodes changed')
    _require(set(final_edges) == expected_edges, 'graph dependencies changed')
    _require(nodes('hipGraphGetRootNodes') == [begin_node.value], 'begin event does not precede all work')
    _require(set(final_nodes) - {source for source, _ in final_edges} == {end_node.value}, 'end event does not follow all work')
    return hip, {'api': 'hipGraphAddEventRecordNode', 'library': str(library_path),
            'original_node_count': len(original_nodes),
            'original_edge_count': len(original_edges), 'original_root_count': len(original_roots),
            'original_sink_count': len(original_sinks), 'final_node_count': len(final_nodes),
            'final_edge_count': len(final_edges), 'original_nodes_and_edges_preserved': True,
            'all_original_roots_depend_on_begin': True, 'end_depends_on_all_original_sinks': True,
            'native_event_record_nodes': 2}


class _TimedGraph:
    def __init__(self, torch, graph, anchor, begin, end, previous, hip, topology):
        self._torch = torch
        self._graph = graph
        self._anchor = anchor
        self._begin = begin
        self._end = end
        self._previous = previous
        self._hip = hip  # Keep the runtime and event owners alive with the graph.
        self._topology = topology
        self._replays = 0
        self._elapsed = None

    def replay(self):
        self._graph.replay()
        # Wait outside the event interval. Event synchronization alone may refer
        # to an earlier record until the graph has executed its new record node.
        self._torch.cuda.current_stream().synchronize()
        self._end.synchronize()
        current = (self._anchor.elapsed_time(self._begin), self._anchor.elapsed_time(self._end))
        _require(all(math.isfinite(now) and now > previous for now, previous in zip(current, self._previous)),
                 'stale event timestamps after graph replay')
        elapsed = self._begin.elapsed_time(self._end)
        _require(math.isfinite(elapsed) and elapsed > 0, 'invalid graph device interval')
        # Read the short interval directly: subtracting large anchor-relative
        # float timestamps can round a short kernel interval to zero.
        self._previous = current
        self._elapsed = elapsed
        self._replays += 1

    def measure(self, call):
        before = self._replays
        call()
        _require(self._replays == before + 1, 'timer must execute exactly one graph replay')
        return self._elapsed

    def proof(self):
        return {'schema': 'hip-graph-event-timing-v1', **self._topology,
                'timestamp_updates_verified': self._replays,
                'host_submissions_between_timing_nodes': 0}


def capture_timed_graph(torch, invoke, calls_per_sample):
    """Capture the original calls, add native timing nodes, then instantiate.

    All setup and synchronization stay outside the GPU event interval. The
    returned replay/measure callbacks preserve checked_replays' one-launch
    contract, including freshness validation on all warmups and samples.
    """
    _require(type(calls_per_sample) is int and calls_per_sample > 0, 'positive call multiplicity required')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(stream)
    anchor = torch.cuda.Event(enable_timing=True)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    anchor.record()
    begin.record()
    end.record()
    end.synchronize()
    previous = (anchor.elapsed_time(begin), anchor.elapsed_time(end))
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        for _ in range(calls_per_sample):
            invoke()
    hip, topology = _add_timing_nodes(graph, begin, end)
    graph.instantiate()
    return _TimedGraph(torch, graph, anchor, begin, end, previous, hip, topology)
