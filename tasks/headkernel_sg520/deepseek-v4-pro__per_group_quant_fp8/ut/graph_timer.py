"""Time the captured native work with HIP event-record nodes inside its graph."""

import ctypes
import math
from pathlib import Path


METHOD = "hip_graph_event_record_nodes"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hip_library():
    # This is the HIP runtime shipped in the task's pinned ROCm image.
    hip = ctypes.CDLL(str(Path("/opt/rocm/lib/libamdhip64.so").resolve(strict=True)))
    pointer = ctypes.c_void_p
    pointer_array = ctypes.POINTER(pointer)
    size_pointer = ctypes.POINTER(ctypes.c_size_t)
    declarations = {
        "hipGraphGetNodes": [pointer, pointer_array, size_pointer],
        "hipGraphGetRootNodes": [pointer, pointer_array, size_pointer],
        "hipGraphGetEdges": [pointer, pointer_array, pointer_array, size_pointer],
        "hipGraphAddEventRecordNode": [pointer_array, pointer, pointer_array, ctypes.c_size_t, pointer],
        "hipGraphAddDependencies": [pointer, pointer_array, pointer_array, ctypes.c_size_t],
    }
    for name, arguments in declarations.items():
        function = getattr(hip, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    hip.hipGetErrorString.restype = ctypes.c_char_p
    return hip


def add_timing_nodes(graph, begin, end):
    """Preserve every captured node/edge and bracket all original roots/sinks."""
    hip = hip_library()
    pointer = ctypes.c_void_p

    def checked(name, *arguments):
        status = getattr(hip, name)(*arguments)
        if status:
            raise RuntimeError(f"{name} returned {status}: {hip.hipGetErrorString(status)!r}")

    raw = pointer(graph.raw_cuda_graph())
    require(raw.value, "missing retained HIP graph")

    def nodes(name):
        count = ctypes.c_size_t()
        checked(name, raw, None, ctypes.byref(count))
        require(0 < count.value <= 10000, "unexpected graph node count")
        capacity = count.value
        array = (pointer * capacity)()
        checked(name, raw, array, ctypes.byref(count))
        require(count.value == capacity, "graph node inventory changed during inspection")
        result = [array[index] for index in range(count.value)]
        require(all(result) and len(set(result)) == len(result), "invalid graph node handles")
        return result

    def edges():
        count = ctypes.c_size_t()
        checked("hipGraphGetEdges", raw, None, None, ctypes.byref(count))
        require(count.value <= 100000, "unexpected graph edge count")
        capacity = count.value
        sources, destinations = (pointer * capacity)(), (pointer * capacity)()
        if capacity:
            checked("hipGraphGetEdges", raw, sources, destinations, ctypes.byref(count))
        require(count.value == capacity, "graph edge inventory changed during inspection")
        result = [(sources[index], destinations[index]) for index in range(count.value)]
        require(len(set(result)) == len(result), "duplicate graph edges")
        return result

    original_nodes = nodes("hipGraphGetNodes")
    original_roots = nodes("hipGraphGetRootNodes")
    original_edges = edges()
    require(all(source in original_nodes and target in original_nodes for source, target in original_edges),
            "graph edge escapes the captured nodes")
    original_sinks = sorted(set(original_nodes) - {source for source, _ in original_edges})
    require(original_sinks and set(original_roots) == set(original_nodes) - {target for _, target in original_edges},
            "invalid captured graph boundary")
    begin_node, end_node = pointer(), pointer()
    begin_event, end_event = pointer(begin.cuda_event), pointer(end.cuda_event)
    require(begin_event.value and end_event.value and begin_event.value != end_event.value,
            "timing requires distinct initialized event handles")
    checked("hipGraphAddEventRecordNode", ctypes.byref(begin_node), raw, None, 0, begin_event)
    sources = (pointer * len(original_roots))(*([begin_node.value] * len(original_roots)))
    destinations = (pointer * len(original_roots))(*original_roots)
    checked("hipGraphAddDependencies", raw, sources, destinations, len(original_roots))
    dependencies = (pointer * len(original_sinks))(*original_sinks)
    checked("hipGraphAddEventRecordNode", ctypes.byref(end_node), raw, dependencies, len(original_sinks), end_event)
    require(begin_node.value and end_node.value and begin_node.value != end_node.value
            and not {begin_node.value, end_node.value}.intersection(original_nodes), "invalid event node handles")
    final_nodes = nodes("hipGraphGetNodes")
    final_edges = edges()
    expected_edges = set(original_edges) | {(begin_node.value, root) for root in original_roots} | {
        (sink, end_node.value) for sink in original_sinks}
    require(set(final_nodes) == set(original_nodes) | {begin_node.value, end_node.value},
            "timing insertion changed captured work nodes")
    require(set(final_edges) == expected_edges, "timing insertion changed captured work dependencies")
    require(nodes("hipGraphGetRootNodes") == [begin_node.value], "begin event does not precede every root")
    require(set(final_nodes) - {source for source, _ in final_edges} == {end_node.value},
            "end event does not follow every sink")
    return hip, {"api": "hipGraphAddEventRecordNode", "original_node_count": len(original_nodes),
            "original_edge_count": len(original_edges), "original_root_count": len(original_roots),
            "original_sink_count": len(original_sinks), "final_node_count": len(final_nodes),
            "final_edge_count": len(final_edges), "original_nodes_and_edges_preserved": True,
            "all_original_roots_depend_on_begin": True, "end_depends_on_all_original_sinks": True}


def validate_replay(stamp, previous):
    require(isinstance(stamp, dict) and set(stamp) == {"begin_ms_from_anchor", "end_ms_from_anchor", "elapsed_ms"},
            "invalid graph timing record")
    require(all(type(value) in (int, float) and math.isfinite(value) for value in stamp.values()),
            "nonfinite graph event timestamp")
    begin, end = stamp["begin_ms_from_anchor"], stamp["end_ms_from_anchor"]
    require(begin > previous[0] and end > previous[1], "stale graph event timestamps")
    # Anchor-relative timestamps are float32 milliseconds and can round to the
    # same value in a long run. The direct begin/end interval must be positive.
    require(end >= begin and stamp["elapsed_ms"] > 0, "invalid graph event interval")
    return begin, end


class NativeGraphTimer:
    def __init__(self, torch):
        self.torch = torch
        self.graph = None
        self.closed = False
        self.instantiated = False
        self.anchor = torch.cuda.Event(enable_timing=True)
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.anchor.record()
        self.begin.record()
        self.end.record()
        self.end.synchronize()
        self.initial = [self.anchor.elapsed_time(self.begin), self.anchor.elapsed_time(self.end)]
        self.previous = tuple(self.initial)
        self.replays = []

    def attach(self, graph):
        require(not self.closed and self.graph is None, "graph timer is closed or already attached")
        # Own the graph before adding event nodes so the caller's finally block
        # can reset even a partially modified or uninstantiated graph.
        self.graph = graph
        self.hip, self.topology = add_timing_nodes(graph, self.begin, self.end)
        # Keep the HIP library and event owners alive for the retained graph.
        graph.instantiate()
        self.instantiated = True

    def close(self):
        """Destroy graph references to events while their Python owners live."""
        if self.graph is None:
            self.closed = True
            return
        try:
            self.torch.cuda.synchronize()
        finally:
            # Attempt reset even if synchronization reports a device error.
            # If reset itself fails, retain ownership and propagate the error;
            # never report successful cleanup or a completed performance call.
            self.graph.reset()
            self.graph = None
            self.instantiated = False
            self.closed = True

    def replay(self):
        require(self.graph is not None and self.instantiated, "graph timer has no instantiated graph")
        self.graph.replay()
        self.torch.cuda.synchronize()
        stamp = {"begin_ms_from_anchor": self.anchor.elapsed_time(self.begin),
                 "end_ms_from_anchor": self.anchor.elapsed_time(self.end),
                 "elapsed_ms": self.begin.elapsed_time(self.end)}
        self.previous = validate_replay(stamp, self.previous)
        self.replays.append(stamp)
        return stamp["elapsed_ms"]

    def proof(self):
        require(self.graph is not None and self.instantiated, "graph timer has no instantiated graph")
        return {"method": METHOD, "topology": self.topology, "initial_timestamps_ms": self.initial,
                "replays": self.replays, "instantiated_after_event_insertion": True}


def validate_timing_proof(proof, samples):
    require(isinstance(proof, dict) and proof.get("method") == METHOD
            and proof.get("instantiated_after_event_insertion") is True, "missing native graph timing proof")
    topology = proof.get("topology", {})
    require(isinstance(topology, dict) and topology.get("api") == "hipGraphAddEventRecordNode"
            and all(topology.get(key) is True for key in ("original_nodes_and_edges_preserved",
                    "all_original_roots_depend_on_begin", "end_depends_on_all_original_sinks")),
            "graph timing boundary was not verified")
    counts = [topology.get(key) for key in ("original_node_count", "original_edge_count",
              "original_root_count", "original_sink_count", "final_node_count", "final_edge_count")]
    require(all(type(value) is int and value >= 0 for value in counts), "invalid graph topology counts")
    nodes, edges, roots, sinks, final_nodes, final_edges = counts
    require(0 < roots <= nodes and 0 < sinks <= nodes and final_nodes == nodes + 2
            and final_edges == edges + roots + sinks, "graph timing topology changed captured work")
    previous = proof.get("initial_timestamps_ms")
    require(isinstance(previous, list) and len(previous) == 2
            and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in previous)
            and previous[1] >= previous[0], "invalid initial event timestamps")
    replays = proof.get("replays")
    require(isinstance(replays, list) and len(replays) == len(samples) + 1,
            "initial validation and every measured graph replay require event timestamps")
    for stamp in replays:
        previous = validate_replay(stamp, previous)
    require([stamp["elapsed_ms"] for stamp in replays[1:]] == samples,
            "device samples differ from graph event intervals")
