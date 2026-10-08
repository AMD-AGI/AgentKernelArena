"""CPU allocation/replay regressions; execute the frozen native wrapper body."""
import ast
from contextlib import contextmanager
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from binding import raw_wrapper_tree
from runtime import CheckedKernel, FP4Case
from write_ownership import WrittenStorages, track_native_allocations

REGRESSION = "minimax_fp4_gemm-7ffa029ca80a67b44dd36927"


class Tensor:
    def __init__(self, shape, values=None, *, events=None, label="tensor", dtype="float32", storage=None, device="cpu"):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.values = values if values is not None else [1.0] * math.prod(shape)
        self.events, self.label = events if events is not None else [], label
        self.storage = storage if storage is not None else self.values

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: id(self.storage))

    def is_floating_point(self):
        return self.dtype in {"float32", "bfloat16"}

    def fill_(self, value):
        self.events.append(("poison", self.label))
        self.values[:] = [value] * len(self.values)
        return self

    def stride(self, dim):
        strides = getattr(self, "_strides", tuple(math.prod(self.shape[i + 1:]) for i in range(len(self.shape))))
        return strides[dim]

    @property
    def T(self):
        result = Tensor(self.shape[::-1], self.values, events=self.events, label=self.label,
                        dtype=self.dtype, storage=self.storage)
        result._strides = tuple(self.stride(i) for i in range(len(self.shape)))[::-1]
        return result


class Torch:
    Tensor, dtype = Tensor, str
    float32, bfloat16 = "float32", "bfloat16"

    def __init__(self, events):
        self.events, self.capturing = events, False
        self.cuda = SimpleNamespace(is_current_stream_capturing=lambda: self.capturing)
        self.allocations = []
        self.return_alias = None

    def empty(self, shape, **kwargs):
        self.events.append(("allocate", tuple(shape), dict(kwargs)))
        if self.return_alias is not None:
            return self.return_alias
        # Model the observed allocator reuse: four valid stale partials sum to 1.
        tensor = Tensor(shape, [0.25] * math.prod(shape), events=self.events, label="partial", **kwargs)
        self.allocations.append(tensor)
        return tensor

    def empty_like(self, other, **kwargs):
        return self.empty(other.shape, **{"dtype": other.dtype, **kwargs})


class Kernel:
    def __init__(self, callback):
        self.callback = callback

    def __getitem__(self, grid):
        return self.callback


def frozen_regression(mode="no_op", *, capturing=False):
    case = next(c for c in json.loads((ROOT / "cases.json").read_text())["cases"] if c["case_id"] == REGRESSION)
    args = case["scalars"]["native_arguments"]
    assert (args["M"], args["N"], args["K"], args["NUM_KSPLIT"]) == (1, 768, 3072, 4)
    events, captured = [], []
    torch = Torch(events)
    torch.capturing = capturing
    tensors = {
        name: Tensor(meta["shape"], [7.0], events=events, label=name, dtype=meta["dtype"])
        for name, meta in case["tensors"].items() if name in {"x", "w", "x_scales", "w_scales"}
    }
    for name in ("x_scales", "w_scales"):
        tensors[name]._strides = tuple(case["tensors"][name]["strides"])
    y = Tensor((1, 768), [1.0] * 768, events=events, label="y", dtype="bfloat16")
    def gemm(*values, **kwargs):
        partial = values[2]
        def run():
            events.append(("gemm", mode))
            if mode != "no_op":
                partial.values[:] = [0.25 if mode == "correct" else 0.0] * len(partial.values)
        captured.append(run) if torch.capturing else run()
        return SimpleNamespace(name="compiled_cpu_model", hash="cpu-model-kernel-hash")
    def reduce(partial, output, *values, **kwargs):
        def run():
            events.append(("reduce",))
            output.values[:] = [sum(partial.values[i * 768 + j] for i in range(4)) for j in range(768)]
        captured.append(run) if torch.capturing else run()
    namespace = {
        "torch": torch, "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        "arch_info": SimpleNamespace(is_fp4_avail=lambda: True),
        "_LOGGER": SimpleNamespace(info=lambda *args: None),
        "deserialize_str": lambda value: value, "serialize_dict": lambda value: value,
        "_USE_GEMM_SPLITK_BF16": False, "_triton_gemm_afp4wfp4_kernel": Kernel(gemm),
        "_gemm_splitk_reduce_kernel": Kernel(reduce),
    }
    tree = raw_wrapper_tree((ROOT / "ut/native/wrapper.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"get_splitk", "gemm_afp4wfp4_", "gemm_afp4wfp4"}]
    exec(compile(tree, "frozen_fp4_wrapper", "exec"), namespace)
    call = namespace["gemm_afp4wfp4"]
    def invoke():
        return call(tensors["x"], tensors["w"], tensors["x_scales"], tensors["w_scales"],
                    "bfloat16", y, dict(case["scalars"]["resolved_config"]), False)
    return SimpleNamespace(torch=torch, events=events, captured=captured, inputs=tensors,
                           output=y, call=call, invoke=invoke, case=case)


def test_saved_splitk_case_reuses_stale_partials_without_ownership_then_rejects_noop():
    before = frozen_regression()
    assert before.invoke().values == [1.0] * 768  # The old no-op false acceptance.
    after = frozen_regression()
    owner = track_native_allocations(after.call, after.inputs.values())
    original_empty = after.torch.empty
    with owner.invocation(capturing=False):
        result = after.invoke()
    assert all(math.isnan(value) for value in result.values)
    assert after.events[0][0] == "allocate"
    assert after.events[1:3] == [("poison", "partial"), ("gemm", "no_op")]
    assert after.torch.empty == original_empty  # The global module was not patched.
    assert all(t.values == [7.0] for t in after.inputs.values())


@pytest.mark.parametrize("mode", ["no_op", "wrong_output", "correct"])
def test_captured_storage_is_retained_and_poisoned_before_each_replay_outside_timer(mode):
    state = frozen_regression(mode, capturing=True)
    owner = track_native_allocations(state.call, state.inputs.values())
    with owner.invocation(capturing=True):
        state.invoke()
    assert not any(event[0] == "poison" for event in state.events)
    assert owner.tensors == tuple(state.torch.allocations)
    with pytest.raises(RuntimeError, match="outside"):
        owner.initialize(state.output)
    state.torch.capturing = False
    for _ in range(2):
        owner.tensors[0].values[:] = [0.25] * (4 * 768)
        owner.initialize(state.output)
        state.events.append(("timer_begin",))
        owner.before_graph_replay()
        for operation in state.captured:
            operation()
        state.events.append(("timer_end",))
        timed = state.events[state.events.index(("timer_begin",)) + 1:]
        assert not any(event[0] == "poison" for event in timed[:-1])
        if mode == "correct":
            assert state.output.values == [1.0] * 768
        else:
            assert state.output.values != [1.0] * 768
        with pytest.raises(RuntimeError, match="each graph replay"):
            owner.before_graph_replay()
        state.events.clear()
    assert all(t.values == [7.0] for t in state.inputs.values())


def test_readonly_alias_is_rejected_before_any_poison_and_factory_arguments_are_unchanged():
    events = []
    torch = Torch(events)
    readonly = Tensor((4,), events=events, label="input")
    torch.return_alias = readonly
    owner = WrittenStorages(torch, [readonly])
    with owner.invocation(capturing=False), pytest.raises(RuntimeError, match="read-only"):
        owner.record(torch.empty((4,), dtype="float32"))
    assert readonly.values == [1.0] * 4
    assert not any(event[0] == "poison" for event in events)
    with pytest.raises(RuntimeError, match="read-only"):
        owner.initialize(Tensor((4,), storage=readonly.storage))


def test_reference_and_candidate_keep_identical_allocation_reset_and_alias_behavior():
    traces = []
    for _leg in ("reference", "candidate"):
        state = frozen_regression("correct")
        owner = track_native_allocations(state.call, state.inputs.values())
        with owner.invocation(capturing=False):
            assert state.invoke() is state.output
        assert state.output.values == [1.0] * 768
        traces.append(state.events)
    assert traces[0] == traces[1]
    assert traces[0][0] == ("allocate", (4, 1, 768), {"dtype": "float32", "device": "cpu"})


def test_runtime_initialization_does_not_poison_ignored_y_for_partial_output():
    events = []
    torch = Torch(events)
    ignored = Tensor((1,), events=events, label="ignored_y")
    partial = Tensor((4,), events=events, label="partial")
    state = object.__new__(FP4Case)
    state.returns_partials, state.inputs, state.result = True, {"y": ignored}, partial
    state.writes = WrittenStorages(torch, [ignored])
    state.initialize()
    assert ignored.values == [1.0]
    assert all(math.isnan(value) for value in partial.values)


def test_private_torch_owners_do_not_change_global_factories_or_other_modules():
    torch = Torch([])
    factories = (torch.empty, torch.empty_like)
    functions = []
    for _ in range(2):
        namespace = {"torch": torch}
        exec("def native(): return torch.empty((4,), dtype='float32')", namespace)
        functions.append(namespace["native"])
    first, second = [track_native_allocations(function, []) for function in functions]
    with first.invocation(capturing=False):
        functions[0]()
    assert len(first.tensors) == 1 and not second.tensors
    with second.invocation(capturing=False):
        functions[1]()
    assert first.tensors != second.tensors
    assert (torch.empty, torch.empty_like) == factories


def test_runtime_capture_and_measure_keep_poison_outside_capture_and_timer():
    model = frozen_regression("no_op")
    events, torch = model.events, model.torch
    class Stream:
        def wait_stream(self, other):
            pass
    @contextmanager
    def stream_scope(stream):
        yield
    @contextmanager
    def graph_scope(graph):
        events.append(("capture_begin",))
        torch.capturing = True
        try:
            yield
        finally:
            torch.capturing = False
            events.append(("capture_end",))
    class Graph:
        def replay(self):
            events.append(("graph_replay",))
            for operation in model.captured:
                operation()
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            events.append(("timer_event",))
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            return 1.0
    torch.cuda.Stream = Stream
    torch.cuda.current_stream = Stream
    torch.cuda.stream = stream_scope
    torch.cuda.graph = graph_scope
    torch.cuda.CUDAGraph = Graph
    torch.cuda.Event = Event
    state = object.__new__(FP4Case)
    state.torch, state.case, state.call = torch, model.case, model.call
    state.inputs = {**model.inputs, "y": model.output}
    state.result, state.returns_partials, state.graph = model.output, False, None
    state.calls, state.proof = 0, {}
    state.writes = track_native_allocations(state.call, model.inputs.values())
    namespace = state.call.__globals__
    state.probe = CheckedKernel(namespace["_triton_gemm_afp4wfp4_kernel"], model.case["scalars"]["native_arguments"])
    namespace["_triton_gemm_afp4wfp4_kernel"] = state.probe
    with pytest.raises(ValueError, match="captured"):
        state.measure(state.replay)
    state.capture_graph()
    capture = events[events.index(("capture_begin",)) + 1:events.index(("capture_end",))]
    assert not any(event[0] == "poison" for event in capture)
    assert events[-2:] == [("poison", "partial"), ("poison", "y")]
    assert state.calls == 4 and len(state.writes.tensors) == 1
    events.clear()
    state.initialize()
    assert state.measure(state.replay) == 1.0
    timed = events[events.index(("timer_event",)) + 1:-1]
    assert ("graph_replay",) in timed and not any(event[0] == "poison" for event in timed)
    assert all(math.isnan(value) for value in state.result.values)
    with pytest.raises(RuntimeError, match="each graph replay"):
        state.replay()


def test_missing_writable_allocation_fails_closed():
    owner = WrittenStorages(Torch([]), [])
    with pytest.raises(RuntimeError, match="allocation sequence"):
        owner.validate_shapes([(4, 1, 768)])
