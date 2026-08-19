import importlib.util
import sys
import types
import warnings
from contextlib import nullcontext
from pathlib import Path

import pytest


HELPER = Path(__file__).parents[1] / "src/tools/perf/aka_benchmark.py"


class _FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.synchronize_calls = 0

    def is_available(self):
        return self.available

    def synchronize(self):
        self.synchronize_calls += 1

    def Stream(self):
        return types.SimpleNamespace(wait_stream=lambda _stream: None)

    def current_stream(self):
        return object()


def _load_helper(monkeypatch, *, available=True):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = _FakeCuda(available=available)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    module_name = f"_aka_benchmark_test_{id(fake_torch)}"
    spec = importlib.util.spec_from_file_location(module_name, HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gpu_absence_never_uses_cpu_wall_clock(monkeypatch):
    helper = _load_helper(monkeypatch, available=False)
    calls = []

    with pytest.raises(RuntimeError, match="CPU wall-clock timing is not a valid"):
        helper.benchmark_cuda_graph_or_events(lambda: calls.append(True))

    assert calls == []


@pytest.mark.parametrize("elapsed", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_gpu_event_elapsed_time_is_rejected(monkeypatch, elapsed):
    helper = _load_helper(monkeypatch)
    event = types.SimpleNamespace(elapsed_time=lambda _end: elapsed)

    with pytest.raises(RuntimeError, match="invalid GPU event elapsed time"):
        helper._event_elapsed_ms(event, object())


def test_graph_capture_adapts_repeats_and_returns_samples(monkeypatch):
    helper = _load_helper(monkeypatch)
    calls = []
    captured_repeats = []

    def capture(fn, repeats, stream, prepare_fn=None):
        del stream, prepare_fn
        captured_repeats.append(repeats)
        for _ in range(repeats):
            fn()
        return object()

    replay_calls = []

    def replay(graph, stream, samples, calls_per_replay, prepare_fn=None):
        del graph, stream, prepare_fn
        replay_calls.append((samples, calls_per_replay))
        if len(replay_calls) <= 2:
            return [0.25]
        return [0.08] * samples

    monkeypatch.setattr(helper, "_capture_graph", capture)
    monkeypatch.setattr(helper, "_graph_replay_samples", replay)

    samples, metadata = helper.benchmark_cuda_graph_or_events_samples(
        lambda: calls.append(True),
        warmup=2,
        repetition=3,
        target_ms=1.0,
        estimate_reps=5,
        max_graph_repeats=99,
    )

    assert samples == [0.08, 0.08, 0.08]
    assert captured_repeats == [5, 4]
    assert replay_calls == [(1, 5), (1, 5), (1, 4), (3, 4)]
    assert metadata["benchmark_method"] == "cuda_graph"
    assert metadata["benchmark_effective_repeats"] == 4
    assert metadata["benchmark_samples"] == 3
    assert metadata["benchmark_warmup"] == 2
    assert len(calls) == 2 + 5 + 4


def test_empty_graph_capture_falls_back_to_gpu_events(monkeypatch):
    helper = _load_helper(monkeypatch)
    monkeypatch.setattr(
        helper,
        "_capture_graph",
        lambda fn, repeats, stream, prepare_fn=None: object(),
    )
    monkeypatch.setattr(
        helper,
        "_graph_replay_samples",
        lambda graph, stream, samples, calls_per_replay, prepare_fn=None: [1.0e-8],
    )
    monkeypatch.setattr(
        helper,
        "benchmark_cuda_event_samples",
        lambda fn, repetition, prepare_fn=None: [0.4] * repetition,
    )

    samples, metadata = helper.benchmark_cuda_graph_or_events_samples(
        lambda: None,
        warmup=0,
        repetition=2,
    )

    assert samples == [0.4, 0.4]
    assert metadata["benchmark_method"] == "cuda_event_fallback"
    assert metadata["benchmark_fallback_reason"] == "empty_cuda_graph_capture"


def test_pytorch_empty_graph_warning_is_detected(monkeypatch):
    helper = _load_helper(monkeypatch)

    class _WarnEmptyGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            warnings.warn("The CUDA Graph is empty.", UserWarning)

    monkeypatch.setattr(helper.torch.cuda, "CUDAGraph", object, raising=False)
    monkeypatch.setattr(
        helper.torch.cuda, "stream", lambda _stream: nullcontext(), raising=False
    )
    monkeypatch.setattr(
        helper.torch.cuda, "graph", lambda _graph: _WarnEmptyGraph(), raising=False
    )

    with pytest.raises(helper._EmptyGraphCapture, match="empty CUDA/HIP Graph"):
        helper._capture_graph(lambda: None, repeats=1, stream=object())


def test_capture_error_has_explicit_event_fallback_reason(monkeypatch):
    helper = _load_helper(monkeypatch)

    def fail_capture(fn, repeats, stream, prepare_fn=None):
        del fn, repeats, stream, prepare_fn
        raise ValueError("host read during capture\nwith detail")

    monkeypatch.setattr(helper, "_capture_graph", fail_capture)
    monkeypatch.setattr(
        helper,
        "benchmark_cuda_event_samples",
        lambda fn, repetition, prepare_fn=None: [0.5] * repetition,
    )

    mean_ms, metadata = helper.benchmark_cuda_graph_or_events(
        lambda: None,
        warmup=0,
        repetition=2,
    )

    assert mean_ms == 0.5
    assert metadata["benchmark_method"] == "cuda_event_fallback"
    assert metadata["benchmark_fallback_reason"].startswith(
        "cuda_graph_failed: ValueError: host read during capture with detail"
    )


def test_graph_can_be_explicitly_disabled_with_auditable_reason(monkeypatch):
    helper = _load_helper(monkeypatch)
    monkeypatch.setattr(
        helper,
        "benchmark_cuda_event_samples",
        lambda fn, repetition, prepare_fn=None: [0.75] * repetition,
    )
    monkeypatch.setattr(
        helper,
        "_capture_graph",
        lambda *args: pytest.fail("capture should not be attempted"),
    )

    mean_ms, metadata = helper.benchmark_cuda_graph_or_events(
        lambda: None,
        warmup=0,
        repetition=2,
        use_cuda_graph=False,
        fallback_reason="source_wrapper_uses_host_scalar",
    )

    assert mean_ms == 0.75
    assert metadata["benchmark_method"] == "cuda_event_fallback"
    assert metadata["benchmark_fallback_reason"] == "source_wrapper_uses_host_scalar"


def test_baseline_force_event_environment_skips_graph(monkeypatch):
    helper = _load_helper(monkeypatch)
    monkeypatch.setenv("AKA_BENCHMARK_FORCE_EVENT", "1")
    monkeypatch.setattr(
        helper,
        "benchmark_cuda_event_samples",
        lambda fn, repetition, prepare_fn=None: [0.625] * repetition,
    )
    monkeypatch.setattr(
        helper,
        "_capture_graph",
        lambda *args, **kwargs: pytest.fail("forced Event baseline captured a graph"),
    )

    mean_ms, metadata = helper.benchmark_cuda_graph_or_events(
        lambda: None,
        warmup=0,
        repetition=2,
    )

    assert mean_ms == 0.625
    assert metadata["benchmark_method"] == "cuda_event_fallback"
    assert metadata["benchmark_fallback_reason"] == "forced_event_baseline"


def test_prepare_fn_forces_one_stateful_call_per_graph_replay(monkeypatch):
    helper = _load_helper(monkeypatch)
    captured_repeats = []
    replay_calls = []
    seen_prepare = []

    def capture(fn, repeats, stream, prepare_fn=None):
        del fn, stream
        captured_repeats.append(repeats)
        seen_prepare.append(prepare_fn)
        return object()

    def replay(graph, stream, samples, calls_per_replay, prepare_fn=None):
        del graph, stream
        replay_calls.append((samples, calls_per_replay))
        seen_prepare.append(prepare_fn)
        return [0.25] if len(replay_calls) <= 3 else [0.2] * samples

    monkeypatch.setattr(helper, "_capture_graph", capture)
    monkeypatch.setattr(helper, "_graph_replay_samples", replay)

    prepare = lambda: None
    samples, metadata = helper.benchmark_cuda_graph_or_events_samples(
        lambda: None,
        prepare_fn=prepare,
        warmup=0,
        repetition=3,
        target_ms=10.0,
        estimate_reps=9,
        max_graph_repeats=99,
    )

    assert samples == [0.2, 0.2, 0.2]
    assert captured_repeats == [1, 1]
    assert replay_calls == [(1, 1), (1, 1), (1, 1), (3, 1)]
    assert all(item is prepare for item in seen_prepare)
    assert metadata["benchmark_method"] == "cuda_graph"
    assert metadata["benchmark_effective_repeats"] == 1


def test_timed_run_binds_outputs_to_the_exact_measured_graph(monkeypatch):
    helper = _load_helper(monkeypatch)
    captured_output = object()

    class Graph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    final_graph = Graph()
    captures = 0

    def capture(fn, repeats, stream, prepare_fn=None, output_holder=None):
        nonlocal captures
        del repeats, stream, prepare_fn
        captures += 1
        if output_holder is not None:
            output_holder[:] = [fn()]
            return final_graph
        return Graph()

    def replay(graph, stream, samples, calls_per_replay, prepare_fn=None):
        del graph, stream, calls_per_replay, prepare_fn
        return [0.25] * samples

    class TimedRun:
        def __init__(self):
            self.outputs = None
            self._rerun = None

        def _bind(self, rerun, outputs=None):
            self._rerun = rerun
            self.outputs = outputs

        def rerun(self):
            return self._rerun()

    monkeypatch.setattr(helper, "_capture_graph", capture)
    monkeypatch.setattr(helper, "_graph_replay_samples", replay)
    monkeypatch.setattr(
        helper.torch.cuda, "stream", lambda _stream: nullcontext(), raising=False
    )
    timed = TimedRun()

    _samples, metadata = helper.benchmark_cuda_graph_or_events_samples(
        lambda: captured_output,
        warmup=0,
        repetition=2,
        timed_run=timed,
    )

    assert captures == 2
    assert metadata["benchmark_method"] == "cuda_graph"
    assert timed.outputs is captured_output
    assert timed.rerun() is captured_output
    assert final_graph.replays == 1


def test_hip_source_policy_accepts_current_stream_launch(monkeypatch, tmp_path):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "auto stream = at::hip::getCurrentHIPStream();\n"
        "hipLaunchKernelGGL(kernel, grid, block, 0, stream, output);\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (True, None)


def test_hip_source_policy_accepts_split_wrapper_stream(monkeypatch, tmp_path):
    helper = _load_helper(monkeypatch)
    binding = tmp_path / "binding.cpp"
    kernel = tmp_path / "kernel.hip"
    binding.write_text(
        "void wrapper(void *output) {\n"
        "  auto launch_stream = at::cuda::getCurrentCUDAStream().stream();\n"
        "  launcher(output, launch_stream);\n"
        "}\n"
    )
    kernel.write_text(
        "void launcher(void *output, hipStream_t launch_stream) {\n"
        "  kernel<<<grid, block, 0, launch_stream>>>(output);\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(binding, kernel) == (True, None)


def test_hip_source_policy_accepts_active_rocm_preprocessor_branch(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "void target() {\n"
        "#if defined(USE_ROCM)\n"
        "  hipStream_t stream = at::hip::getCurrentHIPStream();\n"
        "#else\n"
        "  hipStream_t stream = nullptr;\n"
        "#endif\n"
        "  kernel<<<grid, block, 0, stream>>>(output);\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (True, None)


def test_hip_source_policy_does_not_assume_compound_rocm_condition(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "void target() {\n"
        "#if defined(USE_ROCM) && 0\n"
        "  hipStream_t stream = at::hip::getCurrentHIPStream();\n"
        "#else\n"
        "  hipStream_t stream = nullptr;\n"
        "#endif\n"
        "  kernel<<<grid, block, 0, stream>>>(output);\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_launch_stream_unverified",
    )


def test_hip_source_policy_rejects_stream_reassigned_before_launch(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "void target() {\n"
        "  auto stream = at::hip::getCurrentHIPStream();\n"
        "  stream = 0;\n"
        "  kernel<<<grid, block, 0, stream>>>(output);\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_launch_stream_unverified",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "if (use_default) { stream = 0; }",
        "for (int i = 0; i < count; ++i) { stream = 0; }",
    ],
)
def test_hip_source_policy_rejects_stream_mutated_in_child_scope(
    monkeypatch, tmp_path, mutation
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "void target(bool use_default, int count) {\n"
        "  auto stream = at::hip::getCurrentHIPStream();\n"
        f"  {mutation}\n"
        "  kernel<<<grid, block, 0, stream>>>(output);\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_launch_stream_unverified",
    )


def test_hip_source_policy_rejects_shadowed_stream_at_launch(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "void target() {\n"
        "  auto stream = at::hip::getCurrentHIPStream();\n"
        "  {\n"
        "    hipStream_t stream;\n"
        "    kernel<<<grid, block, 0, stream>>>(output);\n"
        "  }\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_launch_stream_unverified",
    )


def test_hip_source_policy_rejects_parameter_mutated_in_callee(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    binding = tmp_path / "binding.cpp"
    kernel = tmp_path / "kernel.hip"
    binding.write_text(
        "void wrapper(void *output) {\n"
        "  auto stream = at::cuda::getCurrentCUDAStream().stream();\n"
        "  launcher(output, stream, true);\n"
        "}\n"
    )
    kernel.write_text(
        "void launcher(void *output, hipStream_t stream, bool bad) {\n"
        "  if (bad) { stream = 0; }\n"
        "  kernel<<<grid, block, 0, stream>>>(output);\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(binding, kernel) == (
        False,
        "hip_source_launch_stream_unverified",
    )


@pytest.mark.parametrize(
    "source_text",
    [
        (
            "void target(bool condition) {\n"
            "  kernel<<<grid, block, 0, "
            "condition ? at::hip::getCurrentHIPStream() : 0>>>(output);\n"
            "}\n"
        ),
        (
            "void target(bool condition) {\n"
            "  auto stream = condition ? at::hip::getCurrentHIPStream() : 0;\n"
            "  kernel<<<grid, block, 0, stream>>>(output);\n"
            "}\n"
        ),
    ],
)
def test_hip_source_policy_rejects_conditional_current_stream_expression(
    monkeypatch, tmp_path, source_text
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(source_text)

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_launch_stream_unverified",
    )


@pytest.mark.parametrize(
    "getter",
    [
        "getCurrentHIPStream()",
        "evil::getCurrentCUDAStream()",
    ],
)
def test_hip_source_policy_rejects_untrusted_current_stream_getter_name(
    monkeypatch, tmp_path, getter
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "hipStream_t getCurrentHIPStream() { return 0; }\n"
        f"kernel<<<grid, block, 0, {getter}>>>(output);\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_launch_stream_unverified",
    )


def test_hip_source_policy_rejects_global_current_stream_for_function_launch(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "namespace detail {\n"
        "auto stream = at::hip::getCurrentHIPStream();\n"
        "void target() {\n"
        "  kernel<<<grid, block, 0, stream>>>(output);\n"
        "}\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_launch_stream_unverified",
    )


def test_hip_source_policy_rejects_parameter_when_call_passes_default_stream(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    binding = tmp_path / "binding.cpp"
    kernel = tmp_path / "kernel.hip"
    binding.write_text(
        "void unrelated() {\n"
        "  auto stream = at::cuda::getCurrentCUDAStream().stream();\n"
        "}\n"
        "void wrapper(void *output) { launcher(output, 0); }\n"
    )
    kernel.write_text(
        "void launcher(void *output, hipStream_t stream) {\n"
        "  kernel<<<grid, block, 0, stream>>>(output);\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(binding, kernel) == (
        False,
        "hip_source_launch_stream_unverified",
    )


def test_hip_source_policy_requires_every_split_wrapper_call_to_be_current(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    binding = tmp_path / "binding.cpp"
    kernel = tmp_path / "kernel.hip"
    binding.write_text(
        "void good(void *output) {\n"
        "  auto stream = at::cuda::getCurrentCUDAStream().stream();\n"
        "  launcher(output, stream);\n"
        "}\n"
        "void bad(void *output) { launcher(output, 0); }\n"
    )
    kernel.write_text(
        "void launcher(void *output, hipStream_t stream) {\n"
        "  kernel<<<grid, block, 0, stream>>>(output);\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(binding, kernel) == (
        False,
        "hip_source_launch_stream_unverified",
    )


def test_hip_source_policy_does_not_cross_prove_qualified_function_names(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    binding = tmp_path / "binding.cpp"
    kernel = tmp_path / "kernel.hip"
    binding.write_text(
        "void wrapper(void *output) {\n"
        "  auto stream = at::cuda::getCurrentCUDAStream().stream();\n"
        "  other::launcher(output, stream);\n"
        "}\n"
        "void other::launcher(void *output, hipStream_t stream) {}\n"
    )
    kernel.write_text(
        "void target::launcher(void *output, hipStream_t stream) {\n"
        "  kernel<<<grid, block, 0, stream>>>(output);\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(binding, kernel) == (
        False,
        "hip_source_launch_stream_unverified",
    )


@pytest.mark.parametrize(
    "launch",
    [
        "kernel<<<grid, block>>>(output);",
        "kernel<<<grid, block, 0>>>(output);",
        "hipLaunchKernelGGL(kernel, grid, block, 0, 0, output);",
    ],
)
def test_hip_source_policy_rejects_legacy_default_stream(
    monkeypatch, tmp_path, launch
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(launch)

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_uses_legacy_default_stream",
    )


def test_hip_source_policy_honors_protected_event_only_marker(monkeypatch, tmp_path):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "#define AKA_BENCHMARK_EVENT_ONLY 1\n"
        "auto stream = at::hip::getCurrentHIPStream();\n"
        "kernel<<<grid, block, 0, stream>>>(output);\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_declares_event_only",
    )


def test_hip_source_policy_rejects_unverified_and_capture_unsafe_code(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    no_launch = tmp_path / "no_launch.hip"
    no_launch.write_text("void wrapper() {}\n")
    unsafe = tmp_path / "unsafe.hip"
    unsafe.write_text(
        "auto stream = at::hip::getCurrentHIPStream();\n"
        "kernel<<<grid, block, 0, stream>>>(output);\n"
        "hipDeviceSynchronize();\n"
    )

    assert helper.hip_source_graph_capture_policy(no_launch) == (
        False,
        "hip_source_launch_stream_unverified",
    )
    assert helper.hip_source_graph_capture_policy(unsafe) == (
        False,
        "hip_source_contains_capture_unsafe_api",
    )


def test_hip_source_policy_ignores_comments_and_disabled_debug_sync(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "// kernel<<<grid, block>>>(output);\n"
        "auto stream = at::hip::getCurrentHIPStream();\n"
        "kernel<<<grid, block, 0, stream>>>(output);\n"
        "#ifdef DEBUG\n"
        "hipDeviceSynchronize();\n"
        "#endif\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (True, None)


def test_hip_source_policy_keeps_active_debug_else_branch(monkeypatch, tmp_path):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "auto stream = at::hip::getCurrentHIPStream();\n"
        "kernel<<<grid, block, 0, stream>>>(output);\n"
        "#ifdef DEBUG\n"
        "// diagnostics disabled\n"
        "#else\n"
        "hipDeviceSynchronize();\n"
        "#endif\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_contains_capture_unsafe_api",
    )


@pytest.mark.parametrize(
    "api_call",
    [
        "hipMemcpy2DAsync(dst, pitch, src, pitch, w, h, kind, 0);",
        "hipMemcpy3DAsync(&params, 0);",
        "hipMemcpyPeerAsync(dst, 0, src, 1, bytes, 0);",
        "hipExtLaunchKernel(fn, grid, block, args, 0, 0, start, stop, 0);",
    ],
)
def test_hip_source_policy_checks_additional_async_and_launch_streams(
    monkeypatch, tmp_path, api_call
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "auto stream = at::hip::getCurrentHIPStream();\n"
        "kernel<<<grid, block, 0, stream>>>(output);\n"
        f"{api_call}\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_uses_legacy_default_stream",
    )


def test_hip_source_policy_accepts_known_async_memory_api_on_current_stream(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "auto stream = at::hip::getCurrentHIPStream();\n"
        "hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream);\n"
        "hipMemsetAsync(dst, 0, bytes, stream);\n"
        "kernel<<<grid, block, 0, stream>>>(output);\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (True, None)


def test_hip_source_policy_fails_closed_on_unknown_async_api(monkeypatch, tmp_path):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "auto stream = at::hip::getCurrentHIPStream();\n"
        "kernel<<<grid, block, 0, stream>>>(output);\n"
        "hipMysteryAsync(dst, src, bytes, stream);\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_launch_stream_unverified",
    )


@pytest.mark.parametrize(
    "memory_call",
    [
        "hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);",
        "hipMemset(dst, 0, bytes);",
        "cudaMemcpy2D(dst, pitch, src, pitch, width, height, kind);",
    ],
)
def test_hip_source_policy_rejects_streamless_memory_api(
    monkeypatch, tmp_path, memory_call
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "auto stream = at::hip::getCurrentHIPStream();\n"
        "kernel<<<grid, block, 0, stream>>>(output);\n"
        f"{memory_call}\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_contains_capture_unsafe_api",
    )


def test_committed_hip2hip_sources_are_graph_capture_compatible(monkeypatch):
    helper = _load_helper(monkeypatch)
    repo = Path(__file__).parents[1]

    gpumode_sources = sorted(
        (repo / "tasks/hip2hip/gpumode").glob("*/hip/*.hip")
    )
    assert len(gpumode_sources) == 44
    failures = {
        str(path.relative_to(repo)): helper.hip_source_graph_capture_policy(path)
        for path in gpumode_sources
        if helper.hip_source_graph_capture_policy(path) != (True, None)
    }

    other_pairs = {
        "assign_score_withk": ("assign_score_withk.cpp", "assign_score_withk_cuda.hip"),
        "ball_query": ("ball_query.cpp", "ball_query_cuda.hip"),
        "furthest_point_sample": ("furthest_point_sample.cpp", "furthest_point_sample_cuda.hip"),
        "knn": ("knn.cpp", "knn_cuda.hip"),
        "points_in_boxes": ("points_in_boxes.cpp", "points_in_boxes_cuda.hip"),
        "roiaware_pool3d": ("roiaware_pool3d.cpp", "roiaware_pool3d_kernel.hip"),
        "roipoint_pool3d": ("roipoint_pool3d.cpp", "roipoint_pool3d_kernel.hip"),
        "three_nn": ("three_nn.cpp", "three_nn_cuda.hip"),
    }
    others_root = repo / "tasks/hip2hip/others"
    for task_name, names in other_pairs.items():
        paths = tuple(others_root / task_name / "src" / name for name in names)
        policy = helper.hip_source_graph_capture_policy(*paths)
        if policy != (True, None):
            failures[f"tasks/hip2hip/others/{task_name}"] = policy

    assert failures == {
        "tasks/hip2hip/gpumode/CrossEntropyLossLabelSmoothing/hip/"
        "hip_12501_CrossEntropyLossLabelSmoothing_ref.hip": (
            False,
            "hip_source_declares_event_only",
        )
    }


def test_hip_source_policy_does_not_trust_unrelated_stream_assignment(
    monkeypatch, tmp_path
):
    helper = _load_helper(monkeypatch)
    source = tmp_path / "kernel.hip"
    source.write_text(
        "void unrelated() {\n"
        "  auto stream = at::hip::getCurrentHIPStream();\n"
        "}\n"
        "void target(hipStream_t other) {\n"
        "  kernel<<<grid, block, 0, stream>>>(output);\n"
        "}\n"
    )

    assert helper.hip_source_graph_capture_policy(source) == (
        False,
        "hip_source_launch_stream_unverified",
    )
