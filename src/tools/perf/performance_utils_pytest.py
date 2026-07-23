import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass
class BenchConfig:
    warm_up: int = 10
    repetition: int = 100


def do_bench_config(warm_up: int = 10, repetition: int = 100) -> BenchConfig:
    """Create a benchmark configuration object compatible with existing task code."""
    return BenchConfig(warm_up=max(0, int(warm_up)), repetition=max(1, int(repetition)))


_BENCHMARK_RESULTS: list[dict[str, Any]] = []


def _sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _median(values: list[float]) -> float:
    values_sorted = sorted(values)
    return values_sorted[len(values_sorted) // 2]


def _measure_cuda_events(callable_fn: Callable[[], Any], repetition: int) -> list[float]:
    """Measure eager callable executions with CUDA events, falling back to CPU time without CUDA."""
    repetition = max(1, int(repetition))
    if not torch.cuda.is_available():
        times_ms: list[float] = []
        for _ in range(repetition):
            start = time.perf_counter()
            callable_fn()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)
        return times_ms

    times_ms = []
    for _ in range(repetition):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        callable_fn()
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))
    return times_ms


def _measure_times(
    callable_fn: Callable[[], Any],
    config: BenchConfig,
    target_ms: float = 1.0,
    n_retries: int = 5,
    estimate_reps: int = 5,
    max_graph_repeats: int = 1000,
) -> tuple[list[float], dict[str, Any]]:
    """Run warmup + measured iterations and return per-call times in ms plus metadata."""
    # A captured graph whose measured per-iteration time falls below this floor is
    # treated as empty (it recorded no device work) and rejected in favour of
    # per-launch event timing, so a graph that silently captured nothing is never
    # reported as a fabricated ~0 ms cuda_graph result. Real kernels measure far
    # above this floor; an empty-graph replay measures ~1e-5 ms.
    empty_graph_floor_ms = 1e-4

    for _ in range(config.warm_up):
        callable_fn()
    _sync_if_needed()

    max_graph_repeats = max(1, int(max_graph_repeats))
    metadata: dict[str, Any] = {
        "benchmark_target_ms": float(target_ms),
        "benchmark_samples": int(config.repetition),
        "benchmark_max_repeats": int(max_graph_repeats),
    }

    if not torch.cuda.is_available():
        metadata.update({
            "benchmark_method": "cpu_timer_fallback",
            "benchmark_effective_repeats": int(config.repetition),
            "benchmark_fallback_reason": "cuda_unavailable",
        })
        return _measure_cuda_events(callable_fn, config.repetition), metadata

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            estimate_reps = max(1, int(estimate_reps))
            estimate_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(estimate_graph):
                for _ in range(estimate_reps):
                    callable_fn()
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            estimate_graph.replay()
            end_event.record(stream)
            torch.cuda.synchronize()

            estimate_ms = start_event.elapsed_time(end_event) / estimate_reps
            if estimate_ms == 0:
                n_repeat = max_graph_repeats
            else:
                n_repeat = min(max_graph_repeats, max(1, int(float(target_ms) / estimate_ms)))

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(n_repeat):
                    callable_fn()
            torch.cuda.synchronize()

            retry_times: list[float] = []
            for _ in range(max(1, int(config.repetition))):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(stream)
                graph.replay()
                end_event.record(stream)
                torch.cuda.synchronize()
                retry_times.append(start_event.elapsed_time(end_event) / n_repeat)

        graph_mean = sum(retry_times) / len(retry_times)
        if graph_mean < empty_graph_floor_ms:
            metadata.update({
                "benchmark_method": "cuda_event_fallback",
                "benchmark_effective_repeats": int(config.repetition),
                "benchmark_fallback_reason": "empty_cuda_graph_capture",
            })
            return _measure_cuda_events(callable_fn, config.repetition), metadata

        metadata.update({
            "benchmark_method": "cuda_graph",
            "benchmark_effective_repeats": int(n_repeat),
        })
        return retry_times, metadata
    except Exception as exc:
        # Isolate the aborted capture before re-measuring so the fallback timing is
        # not polluted by the failed attempt (a mid-capture failure can leave the
        # first few launches abnormally slow).
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        for _ in range(min(3, max(1, int(config.warm_up)))):
            callable_fn()
        _sync_if_needed()
        metadata.update({
            "benchmark_method": "cuda_event_fallback",
            "benchmark_effective_repeats": int(config.repetition),
            "benchmark_fallback_reason": f"cuda_graph_failed: {type(exc).__name__}: {str(exc)[:160]}",
        })
        return _measure_cuda_events(callable_fn, config.repetition), metadata


def _compute_timing_stats(times_ms: list[float], config: BenchConfig) -> dict[str, Any]:
    """Compute mean, median, p90, min, max from a list of times."""
    times_sorted = sorted(times_ms)
    n = len(times_sorted)
    return {
        "mean": sum(times_sorted) / n,
        "median": _median(times_sorted),
        "p90": times_sorted[min(n - 1, int(round(0.9 * (n - 1))))],
        "min": times_sorted[0],
        "max": times_sorted[-1],
        "repetition": config.repetition,
        "warm_up": config.warm_up,
    }


class PytestBenchmarker:
    """Simple benchmark helper used by rocmbench pytest performance tests."""

    def __init__(self, op_callable: Callable[[], Any], op_name: str, config: BenchConfig) -> None:
        self.op_callable = op_callable
        self.op_name = op_name
        self.config = config

    def run_benchmark(
        self,
        current_params_dict: dict[str, Any],
        gbps_calculator: Callable[[dict[str, Any], float], float] | None = None,
        tflops_calculator: Callable[[dict[str, Any], float], float] | None = None,
        baseline_callable: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        # Measure the main (optimized/triton) operation.
        times_ms, benchmark_metadata = _measure_times(self.op_callable, self.config)
        timing_stats = _compute_timing_stats(times_ms, self.config)
        mean_ms = timing_stats["mean"]

        result: dict[str, Any] = {
            "op_name": self.op_name,
            "params": current_params_dict,
            "timing_ms": timing_stats,
            **benchmark_metadata,
        }

        if gbps_calculator is not None:
            try:
                result["gbps"] = float(gbps_calculator(current_params_dict, mean_ms))
            except Exception as exc:
                result["gbps_error"] = str(exc)
        if tflops_calculator is not None:
            try:
                result["tflops"] = float(tflops_calculator(current_params_dict, mean_ms))
            except Exception as exc:
                result["tflops_error"] = str(exc)

        # Measure baseline (e.g. PyTorch reference) if provided.
        if baseline_callable is not None:
            baseline_times, baseline_metadata = _measure_times(baseline_callable, self.config)
            baseline_stats = _compute_timing_stats(baseline_times, self.config)
            result["baseline_timing_ms"] = baseline_stats
            for key, value in baseline_metadata.items():
                result[f"baseline_{key}"] = value
            baseline_mean = baseline_stats["mean"]
            if mean_ms > 0:
                result["speedup_ratio"] = baseline_mean / mean_ms
            else:
                result["speedup_ratio"] = 1.0

        _BENCHMARK_RESULTS.append(result)
        return result


def save_all_benchmark_results(output_directory: str) -> None:
    """Persist collected benchmark entries to a single JSON file."""
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(_BENCHMARK_RESULTS, indent=2, sort_keys=True), encoding="utf-8")
