"""ROCmBench compatibility layer over the canonical graph-first benchmark."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from _aka_benchmark import benchmark_cuda_graph_or_events_samples


@dataclass
class BenchConfig:
    warm_up: int = 10
    repetition: int = 100


def do_bench_config(warm_up: int = 10, repetition: int = 100) -> BenchConfig:
    """Create a benchmark configuration object compatible with existing tasks."""

    return BenchConfig(warm_up=max(0, int(warm_up)), repetition=max(1, int(repetition)))


_BENCHMARK_RESULTS: list[dict[str, Any]] = []


def _median(values: list[float]) -> float:
    values_sorted = sorted(values)
    return values_sorted[len(values_sorted) // 2]


def _measure_times(
    callable_fn: Callable[[], Any],
    config: BenchConfig,
    target_ms: float = 1.0,
    n_retries: int = 5,
    estimate_reps: int = 5,
    max_graph_repeats: int = 1000,
) -> tuple[list[float], dict[str, Any]]:
    """Return canonical per-call device samples and benchmark metadata."""

    return benchmark_cuda_graph_or_events_samples(
        callable_fn,
        warmup=config.warm_up,
        repetition=config.repetition,
        target_ms=target_ms,
        n_retries=n_retries,
        estimate_reps=estimate_reps,
        max_graph_repeats=max_graph_repeats,
    )


def _compute_timing_stats(times_ms: list[float], config: BenchConfig) -> dict[str, Any]:
    """Compute mean, median, p90, min, and max from per-call samples."""

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
    """Simple benchmark helper used by ROCmBench pytest performance tests."""

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

        if baseline_callable is not None:
            baseline_times, baseline_metadata = _measure_times(baseline_callable, self.config)
            baseline_stats = _compute_timing_stats(baseline_times, self.config)
            result["baseline_timing_ms"] = baseline_stats
            for key, value in baseline_metadata.items():
                result[f"baseline_{key}"] = value
            baseline_method = baseline_metadata.get("benchmark_method")
            optimized_method = benchmark_metadata.get("benchmark_method")
            if baseline_method == optimized_method and mean_ms > 0:
                result["speedup_ratio"] = baseline_stats["mean"] / mean_ms
            else:
                result["speedup_error"] = (
                    "benchmark method mismatch: "
                    f"baseline={baseline_method!r}, optimized={optimized_method!r}"
                )

        _BENCHMARK_RESULTS.append(result)
        return result


def save_all_benchmark_results(output_directory: str) -> None:
    """Persist collected benchmark entries to a single JSON file."""

    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(_BENCHMARK_RESULTS, indent=2, sort_keys=True), encoding="utf-8")
