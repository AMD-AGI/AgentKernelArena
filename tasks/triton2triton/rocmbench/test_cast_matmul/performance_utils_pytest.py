import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass
class BenchConfig:
    warm_up: int = 25
    repetition: int = 100


def do_bench_config(warm_up: int = 25, repetition: int = 100) -> BenchConfig:
    """Create a benchmark configuration object compatible with existing task code."""
    return BenchConfig(warm_up=max(0, int(warm_up)), repetition=max(1, int(repetition)))


_BENCHMARK_RESULTS: list[dict[str, Any]] = []


def _sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
    ) -> dict[str, Any]:
        # Warmup first to avoid measuring one-time initialization effects.
        for _ in range(self.config.warm_up):
            self.op_callable()
        _sync_if_needed()

        times_ms: list[float] = []
        for _ in range(self.config.repetition):
            _sync_if_needed()
            start = time.perf_counter()
            self.op_callable()
            _sync_if_needed()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

        times_ms_sorted = sorted(times_ms)
        n = len(times_ms_sorted)
        median_ms = times_ms_sorted[n // 2]
        p90_ms = times_ms_sorted[min(n - 1, int(round(0.9 * (n - 1))))]
        mean_ms = sum(times_ms_sorted) / n

        result: dict[str, Any] = {
            "op_name": self.op_name,
            "params": current_params_dict,
            "timing_ms": {
                "mean": mean_ms,
                "median": median_ms,
                "p90": p90_ms,
                "min": times_ms_sorted[0],
                "max": times_ms_sorted[-1],
                "repetition": self.config.repetition,
                "warm_up": self.config.warm_up,
            },
        }

        if gbps_calculator is not None:
            try:
                result["gbps"] = float(gbps_calculator(current_params_dict, median_ms))
            except Exception as exc:
                result["gbps_error"] = str(exc)
        if tflops_calculator is not None:
            try:
                result["tflops"] = float(tflops_calculator(current_params_dict, median_ms))
            except Exception as exc:
                result["tflops_error"] = str(exc)

        _BENCHMARK_RESULTS.append(result)
        return result


def save_all_benchmark_results(output_directory: str) -> None:
    """Persist collected benchmark entries to a single JSON file."""
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(_BENCHMARK_RESULTS, indent=2, sort_keys=True), encoding="utf-8")

