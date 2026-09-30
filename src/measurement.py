"""Measurement identity shared by task parsing, validation and reporting."""
from __future__ import annotations

import math
import statistics

DEVICE_METHODS = frozenset({"cuda_graph", "cuda_event_fallback"})
SERVING_METHOD = "serving_wall_clock"


def measurement_kind(config: dict) -> str:
    return config.get("evaluation", {}).get("measurement", {}).get("kind", "kernel")


def validate_serving_case(row: dict) -> None:
    metrics, params = row.get("metrics", {}), row.get("params", {})
    for key in ("duration_s", "output_tokens_per_s"):
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Serving measurement requires positive finite {key}")
    if "p99_tpot_ms" in metrics:
        latency = metrics["p99_tpot_ms"]
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
            raise ValueError("Serving tail latency must be positive and finite")
    for key in ("completed_requests", "output_tokens"):
        if type(metrics.get(key)) is not int or metrics[key] <= 0:
            raise ValueError(f"Serving measurement requires positive integer {key}")
    requests, tokens = params.get("num_requests"), params.get("output_tokens_per_request")
    if type(requests) is not int or requests <= 0 or type(tokens) is not int or tokens <= 0:
        raise ValueError("Serving manifest must fix request and output token counts")
    if metrics["completed_requests"] != requests or metrics["output_tokens"] != requests * tokens:
        raise ValueError("Serving measurement changed work or omitted requests")
    if "input_tokens" in params:
        expected = params["input_tokens"]
        if type(expected) is not int or expected <= 0 or metrics.get("input_tokens") != requests * expected:
            raise ValueError("Serving measurement changed input token counts")
    if not math.isclose(row["execution_time_ms"], metrics["duration_s"] * 1000, rel_tol=1e-8):
        raise ValueError("Serving duration contradicts execution_time_ms")
    if not math.isclose(metrics["output_tokens_per_s"], metrics["output_tokens"] / metrics["duration_s"], rel_tol=1e-8):
        raise ValueError("Serving throughput contradicts token count and elapsed time")


def paired_throughput(pairs: list[tuple[list, list]]) -> dict:
    """Median of paired ratios per case, geometric mean across fixed cases."""
    samples: dict[str, list[float]] = {}
    identities = {}
    for baseline, candidate in pairs:
        b = {case.test_case_id: case for case in baseline}
        c = {case.test_case_id: case for case in candidate}
        if not b or b.keys() != c.keys() or len(b) != len(baseline) or len(c) != len(candidate):
            raise ValueError("Serving pair has missing or duplicate cases")
        for key in b:
            bm, cm = b[key].metadata, c[key].metadata
            if bm.get("params") != cm.get("params") or b[key].shape != c[key].shape or bm.get("dtype") != cm.get("dtype"):
                raise ValueError("Serving pair changed workload identity")
            if any(m.get("benchmark_method") != SERVING_METHOD for m in (bm, cm)):
                raise ValueError("Serving pair changed measurement method")
            runtime = bm.get("runtime_fingerprint")
            if not runtime or runtime != cm.get("runtime_fingerprint"):
                raise ValueError("Serving pair changed runtime identity")
            identity = (runtime, bm.get("params"), b[key].shape, bm.get("dtype"))
            if key in identities and identities[key] != identity:
                raise ValueError("Serving workload or runtime changed between pairs")
            identities[key] = identity
            ratio = cm["metrics"]["output_tokens_per_s"] / bm["metrics"]["output_tokens_per_s"]
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError("Invalid throughput ratio")
            samples.setdefault(key, []).append(ratio)
    if not samples or any(len(v) != len(pairs) for v in samples.values()):
        raise ValueError("Serving cases differ across pairs")
    medians = {k: statistics.median(v) for k, v in samples.items()}
    return {"speedup_ratio": math.exp(sum(math.log(v) for v in medians.values()) / len(medians)),
            "case_ratios": medians, "paired_ratios": samples, "pairs": len(pairs),
            "metric": "output_tokens_per_s", "unit": "tokens/s", "direction": "higher",
            "gain_observed_in_all_pairs": all(v > 1 for values in samples.values() for v in values)}
