"""Measurement identity shared by task parsing, validation and reporting."""
from __future__ import annotations

import math
import statistics

DEVICE_METHODS = frozenset({"cuda_graph", "cuda_event_fallback", "cuda_graph_cold_weights",
                            "cuda_graph_cold_cache"})
SERVING_METHOD = "serving_wall_clock"

COLD_CACHE_POLICY = {
    "benchmark_cache_policy": "read_only_eviction_before_each_invocation",
    "benchmark_cache_flush_in_timing": False,
    "benchmark_effective_repeats": 1,
    "benchmark_metric": "gpu_event_graph_span",
}


def cold_cache_identity(metadata: dict) -> tuple:
    """Validated cache/timing identity; method name alone cannot establish fairness."""
    for key, expected in COLD_CACHE_POLICY.items():
        if type(metadata.get(key)) is not type(expected) or metadata[key] != expected:
            raise ValueError(f"cold-cache timing requires {key}={expected!r}")
    nbytes = metadata.get("benchmark_cache_flush_bytes")
    if type(nbytes) is not int or nbytes <= 0 or nbytes % 16384:
        raise ValueError("cold-cache timing requires a positive eviction size divisible by 16384")
    return tuple(metadata[key] for key in COLD_CACHE_POLICY) + (nbytes,)

# Serving runtime limits a task may tighten or relax in its runtime lock. The
# adapter is materialized beside the task, so this table is the single source
# for both the host executor and the in-container client.
LIMIT_DEFAULTS = {
    "server_startup_timeout_s": 600,
    "client_timeout_s": 500,
    "max_candidate_source_bytes": 16 * 1024 * 1024,
    "idle_vram_limit_mib": 2048,
    "minimum_measurement_s": 30,
}


def lock_limits(lock: dict) -> dict:
    """Validated runtime limits with defaults; every value is a positive integer."""
    declared = lock.get("limits", {})
    if not isinstance(declared, dict) or set(declared) - set(LIMIT_DEFAULTS):
        raise ValueError(f"Runtime lock limits accept only {sorted(LIMIT_DEFAULTS)}")
    for key, value in declared.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"Runtime lock limit {key} must be a positive integer")
    return {**LIMIT_DEFAULTS, **declared}


def measurement_kind(config: dict) -> str:
    return config.get("evaluation", {}).get("measurement", {}).get("kind", "kernel")


def validate_pair_count(value: int) -> int:
    if type(value) is not int or not 3 <= value <= 100:
        raise ValueError("final_evaluation_pairs must be an integer from 3 to 100")
    return value


def serving_pair_count(config: dict, override: int | None = None) -> int:
    """A run may increase, but never reduce, the task's measurement count."""
    declared = validate_pair_count(config["evaluation"]["measurement"]["pairs"])
    count = declared if override is None else validate_pair_count(override)
    if count < declared:
        raise ValueError("final_evaluation_pairs cannot reduce the task's declared pairs")
    return count


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
    requests = params.get("num_requests")
    if type(requests) is not int or requests <= 0:
        raise ValueError("Serving manifest must fix request and output token counts")

    def expected_tokens(scalar, sequence):
        if sequence in params:
            values = params[sequence]
            if scalar in params or not isinstance(values, list) or len(values) != requests or any(
                type(v) is not int or v <= 0 for v in values
            ):
                raise ValueError("Serving manifest requires unambiguous positive per-request token counts")
            if metrics.get(sequence) != values:
                raise ValueError("Serving measurement changed per-request token counts")
            return sum(values)
        value = params.get(scalar)
        if type(value) is not int or value <= 0:
            raise ValueError("Serving manifest must fix request and output token counts")
        return requests * value

    output_tokens = expected_tokens("output_tokens_per_request", "output_lens")
    if metrics["completed_requests"] != requests or metrics["output_tokens"] != output_tokens:
        raise ValueError("Serving measurement changed work or omitted requests")
    if "input_tokens" in params or "input_lens" in params:
        expected = expected_tokens("input_tokens", "input_lens")
        if metrics.get("input_tokens") != expected:
            raise ValueError("Serving measurement changed input token counts")
    if not math.isclose(row["execution_time_ms"], metrics["duration_s"] * 1000, rel_tol=1e-8):
        raise ValueError("Serving duration contradicts execution_time_ms")
    if not math.isclose(metrics["output_tokens_per_s"], metrics["output_tokens"] / metrics["duration_s"], rel_tol=1e-8):
        raise ValueError("Serving throughput contradicts token count and elapsed time")


def paired_throughput(pairs: list[tuple[list, list]], *, minimum_speedup_ratio: float = 1.0) -> dict:
    """Median of paired ratios per case, geometric mean across fixed cases."""
    if (type(minimum_speedup_ratio) not in (int, float)
            or not math.isfinite(minimum_speedup_ratio) or minimum_speedup_ratio < 1):
        raise ValueError("minimum_speedup_ratio must be finite and at least 1")
    samples: dict[str, list[float]] = {}
    identities = {}
    kernel_identities = {}
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
            kernels = (bm.get("kernel_sha256"), cm.get("kernel_sha256"))
            if any(not isinstance(value, str) or len(value) != 64 or
                   any(char not in "0123456789abcdef" for char in value) for value in kernels):
                raise ValueError("Serving pair requires baseline and candidate kernel hashes")
            if key in kernel_identities and kernel_identities[key] != kernels:
                raise ValueError("Serving kernel identity changed between pairs")
            kernel_identities[key] = kernels
            ratio = cm["metrics"]["output_tokens_per_s"] / bm["metrics"]["output_tokens_per_s"]
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError("Invalid throughput ratio")
            samples.setdefault(key, []).append(ratio)
    if not samples or any(len(v) != len(pairs) for v in samples.values()):
        raise ValueError("Serving cases differ across pairs")
    medians = {k: statistics.median(v) for k, v in samples.items()}
    changed = {k: baseline != candidate for k, (baseline, candidate) in kernel_identities.items()}
    # An unchanged implementation cannot earn kernel-optimization credit from
    # runtime noise. Preserve every raw measurement, including regressions.
    gain = {k: changed[k] and value > 1 and value >= minimum_speedup_ratio for k, value in medians.items()}
    scored = {k: value if gain[k] else min(value, 1.0) for k, value in medians.items()}
    return {"speedup_ratio": math.exp(sum(math.log(v) for v in scored.values()) / len(scored)),
            "raw_speedup_ratio": math.exp(sum(math.log(v) for v in medians.values()) / len(medians)),
            "scored_case_ratios": scored, "kernel_changed": changed,
            "minimum_speedup_ratio": minimum_speedup_ratio, "gain_threshold_met": gain,
            "significance": "not_assessed",
            "case_ratios": medians, "paired_ratios": samples, "pairs": len(pairs),
            "metric": "output_tokens_per_s", "unit": "tokens/s", "direction": "higher",
            "gain_observed_in_all_pairs": all(v > 1 for values in samples.values() for v in values),
            "uncertainty": _uncertainty(pairs, samples)}


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def _uncertainty(pairs: list[tuple[list, list]], samples: dict[str, list[float]]) -> dict:
    """Spread of the raw samples, so a median ratio is never read as a proven gain."""
    result = {}
    for key, ratios in samples.items():
        throughput = {"baseline": [], "candidate": []}
        for baseline, candidate in pairs:
            for role, rows in (("baseline", baseline), ("candidate", candidate)):
                case = next(c for c in rows if c.test_case_id == key)
                throughput[role].append(case.metadata["metrics"]["output_tokens_per_s"])
        if len(ratios) == 1:
            result[key] = {
                "ratio_min": ratios[0], "ratio_max": ratios[0],
                "baseline_throughput_cv": None, "candidate_throughput_cv": None,
                "median_within_noise": None,
                "status": "not_estimable_from_one_pair",
            }
            continue
        baseline_cv = _coefficient_of_variation(throughput["baseline"])
        candidate_cv = _coefficient_of_variation(throughput["candidate"])
        result[key] = {
            "ratio_min": min(ratios), "ratio_max": max(ratios),
            "baseline_throughput_cv": baseline_cv, "candidate_throughput_cv": candidate_cv,
            # A median ratio inside the repeat-to-repeat spread of either side
            # is indistinguishable from noise with this many pairs.
            "median_within_noise": abs(statistics.median(ratios) - 1) <= max(baseline_cv, candidate_cv),
        }
    return result
