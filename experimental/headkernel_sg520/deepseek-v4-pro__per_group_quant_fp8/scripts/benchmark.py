"""Paired native graph timings; unmodified-source experiment, not a speedup claim."""

import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from native import baseline, candidate
from oracle import compare, generate, invoke, load_cases, reference


WARMUP = 10
ITERATIONS = 100


def main():
    output = ROOT / "build/performance_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    manifest, cases = load_cases(ROOT / "cases.json")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("duplicate workload case IDs")
    if any(type(c["trace_call_count"]) is not int or c["trace_call_count"] <= 0 for c in cases):
        raise ValueError("every case requires a positive observed trace weight")

    import torch
    from aiter.jit.utils.chip_info import get_gfx
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        raise RuntimeError("the observed gfx950 runtime is required")
    functions = {"production_native": baseline(), "candidate_native": candidate()}
    paired = []
    for case in cases:
        x = generate(case, 0, "cuda")
        before = x.clone()
        expected = reference(x)
        graphs, results = {}, {}
        for leg, function in functions.items():
            warm_stream = torch.cuda.Stream()
            warm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm_stream):
                for _ in range(WARMUP):
                    invoke(function, x)
            torch.cuda.current_stream().wait_stream(warm_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = invoke(function, x)
            graph.replay()
            torch.cuda.synchronize()
            compare(result, expected, x)
            graphs[leg], results[leg] = graph, result

        events = {leg: [] for leg in functions}
        for iteration in range(ITERATIONS):
            order = tuple(functions) if iteration % 2 == 0 else tuple(reversed(functions))
            for leg in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[leg].replay()
                end.record()
                events[leg].append((start, end))
        torch.cuda.synchronize()
        if not torch.equal(x, before):
            raise AssertionError("graph replay modified the input")
        times = {}
        for leg in functions:
            compare(results[leg], expected, x)
            samples = [a.elapsed_time(b) for a, b in events[leg]]
            if len(samples) != ITERATIONS or any(not math.isfinite(v) or v <= 0 for v in samples):
                raise ValueError("invalid or incomplete graph timing samples")
            times[leg] = {"mean_ms": sum(samples) / ITERATIONS,
                          "min_ms": min(samples), "max_ms": max(samples),
                          "samples_ms": samples}
        paired.append({"case_id": case["case_id"], "shape": case["shape"],
                       "trace_call_count": case["trace_call_count"],
                       "graph_correctness": True, "timings": times})
        del graphs, results, x, before, expected

    if len(paired) != len(cases):
        raise ValueError("incomplete workload coverage")
    weight_sum = sum(row["trace_call_count"] for row in paired)
    report = {
        "status": "ok", "experiment": "unmodified-source native comparison",
        "task_validator_status": "pending", "speedup_claim": None,
        "source_run": manifest["source_run"], "count_scope": manifest["count_scope"],
        "benchmark_method": "cuda_graph", "warmup_iterations": WARMUP,
        "benchmark_iterations": ITERATIONS, "paired_order": "alternating each iteration",
        "separate_baseline_candidate_graphs": True, "weight_sum_per_rank": weight_sum,
        "weighted_mean_ms": {leg: sum(row["trace_call_count"] * row["timings"][leg]["mean_ms"] for row in paired) / weight_sum for leg in functions},
        "paired_cases": paired,
        "test_cases": [{"test_case_id": row["case_id"],
                        "execution_time_ms": row["timings"]["candidate_native"]["mean_ms"],
                        "shape": row["shape"], "params": {"trace_call_count": row["trace_call_count"]},
                        "metadata": {"benchmark_method": "cuda_graph"}}
                       for row in paired],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "ok", "case_count": len(paired), "weight_sum_per_rank": weight_sum,
                      "weighted_mean_ms": report["weighted_mean_ms"], "speedup_claim": None,
                      "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
