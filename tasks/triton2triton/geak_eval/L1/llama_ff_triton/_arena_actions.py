"""Explicit task actions. Public evaluation dispatch is independent of agents."""
import os
import test_kernel_harness as h


def inputs():
    return {'performance': list(h.ALL_SHAPES), 'original_correctness': list(h.HARNESS_SHAPES)}


def validate():
    return None


def correctness(require):
    original = list(h.HARNESS_SHAPES)
    require(h.run_correctness(original), 'zero', len(original))
    additional = [cfg for cfg in h.ALL_SHAPES if cfg not in original]
    if additional:
        require(h.run_correctness(additional), 'zero', len(additional))


def performance():
    from _arena_checks import checked_benchmark

    benchmark = h.benchmark_cuda_graph_or_events
    h.benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(h, benchmark, fn, **kwargs)
    try:
        return h.run_benchmark(h.ALL_SHAPES, warmup=50, iterations=int(os.environ.get('GEAK_BENCHMARK_ITERATIONS', '200')))
    finally:
        h.benchmark_cuda_graph_or_events = benchmark
