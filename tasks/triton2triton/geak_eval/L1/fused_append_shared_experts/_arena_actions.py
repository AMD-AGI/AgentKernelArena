"""Explicit task actions. Public evaluation dispatch is independent of agents."""
import os
import test_kernel_harness as h


def inputs():
    return {'performance': list(h.ALL_CONFIGS), 'original_correctness': list(h.ALL_CONFIGS)}


def validate():
    return None


def correctness(require):
    original_indices = list(range(len(h.ALL_CONFIGS)))
    require(h.run_correctness(original_indices), 'none', len(original_indices))
    additional = [i for i in range(len(h.ALL_CONFIGS)) if i not in original_indices]
    if additional:
        require(h.run_correctness(additional), 'none', len(additional))


def performance():
    from _arena_checks import checked_benchmark

    benchmark = h.benchmark_cuda_graph_or_events
    h.benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(h, benchmark, fn, **kwargs)
    try:
        return h.run_benchmark(list(range(len(h.ALL_CONFIGS))))
    finally:
        h.benchmark_cuda_graph_or_events = benchmark
