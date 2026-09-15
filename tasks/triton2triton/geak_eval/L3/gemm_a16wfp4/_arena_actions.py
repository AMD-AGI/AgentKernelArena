"""Explicit task actions. Public evaluation dispatch is independent of agents."""
import os
import test_kernel_harness as h
from _arena_checks import checked_benchmark, checked_correctness


def inputs():
    return {'performance': list(h.ALL_SHAPES), 'original_correctness': list(h.HARNESS_SHAPES)}


def validate():
    if not h.is_fp4_avail():
        raise RuntimeError('Required MXFP4 hardware is unavailable; this is not a passing skip')


def correctness(require):
    original = list(h.HARNESS_SHAPES)
    additional = [cfg for cfg in h.ALL_SHAPES if cfg not in original]
    with checked_correctness(h):
        require(h.run_correctness(original), 'dict', len(original))
        if additional:
            require(h.run_correctness(additional), 'dict', len(additional))


def performance():
    benchmark = h.benchmark_cuda_graph_or_events
    h.benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(h, benchmark, fn, **kwargs)
    try:
        return h.run_benchmark(h.ALL_SHAPES, warmup=50, iters=int(os.environ.get('GEAK_BENCHMARK_ITERATIONS', '200')))
    finally:
        h.benchmark_cuda_graph_or_events = benchmark
