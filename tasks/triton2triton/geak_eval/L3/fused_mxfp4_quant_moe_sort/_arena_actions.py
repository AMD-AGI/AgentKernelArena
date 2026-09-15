"""Explicit task actions. Public evaluation dispatch is independent of agents."""
import os
import test_kernel_harness as h
from _arena_checks import checked_benchmark, checked_correctness


def inputs():
    return {'performance': list(h.ALL_CONFIGS), 'original_correctness': list(h.ALL_CONFIGS)}


def validate():
    if not h._is_fp4_avail():
        raise RuntimeError('Required MXFP4 hardware is unavailable; this is not a passing skip')


def correctness(require):
    original_indices = list(range(len(h.ALL_CONFIGS)))
    with checked_correctness(h):
        require(h.run_correctness(original_indices), 'bool', len(original_indices))
    additional = [i for i in range(len(h.ALL_CONFIGS)) if i not in original_indices]
    if additional:
        require(h.run_correctness(additional), 'bool', len(additional))


def performance():
    benchmark = h.benchmark_cuda_graph_or_events
    h.benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(h, benchmark, fn, **kwargs)
    try:
        return h.run_benchmark(list(range(len(h.ALL_CONFIGS))))
    finally:
        h.benchmark_cuda_graph_or_events = benchmark
