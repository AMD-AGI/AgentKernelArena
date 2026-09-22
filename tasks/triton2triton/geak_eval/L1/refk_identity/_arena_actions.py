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
    return h.run_benchmark(list(range(len(h.ALL_CONFIGS))), warmup=10, iters=int(os.environ.get('GEAK_BENCHMARK_ITERATIONS', '100')))
