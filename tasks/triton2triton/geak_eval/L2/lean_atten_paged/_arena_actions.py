"""Explicit task actions. Public evaluation dispatch is independent of agents."""
import os
import test_kernel_harness as h


def inputs():
    return {'performance': list(h.ALL_SHAPES), 'original_correctness': list(h.CORRECTNESS_CONFIGS)}


def validate():
    return None


def correctness(require):
    original = list(h.CORRECTNESS_CONFIGS)
    require(h.run_correctness(original), 'dict', len(original))
    additional = [cfg for cfg in h.ALL_SHAPES if cfg not in original]
    if additional:
        require(h.run_correctness(additional), 'dict', len(additional))


def performance():
    return h.run_benchmark(h.ALL_SHAPES, warmup=50, iters=int(os.environ.get('GEAK_BENCHMARK_ITERATIONS', '200')))
