"""Explicit task actions. Public evaluation dispatch is independent of agents."""
import os
import test_kernel_harness as h


def inputs():
    return {'performance': list(h.ALL_CONFIGS), 'original_correctness': list(h.ALL_CONFIGS)}


def validate():
    if not h._is_fp4_avail():
        raise RuntimeError('Required MXFP4 hardware is unavailable; this is not a passing skip')


def correctness(require):
    original_indices = list(range(len(h.ALL_CONFIGS)))
    require(h.do_correctness(original_indices), 'zero', len(original_indices))
    additional = [i for i in range(len(h.ALL_CONFIGS)) if i not in original_indices]
    if additional:
        require(h.do_correctness(additional), 'zero', len(additional))


def performance():
    return h.do_benchmark(list(range(len(h.ALL_CONFIGS))))
