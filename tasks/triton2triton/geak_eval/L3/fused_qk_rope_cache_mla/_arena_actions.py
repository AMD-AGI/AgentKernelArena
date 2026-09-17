"""Explicit task actions. Public evaluation dispatch is independent of agents."""
import os
import test_kernel_harness as h
from _arena_checks import install

install(h)


def inputs():
    return {'performance': list(h.ALL_CONFIGS), 'original_correctness': list(h.ALL_CONFIGS)}


def validate():
    return None


def correctness(require):
    for cfg in h.ALL_CONFIGS:
        require(h._check_correctness_single(cfg), 'bool', 1)


def performance():
    records = []
    for i, cfg in enumerate(h.ALL_CONFIGS):
        ms, metadata = h._benchmark_single(cfg)
        records.append((i, cfg, ms, metadata))
    h._write_performance_report(records)
