"""Compatibility adapters for vLLM task runners.

This file is injected between AKA-GENERATED markers.  The implementation lives
in the sibling materialized ``_aka_benchmark.py`` module.
"""

import sys as _aka_sys
from pathlib import Path as _AkaPath


# Task-provided forge drivers are copied to the workspace root and load this
# runner by file path.  Python does not add a file-loaded module's directory to
# sys.path, so make the sibling materialized helper discoverable explicitly.
_AKA_HELPER_DIR = str(_AkaPath(__file__).resolve().parent)
if _AKA_HELPER_DIR not in _aka_sys.path:
    _aka_sys.path.insert(0, _AKA_HELPER_DIR)


def _measure_cuda_event_fallback(fn, repetition, prepare_fn=None):
    try:
        from _aka_benchmark import benchmark_cuda_event_samples
    except ModuleNotFoundError:  # Direct source-tree unit tests.
        from src.tools.perf.aka_benchmark import benchmark_cuda_event_samples

    return benchmark_cuda_event_samples(
        fn, repetition=repetition, prepare_fn=prepare_fn
    )


class _TimedRun:
    """Handle on the exact invocation a benchmark measured.

    Timing and correctness are otherwise separate invocations, so a kernel can
    tell them apart and do less work in the one that is scored. Passing this
    collector to the benchmark makes the scored invocation itself observable:
    ``outputs`` aliases the buffers the timed unit last wrote, and ``rerun``
    executes that same unit again.

    Under CUDA-graph timing the buffers are captured once and every replay
    writes to those same addresses, so ``outputs`` keeps tracking replays. Under
    event-timing fallback the measured outputs cannot be observed reliably, so a
    benchmark that requests this collector fails closed instead of validating a
    separate post-timing invocation.
    """

    def __init__(self):
        self._rerun = None
        self.outputs = None

    def _bind(self, rerun, outputs=None):
        self._rerun = rerun
        self.outputs = outputs

    @property
    def bound(self):
        return self._rerun is not None

    def rerun(self):
        if self._rerun is None:
            raise RuntimeError(
                "timed run was never bound; the benchmark did not reach a "
                "measurement path"
            )
        self.outputs = self._rerun()
        return self.outputs


def _benchmark_cuda_graph_or_events(
    fn,
    warmup=10,
    repetition=100,
    target_ms=1.0,
    n_retries=5,
    estimate_reps=5,
    max_graph_repeats=1000,
    use_cuda_graph=True,
    fallback_reason=None,
    timed_run=None,
    prepare_fn=None,
):
    try:
        from _aka_benchmark import benchmark_cuda_graph_or_events
    except ModuleNotFoundError:  # Direct source-tree unit tests.
        from src.tools.perf.aka_benchmark import benchmark_cuda_graph_or_events

    return benchmark_cuda_graph_or_events(
        fn,
        warmup=warmup,
        repetition=repetition,
        target_ms=target_ms,
        n_retries=n_retries,
        estimate_reps=estimate_reps,
        max_graph_repeats=max_graph_repeats,
        use_cuda_graph=use_cuda_graph,
        fallback_reason=fallback_reason,
        prepare_fn=prepare_fn,
        timed_run=timed_run,
    )
