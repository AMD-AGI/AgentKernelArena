"""Compatibility adapters for vLLM task runners.

This file is injected between AKA-GENERATED markers.  The implementation lives
in the sibling materialized ``_aka_benchmark.py`` module.
"""


def _measure_cuda_event_fallback(fn, repetition, prepare_fn=None):
    return _aka_benchmark_module().benchmark_cuda_event_samples(
        fn, repetition=repetition, prepare_fn=prepare_fn
    )


def _aka_benchmark_module():
    """Resolve the materialized helper regardless of how this file was imported.

    ``import _aka_benchmark`` only works when the helper's directory happens to
    be on ``sys.path``, which is true when the process entry point sits beside
    it -- Arena's own evaluation runs ``scripts/task_runner.py`` directly, so
    ``sys.path[0]`` is ``scripts/`` and the bare import resolves.

    KernelForge does not run it that way.  The forge launcher copies a
    task-shipped ``scripts/forge_driver.py`` to the workspace ROOT, and that
    driver loads ``scripts/task_runner.py`` by path via
    ``spec_from_file_location``, which deliberately does not touch ``sys.path``.
    So ``sys.path[0]`` is the workspace root, where no ``_aka_benchmark.py``
    exists: the materializer only copies the helper beside files whose text
    mentions it, and the root driver's text does not.  The old
    ``src.tools.perf`` fallback does not save it either, because Arena's repo
    root is not importable from the forge subprocess.  The bare import plus that
    fallback therefore both raised, the driver exited 1, and forge's preflight
    reported BENCH CRASHED and burned an LLM agent re-authoring the driver.

    Resolving by path relative to ``__file__`` is what makes this independent of
    the importer.  The ``src.tools.perf`` import stays as the last resort for
    direct source-tree unit tests, where nothing has been materialized at all.

    Imports are function-local on purpose: everything from the anchor down is
    spliced into each task's ``scripts/task_runner.py``, so module-level imports
    here would land in the middle of that file.
    """
    import importlib.util
    import sys
    from pathlib import Path

    module = sys.modules.get("_aka_benchmark")
    if module is not None:
        return module
    helper = Path(__file__).resolve().parent / "_aka_benchmark.py"
    if helper.is_file():
        spec = importlib.util.spec_from_file_location("_aka_benchmark", helper)
        module = importlib.util.module_from_spec(spec)
        # Register before exec so a re-entrant import sees the partial module
        # rather than executing the helper a second time.
        sys.modules["_aka_benchmark"] = module
        spec.loader.exec_module(module)
        return module
    from src.tools.perf import aka_benchmark

    return aka_benchmark


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
    return _aka_benchmark_module().benchmark_cuda_graph_or_events(
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
