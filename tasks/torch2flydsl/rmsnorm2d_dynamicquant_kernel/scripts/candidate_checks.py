"""Candidate-only execution audit with a per-call performance dispatch policy.

Static dependency checks reject AITER and alternative kernel backends. This
runtime check also catches PyTorch tensor-method/operator decompositions that
an import/name blacklist cannot resolve. It is not a Python security sandbox.
"""
from contextlib import contextmanager
from functools import wraps
import sys

# Allocation, initialization, copying and metadata/view operations are host
# launch preparation. Arithmetic, reductions, sorting and library operators
# must be implemented by the submitted FlyDSL kernels instead.
PREPARATION_OPS = frozenset({
    "aten::empty", "aten::empty_like", "aten::empty_strided",
    "aten::new_empty", "aten::new_empty_strided", "aten::new_zeros",
    "aten::new_ones", "aten::new_full", "aten::zeros", "aten::zeros_like",
    "aten::ones", "aten::ones_like", "aten::full", "aten::full_like",
    "aten::zero_", "aten::fill_", "aten::copy_", "aten::clone",
    "aten::view", "aten::_unsafe_view", "aten::reshape", "aten::alias",
    "aten::detach", "aten::as_strided", "aten::transpose", "aten::t",
    "aten::permute", "aten::unsqueeze", "aten::squeeze", "aten::slice",
    "aten::select", "aten::expand", "aten::narrow", "aten::contiguous",
    "aten::to", "aten::_to_copy", "aten::view_as_real", "aten::view_as_complex",
})


class CandidateDependencyViolation(RuntimeError):
    """A forbidden candidate computation, even if a capture fallback catches it."""


@contextmanager
def candidate_preparation_only(violations=None):
    from torch.utils._python_dispatch import TorchDispatchMode

    class PreparationOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = func._schema.name
            if name not in PREPARATION_OPS:
                reason = (f"Candidate issued non-preparation PyTorch operation {name}; "
                          "operator computation must use FlyDSL kernels")
                if violations is not None:
                    violations.append(reason)
                raise CandidateDependencyViolation(reason)
            return func(*args, **(kwargs or {}))

    with PreparationOnly():
        yield


def checked_candidate_invocation(fn, observed, *args, _policy_violations=None, **kwargs):
    """Require FlyDSL launch evidence from this candidate call, not the oracle."""
    seen = set()
    previous = sys.getprofile()
    def profile(frame, event, arg):
        if event != "call" or not frame.f_globals.get("__name__", "").startswith("flydsl."):
            return
        owner = type(frame.f_locals.get("self")).__name__
        if frame.f_code.co_name == "__call__" and any(s in owner for s in ("Jit", "Compiled", "Kernel")):
            seen.add(frame.f_globals["__name__"] + "." + owner)
    try:
        sys.setprofile(profile)
        with candidate_preparation_only(_policy_violations):
            result = fn(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    if not seen:
        raise RuntimeError("No FlyDSL kernel runtime invocation observed in candidate operator call")
    observed.update(seen)
    return result


@contextmanager
def audit_candidate_calls(h, *, performance=False):
    """Audit candidate calls; suppress only Python profiling inside the timer.

    The preparation-only dispatch mode remains on the candidate callback during
    all canonical capture/warmup/event calls. Baseline/reference calls are outside
    candidate wrappers; the loader and benchmark binding are always restored.
    """
    observed = set()
    if h.ARENA_PROVIDED_BASELINE:
        yield observed
        return
    original_benchmark = getattr(h, "benchmark_cuda_graph_or_events", None)
    benchmark_active = False
    probe_active = False
    violations = []
    if performance and original_benchmark is None:
        raise RuntimeError("Performance audit requires the protected canonical benchmark binding")

    def candidate_call(target, *args, **kwargs):
        if performance and not probe_active:
            # This is the actual callback used by canonical warmup/capture/event
            # timing. Only host-side dispatch policy runs here: no profiling,
            # synchronization or additional device operations are inserted.
            with candidate_preparation_only(violations):
                return target(*args, **kwargs)
        return checked_candidate_invocation(target, observed, *args, _policy_violations=violations, **kwargs)

    def audited_benchmark(fn, *args, **kwargs):
        nonlocal benchmark_active, probe_active
        if benchmark_active:
            raise RuntimeError("Nested candidate benchmark audit is not supported")
        prepare = kwargs.get("prepare_fn")
        if prepare is not None:
            prepare()
        try:
            probe_active = True
            # Probe this actual performance closure outside every clock/capture.
            # Candidate wrappers observe FlyDSL calls; oracle calls are untouched.
            fn()
        finally:
            probe_active = False
            if prepare is not None:
                prepare()
        benchmark_active = True
        try:
            result = original_benchmark(fn, *args, **kwargs)
            if violations:
                raise CandidateDependencyViolation(violations[0])
            return result
        finally:
            benchmark_active = False

    original = h._load_module

    def audited_loader(kernel_dir, filename, alias):
        if filename != h.KERNEL_FILE:
            return original(kernel_dir, filename, alias)
        with candidate_preparation_only(violations):
            mod = original(kernel_dir, filename, alias)
        if mod is not None:
            for name, target in tuple(vars(mod).items()):
                if not name.startswith("flydsl_") or not callable(target):
                    continue
                @wraps(target)
                def checked(*args, __target=target, **kwargs):
                    return candidate_call(__target, *args, **kwargs)
                setattr(mod, name, checked)
        return mod

    h._load_module = audited_loader
    if performance:
        h.benchmark_cuda_graph_or_events = audited_benchmark
    try:
        yield observed
    finally:
        if performance:
            h.benchmark_cuda_graph_or_events = original_benchmark
        h._load_module = original
    if violations:
        raise CandidateDependencyViolation(violations[0])
