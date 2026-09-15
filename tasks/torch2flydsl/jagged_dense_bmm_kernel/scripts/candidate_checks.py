"""Candidate-only execution audit, outside all benchmark timing windows.

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


@contextmanager
def candidate_preparation_only():
    from torch.utils._python_dispatch import TorchDispatchMode

    class PreparationOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = func._schema.name
            if name not in PREPARATION_OPS:
                raise RuntimeError(
                    f"Candidate issued non-preparation PyTorch operation {name}; "
                    "operator computation must use FlyDSL kernels"
                )
            return func(*args, **(kwargs or {}))

    with PreparationOnly():
        yield


def checked_candidate_invocation(fn, observed, *args, **kwargs):
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
        with candidate_preparation_only():
            result = fn(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    if not seen:
        raise RuntimeError("No FlyDSL kernel runtime invocation observed in candidate operator call")
    observed.update(seen)
    return result


@contextmanager
def audit_candidate_calls(h):
    """Audit imported candidate and its operator calls during correctness only.

    Baseline/reference calls run outside this scope. The loader is restored
    before the harness reloads the candidate for ordinary device timing; no
    dispatcher audit is added to a measured call or its captured graph.
    """
    observed = set()
    if h.ARENA_PROVIDED_BASELINE:
        yield observed
        return
    original = h._load_module

    def audited_loader(kernel_dir, filename, alias):
        if filename != h.KERNEL_FILE:
            return original(kernel_dir, filename, alias)
        with candidate_preparation_only():
            mod = original(kernel_dir, filename, alias)
        if mod is not None:
            from types import SimpleNamespace
            proxy = SimpleNamespace(**vars(mod))
            for name, target in tuple(vars(mod).items()):
                if not (name.startswith("flydsl_") or name == "jagged_dense_bmm") or not callable(target):
                    continue
                @wraps(target)
                def checked(*args, __target=target, **kwargs):
                    return checked_candidate_invocation(__target, observed, *args, **kwargs)
                setattr(proxy, name, checked)
            return proxy
        return mod

    h._load_module = audited_loader
    try:
        yield observed
    finally:
        h._load_module = original
