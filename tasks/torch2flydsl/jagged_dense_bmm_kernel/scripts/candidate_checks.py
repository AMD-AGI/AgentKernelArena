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


# Only these value-preserving operations carry the caller's integer offset
# provenance. Allocation/fill/copy and float conversions do not create metadata.
METADATA_VIEWS = frozenset({
    "aten::view", "aten::_unsafe_view", "aten::reshape", "aten::alias",
    "aten::detach", "aten::as_strided", "aten::transpose", "aten::t",
    "aten::permute", "aten::unsqueeze", "aten::squeeze", "aten::slice",
    "aten::select", "aten::expand", "aten::narrow", "aten::contiguous",
    "aten::to", "aten::_to_copy", "aten::clone",
})
METADATA_ARITHMETIC = frozenset({
    ("aten::sub", "Tensor"), ("aten::max", ""),
    ("aten::_local_scalar_dense", ""),
})


@contextmanager
def candidate_preparation_only(metadata_inputs=()):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    integer_types = {torch.int32, torch.int64}
    # Strong references prevent object-ID reuse; versions invalidate aliases
    # whose contents were overwritten after deriving them from the offsets.
    tracked = {}
    def remember(value, sources=()):
        if isinstance(value, torch.Tensor) and value.dtype in integer_types:
            version = value._version
            # Dispatch returns before PyTorch attaches view metadata. A new
            # view initially appears at version zero, then inherits its base's
            # counter; original offsets have already been filled in-place.
            for source in sources:
                if value.untyped_storage()._cdata == source.untyped_storage()._cdata:
                    version = source._version
                    break
            tracked[id(value)] = (value, version)
    for value in metadata_inputs:
        remember(value)

    class PreparationOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = func._schema.name
            tensors = [x for x in tree_leaves((args, kwargs)) if isinstance(x, torch.Tensor)]
            def is_metadata(value):
                entry = tracked.get(id(value))
                return (entry is not None and entry[0] is value
                        and value.dtype in integer_types and entry[1] == value._version)
            metadata_only = bool(tensors) and all(is_metadata(x) for x in tensors)
            operation = (name, func._schema.overload_name)
            metadata_arithmetic = metadata_only and operation in METADATA_ARITHMETIC
            if func._schema.is_mutable and any(id(x) in tracked for x in tensors):
                raise RuntimeError("Candidate attempted to overwrite protected offset metadata")
            if name not in PREPARATION_OPS and not metadata_arithmetic:
                raise RuntimeError(
                    f"Candidate issued non-preparation PyTorch operation {name}; "
                    "operator computation must use FlyDSL kernels"
                )
            result = func(*args, **kwargs)
            if metadata_only and (name in METADATA_VIEWS or metadata_arithmetic):
                for value in tree_leaves(result):
                    remember(value, tensors)
            return result

    with PreparationOnly():
        yield


def checked_candidate_invocation(fn, observed, *args, _metadata_inputs=(), **kwargs):
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
        with candidate_preparation_only(_metadata_inputs):
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
                def checked(*args, __target=target, __entry=name, **kwargs):
                    # These positions belong to the protected public/prepared
                    # interfaces. Dense/jagged/bias tensors never become roots.
                    offset_position, offset_name = {
                        "flydsl_jagged_dense_bmm": (3, "seq_offsets"),
                        "jagged_dense_bmm": (4, "SEQ_OFFSETS"),
                    }.get(__entry, (None, None))
                    offsets = (args[offset_position] if offset_position is not None
                               and len(args) > offset_position else kwargs.get(offset_name))
                    return checked_candidate_invocation(
                        __target, observed, *args, _metadata_inputs=(offsets,), **kwargs)
                setattr(proxy, name, checked)
            return proxy
        return mod

    h._load_module = audited_loader
    try:
        yield observed
    finally:
        h._load_module = original
