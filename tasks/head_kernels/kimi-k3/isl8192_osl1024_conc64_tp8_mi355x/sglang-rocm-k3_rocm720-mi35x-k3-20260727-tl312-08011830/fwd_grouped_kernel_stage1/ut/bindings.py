"""Bind the captured Kimi MLA reference and editable launcher independently."""
import importlib.machinery
import importlib.util
import os
import sys


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def resolve_pair(task_ut_dir):
    """Use the captured reference module as the factory's immutable fallback.

    The original reference carries the captured launch-geometry helpers consumed
    by make_launcher. The pristine upstream file lacks those helpers and is not
    a compatible substitute. Neither module is rebound in the serving package.
    """
    baseline = _load(
        "_aka_kimi_mla_frozen_reference",
        os.path.join(task_ut_dir, "baseline_ref", "decode_attention.py.orig"),
    )
    baseline_fn = baseline._decode_grouped_att_m_fwd
    candidate = _load(
        "_aka_kimi_mla_candidate",
        os.path.join(task_ut_dir, "kernel_src", "geak_mla_stage1.py"),
    )
    candidate_fn = candidate.make_launcher(baseline)
    if not callable(candidate_fn) or candidate_fn is baseline_fn:
        raise RuntimeError("candidate factory did not bind an independent launcher")
    if baseline._decode_grouped_att_m_fwd is not baseline_fn:
        raise RuntimeError("candidate factory replaced the frozen baseline binding")
    return baseline_fn, candidate_fn


REQUIRED_BINDING_FILES = ('bindings.py', 'baseline_ref/decode_attention.py.orig', 'kernel_src/geak_mla_stage1.py')

def validate_layout(task_ut_dir):
    """Validate the direct binding inputs without importing GPU packages."""
    from pathlib import Path
    task_ut = Path(task_ut_dir).resolve()
    task_root = task_ut.parent
    for relative in REQUIRED_BINDING_FILES:
        path = (task_ut / relative).resolve()
        if not path.is_relative_to(task_root) or not path.is_file():
            raise RuntimeError(f"independent Kimi binding input is missing or external: ut/{relative}")
    return True
