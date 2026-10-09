# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Performance baseline for this workload, copied from the schema bundle.

Everything from ``from __future__`` on is the bundle's ``baseline`` solution,
taken without edit from schema v2 except for one line: the bundle imports
``topk_transform_512_v2``, which sglang renamed to ``topk_transform_paged_v2``
without changing it (sglang PR #36831). ``resolve_transform`` below binds that
name to whichever of the two the installed sglang exports, preferring the
current one; the call itself is unchanged. It is the production implementation
a ported FlyDSL kernel is scored against. Do not edit it here: change it in the
bundle and copy it down again.

``run`` is the entry point the bundle exports.
"""

from __future__ import annotations


def _paged_baseline(
    scores, seq_lens, metadata, page_size, page_tables, out_page_indices
):
    topk_transform_512_v2 = resolve_transform()[1]

    topk_transform_512_v2(
        scores=scores,
        seq_lens=seq_lens,
        page_tables=page_tables,
        out_page_indices=out_page_indices,
        page_size=page_size,
        metadata=metadata,
    )


_callable = _paged_baseline


def run(*args, **kwargs):
    return _callable(*args, **kwargs)


TRANSFORM_MODULE = "sglang.kernels.ops.attention.dsv4.topk"
TRANSFORM_NAMES = ("topk_transform_paged_v2", "topk_transform_512_v2")


def resolve_transform():
    """Return (name, function) of the installed paged top-k v2 entry point."""
    import importlib

    module = importlib.import_module(TRANSFORM_MODULE)
    for name in TRANSFORM_NAMES:
        function = getattr(module, name, None)
        if callable(function):
            return name, function
    raise ImportError(f"{TRANSFORM_MODULE} exports none of {TRANSFORM_NAMES}")
