from __future__ import annotations


def _paged_baseline(
    scores, seq_lens, metadata, page_size, page_tables, out_page_indices
):
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512_v2

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
