"""Ragged sequence metadata of the paged batch-prefill scenario.

Every sequence appends qo tokens to a cached prefix and attends to all kv = prefix + qo keys
(a fresh prompt has no prefix, so kv == qo). Lengths are deterministic per case so both input
draws do the same work. The batch's mean query length is qo_len and its mean key length is
kv_len. Query lengths are lengthened and shortened in pairs by qo_len/8, 2*qo_len/8 and
3*qo_len/8 in turn. Cached prefixes (prefix = kv_len - qo_len on average) are shortened and
lengthened in pairs by prefix/8, 3*prefix/8 and 2*prefix/8 in turn, so a batch mixes prompt
and prefix lengths the way a real prefill/extend batch does. Each sequence owns
ceil(kv / page_size) pages; the page list kv_page_indices is a random permutation of the
whole page pool, of which the first kv_indptr[batch] entries are used.
"""
import torch


def _paired(batch, mean, pattern, sign):
    lengths = [mean] * batch
    for pair in range(batch // 2):
        delta = mean * pattern[pair % 3] // 8
        lengths[2 * pair] += sign * delta
        lengths[2 * pair + 1] -= sign * delta
    return lengths


def sequence_lengths(batch, qo_len, kv_len):
    """Per-sequence (query length, key length) pairs of a case."""
    if kv_len < qo_len:
        raise ValueError("kv_len must be at least qo_len: the keys include the new tokens")
    queries = _paired(batch, qo_len, (1, 2, 3), 1)
    prefixes = _paired(batch, kv_len - qo_len, (1, 3, 2), -1)
    return [(q, q + p) for q, p in zip(queries, prefixes)]


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    page_size = axes["page_size"]
    lengths = sequence_lengths(axes["batch"], axes["qo_len"], axes["kv_len"])
    cu_seqlens_q, kv_indptr, last_page_lens = [0], [0], []
    for qo, kv in lengths:
        pages = -(-kv // page_size)
        cu_seqlens_q.append(cu_seqlens_q[-1] + qo)
        kv_indptr.append(kv_indptr[-1] + pages)
        last_page_lens.append(kv - (pages - 1) * page_size)
    if cu_seqlens_q[-1] != axes["num_tokens"] or kv_indptr[-1] > axes["num_pages"]:
        raise ValueError("sequence layout does not fit the declared tensors")

    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return {"cu_seqlens_q": tensor(cu_seqlens_q), "kv_indptr": tensor(kv_indptr),
            "kv_last_page_lens": tensor(last_page_lens)}
