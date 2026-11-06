"""Token-granular (one token per page) KV pool and ragged page lists of one decode batch.

As in SGLang's AITER attention backend, the cache is a token pool ``[slots, kv_heads, head_size]``
and each sequence's tokens are listed in CSR form: the slots of sequence ``s`` are
``kv_indices[kv_indptr[s]:kv_indptr[s + 1]]``, in token order. Slot 0 is the pool's reserved
padding slot. Sequences take distinct slots in random order from 1..num_seqs*max_seq_len, so a
sequence's tokens are scattered. ``kv_indices`` is a buffer with room for num_seqs*max_seq_len
entries, as a CUDA-graph decode buffer is; entries past ``kv_indptr[num_seqs]`` are 0.

Context lengths are ragged: most sequences hold between half of ``max_seq_len`` and all of it, about
one in sixteen is a short request of 1 to 512 tokens, and one sequence holds exactly
``max_seq_len``. Every slot that holds no token of any sequence, slot 0 included, is NaN.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    num_seqs = axes["num_seqs"]
    max_seq_len = axes["max_seq_len"]
    num_slots = axes["num_slots"]
    num_kv_heads = axes["num_kv_heads"]
    head_size = axes["head_size"]
    if case["scalars"]["max_context_len"] != max_seq_len:
        raise ValueError("Each case's max_context_len scalar must equal its max_seq_len axis")

    low = (max_seq_len + 1) // 2
    lengths = torch.randint(low, max_seq_len + 1, (num_seqs,), generator=generator, device=device)
    short = torch.rand((num_seqs,), generator=generator, device=device) < 1.0 / 16
    short_lengths = torch.randint(1, min(max_seq_len, 512) + 1, (num_seqs,), generator=generator, device=device)
    lengths = torch.where(short, short_lengths, lengths)
    longest = torch.randint(0, num_seqs, (1,), generator=generator, device=device)
    lengths[longest] = max_seq_len

    kv_indptr = torch.zeros(num_seqs + 1, dtype=torch.int64, device=device)
    kv_indptr[1:] = lengths.cumsum(0)
    total = int(kv_indptr[-1])
    slots = torch.randperm(num_slots - 1, generator=generator, device=device)[:total] + 1
    kv_indices = torch.zeros(num_slots - 1, dtype=torch.int32, device=device)
    kv_indices[:total] = slots.to(torch.int32)
    unused = torch.ones(num_slots, dtype=torch.bool, device=device)
    unused[slots] = False

    key_cache = torch.randn((num_slots, num_kv_heads, head_size), generator=generator, device=device,
                            dtype=torch.float32).to(torch.bfloat16)
    key_cache[unused] = float("nan")
    value_cache = torch.randn((num_slots, num_kv_heads, head_size), generator=generator, device=device,
                              dtype=torch.float32).to(torch.bfloat16)
    value_cache[unused] = float("nan")
    return {"key_cache": key_cache, "value_cache": value_cache, "kv_indptr": kv_indptr.to(torch.int32),
            "kv_indices": kv_indices}
