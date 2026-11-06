"""Generator for the batch metadata: uniform query length, ragged KV lengths, scattered pages."""
import torch


def make_inputs(case, inputs, generator, device):
    a = case["axes"]
    num_seqs, q_len, ctx_len = int(a["num_seqs"]), int(a["q_len"]), int(a["ctx_len"])
    pages = int(a["pages_per_seq"])
    if q_len > ctx_len:
        raise ValueError("A sequence cannot have more query tokens than KV tokens")
    # KV length of every sequence (context plus the new query tokens): the longest sequence has
    # ctx_len, the others between half of it and ctx_len, never fewer than q_len.
    low = max(q_len, ctx_len // 2)
    lengths = torch.randint(low, ctx_len + 1, (num_seqs,), generator=generator, device=device,
                            dtype=torch.int64)
    lengths[0] = ctx_len
    cu_seqlens_q = torch.arange(num_seqs + 1, device=device, dtype=torch.int64) * q_len
    # Pages of all sequences are interleaved in the pool, as after a long serving run.
    pool = torch.randperm(int(a["num_blocks"]), generator=generator, device=device)
    return {
        "cu_seqlens_q": cu_seqlens_q.to(torch.int32),
        "seqused_k": lengths.to(torch.int32),
        "block_table": pool.reshape(num_seqs, pages).to(torch.int32),
    }
