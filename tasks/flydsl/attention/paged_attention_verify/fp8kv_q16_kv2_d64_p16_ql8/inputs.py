"""Inputs of the Muse Spark verify attention (one full-attention layer on one TP8 rank).

A DFlash2 verify step of ``requests`` requests x 8 rows. Request 0 holds ``ctx`` keys including the
step's 8 rows; every other request draws its key count uniformly in [max(8, ceil(ctx / 2)), ctx]
(ragged batches, as the repository's paged-attention tasks draw them). ``seq_lens[r]`` is the request's
committed context, its keys before the step's rows (SGLang's ``forward_batch.seq_lens`` in a target
verify; the kernel adds the 8 rows itself).

Cache: page-16 NHD planes ``[slot, kv_head, 64]`` of float8_e4m3fn codes, used as exact values (the
layer keeps no KV scale). Every request owns ``ceil(keys / 16)`` distinct pages in random order (page
0 and the pages no request owns hold stale codes); the page table's entries past a request's pages
are random stale page ids, so an implementation that reads past ``seq_lens + 8`` keys picks up
unrelated keys.

Data. ``q`` is what the layer's K1 hands the attention: each head's row RMS-normalized and multiplied
by the checkpoint's query scale 5.473 (``config.QK_SCALE``), bf16. K / V codes are unit normal. The
attention is made peaked, as real heads are, and the masks made visible:

* every (request, KV head) has a direction c; each query head of that KV head is
  ``a c + noise`` (a in [0.5, 1.5], before the norm), so it scores keys along c highly;
* planted keys ``s c + 0.3 noise`` (s in [6, 10]: logits of about 15-45, where a random key's have a
  standard deviation of about 5.5) sit at the step's own 8 positions (so the causal mask among the step's rows decides which of
  them a row sees), at key 0 (a sink), at the middle of the committed context and at its last key
  (maxima in different partitions, a late rescale);
* the slots of a request's last page past its keys hold stronger planted keys (s in [10, 12]): they
  must be masked.

No discrete decision depends on these values (softmax is continuous), so no margin is needed.
"""

import torch

Q_LEN = 8  # rows a request: DFlash2's 8 draft tokens
PAGE = 16
HEAD_DIM = 64
KV_HEADS = 2  # a rank's KV heads (16 / TP8)
GQA = 8  # query heads a KV head
QK_SCALE = 5.4730064863875  # the query scale K1 applies after the per-head norm


def _uniform(generator, device, low, high, *shape):
    return low + (high - low) * torch.rand(shape, generator=generator, device=device)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    R, ctx = axes["requests"], axes["ctx"]
    max_blocks, kv_pages = axes["max_blocks"], axes["kv_pages"]
    kv_slots = kv_pages * PAGE
    g = generator

    # key counts (the step's 8 rows included) and committed contexts
    keys = [ctx]
    low = max(Q_LEN, -(-ctx // 2))
    for _ in range(1, R):
        keys.append(int(torch.randint(low, ctx + 1, (1,), generator=g, device=device)))
    seq_lens = torch.tensor([k - Q_LEN for k in keys], dtype=torch.int32, device=device)

    # pages: distinct ones a request in random order; stale ids past each request's pages
    table = torch.randint(0, kv_pages, (R, max_blocks), generator=g, device=device, dtype=torch.int32)
    perm = (torch.randperm(kv_pages - 1, generator=g, device=device) + 1).to(torch.int32)
    at = 0
    for r in range(R):
        n = -(-keys[r] // PAGE)
        table[r, :n] = perm[at:at + n]
        at += n

    k = torch.randn(kv_slots, KV_HEADS, HEAD_DIM, generator=g, device=device)
    v = torch.randn(kv_slots, KV_HEADS, HEAD_DIM, generator=g, device=device)
    q = torch.empty(R * Q_LEN, KV_HEADS, GQA, HEAD_DIM, device=device)
    for r in range(R):
        prefix = keys[r] - Q_LEN
        n = -(-keys[r] // PAGE)
        c = torch.randn(KV_HEADS, HEAD_DIM, generator=g, device=device)
        c = c / c.norm(dim=-1, keepdim=True)  # [h, 64] unit directions
        a = _uniform(g, device, 0.5, 1.5, Q_LEN, KV_HEADS, GQA, 1)
        u = a * 8.0 * c[None, :, None, :] + torch.randn(Q_LEN, KV_HEADS, GQA, HEAD_DIM, generator=g,
                                                        device=device)
        q[r * Q_LEN:(r + 1) * Q_LEN] = u * torch.rsqrt(u.square().mean(-1, keepdim=True)) * QK_SCALE

        def slot_of(pos):
            pos = torch.as_tensor(pos, dtype=torch.int64, device=device)
            return table[r, pos // PAGE].long() * PAGE + pos % PAGE

        # visible planted keys: the step's 8 rows, a sink, the middle and the last committed key
        visible = [prefix + j for j in range(Q_LEN)]
        if prefix > 0:
            visible += [0, prefix // 2, prefix - 1]
        visible = sorted(set(visible))
        tail = list(range(keys[r], n * PAGE))  # the last page's slots past the request's keys
        for positions, low_s, high_s in ((visible, 6.0, 10.0), (tail, 10.0, 12.0)):
            if not positions:
                continue
            slots = slot_of(positions)
            s = _uniform(g, device, low_s, high_s, len(positions), KV_HEADS, 1)
            noise = 0.3 * torch.randn(len(positions), KV_HEADS, HEAD_DIM, generator=g, device=device)
            k[slots] = s * c[None] + noise

    return {
        "q": q.reshape(R * Q_LEN, KV_HEADS * GQA * HEAD_DIM).to(torch.bfloat16),
        "k_cache": k.to(torch.float8_e4m3fn),
        "v_cache": v.to(torch.float8_e4m3fn),
        "page_table": table,
        "seq_lens": seq_lens,
    }
