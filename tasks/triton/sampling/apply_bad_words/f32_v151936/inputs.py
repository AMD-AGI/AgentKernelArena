"""Request tables of a speculative-decoding batch with bad-word constraints.

Every request owns FOUR logit rows (one verified token and three draft tokens) and sits in a
randomly permuted request slot. About 80% of the slots carry 1..max_num_bad_words bad words of one
to four tokens; the first slot always carries max_num_bad_words so the launch width is real. The
prefix tokens of the bad words and the generated history are drawn from one small pool of token
ids, so prefixes match regularly (including matches that reach into the speculative input
tokens), while the final token of each word is a uniformly random vocabulary id.
"""
import torch

SPEC_STRIDE = 4
POOL = 6
WORD_LENGTHS = (1, 1, 2, 2, 2, 3, 3, 4)


def _randint(low, high, shape, generator, device):
    return torch.randint(low, high, shape, generator=generator, device=device, dtype=torch.int64)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, vocab = axes["n_tokens"], axes["vocab_size"]
    n_reqs, width, offsets_width = axes["n_reqs"], axes["bad_word_token_width"], axes["offsets_width"]
    history = axes["max_model_len"]
    max_words = int(case["scalars"]["max_num_bad_words"])
    pool = _randint(0, vocab, (POOL,), generator, device)

    counts = _randint(1, max_words + 1, (n_reqs,), generator, device)
    counts = torch.where(_randint(0, 5, (n_reqs,), generator, device) == 0, torch.zeros_like(counts), counts)
    counts[0] = max_words
    lengths = torch.tensor(WORD_LENGTHS, device=device)[_randint(0, len(WORD_LENGTHS), (n_reqs, max_words),
                                                                 generator, device)]
    lengths = torch.where(torch.arange(max_words, device=device) < counts[:, None], lengths,
                          torch.zeros_like(lengths))
    ends = lengths.cumsum(-1)
    offsets = torch.zeros((n_reqs, offsets_width), dtype=torch.int64, device=device)
    offsets[:, 1:max_words + 1] = ends

    flat = torch.arange(max_words * max(WORD_LENGTHS), device=device).expand(n_reqs, -1).contiguous()
    word = torch.searchsorted(ends.contiguous(), flat, right=True).clamp(max=max_words - 1)
    is_last = flat == (ends.gather(1, word) - 1)
    valid = flat < ends[:, -1:]
    random_tokens = _randint(0, vocab, flat.shape, generator, device)
    pooled = pool[_randint(0, POOL, flat.shape, generator, device)]
    tokens = torch.zeros((n_reqs, width), dtype=torch.int64, device=device)
    tokens[:, :flat.shape[1]] = torch.where(valid, torch.where(is_last, random_tokens, pooled),
                                            torch.zeros_like(flat))

    all_ids = pool[_randint(0, POOL, (n_reqs, history), generator, device)]
    prompt = _randint(1, 1025, (n_reqs,), generator, device)
    produced = _randint(0, 49, (n_reqs,), generator, device)
    produced = torch.where(_randint(0, 8, (n_reqs,), generator, device) == 0,
                           _randint(0, 3, (n_reqs,), generator, device), produced)
    rows = torch.arange(n_tokens, device=device)
    slot_of_request = torch.randperm(n_reqs, generator=generator, device=device)
    spec = pool[_randint(0, POOL, (n_tokens,), generator, device)]
    return {
        "expanded_idx_mapping": slot_of_request[rows // SPEC_STRIDE].to(torch.int32),
        "bad_word_token_ids": tokens.to(torch.int32),
        "bad_word_offsets": offsets.to(torch.int32),
        "num_bad_words": counts.to(torch.int32),
        "all_token_ids": all_ids.to(torch.int32),
        "prompt_len": prompt.to(torch.int32),
        "total_len": (prompt + produced).to(torch.int32),
        "input_ids": spec.to(torch.int32),
        "expanded_local_pos": (rows % SPEC_STRIDE).to(torch.int32),
    }
