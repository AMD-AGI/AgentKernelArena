"""Inputs for apply_penalties: grouped speculative-decode rows, heterogeneous penalties, sparse prompt/output histories."""
import torch

_REP = (1.05, 1.1, 1.2, 1.5)
_FREQ = (0.1, 0.3, 0.5)
_PRES = (0.1, 0.3, 0.6)
_PROMPT_DENSITY = 0.03     # fraction of the vocabulary present in a prompt
_OUTPUT_DENSITY = 0.01     # fraction of the vocabulary generated so far
_CHUNK = 64


def _pick(values, n, generator, device):
    table = torch.tensor(values, dtype=torch.float32, device=device)
    return table[torch.randint(0, len(values), (n,), generator=generator, device=device)]


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_reqs, per_req, vocab = axes["n_reqs"], axes["tokens_per_req"], axes["vocab_size"]
    n_tokens, words = axes["n_tokens"], axes["n_mask_words"]
    # Request states are visited in a shuffled order; the rows of one request are consecutive.
    states = torch.randperm(n_reqs, generator=generator, device=device).to(torch.int32)
    mapping = states.repeat_interleave(per_req)
    local_pos = torch.arange(per_req, device=device, dtype=torch.int32).repeat(n_reqs)
    token_ids = torch.randint(0, vocab, (n_tokens,), generator=generator, device=device, dtype=torch.int64).to(torch.int32)

    ids = torch.arange(n_reqs, device=device)
    rep, freq, pres = (_pick(_REP, n_reqs, generator, device), _pick(_FREQ, n_reqs, generator, device),
                       _pick(_PRES, n_reqs, generator, device))
    cls = ids % 8
    rep = torch.where((cls == 3) | (cls == 5), torch.ones_like(rep), rep)               # repetition disabled
    freq = torch.where((cls == 3) | (cls == 4), torch.zeros_like(freq), freq)           # frequency disabled
    pres = torch.where((cls == 3) | (cls == 4), torch.zeros_like(pres), pres)           # presence disabled

    bits = torch.arange(32, device=device, dtype=torch.int64)
    prompt = torch.empty(n_reqs, words, dtype=torch.int32, device=device)
    counts = torch.zeros(n_reqs, vocab, dtype=torch.int32, device=device)
    for start in range(0, n_reqs, _CHUNK):
        end = min(n_reqs, start + _CHUNK)
        hit = torch.rand(end - start, words, 32, generator=generator, device=device) < _PROMPT_DENSITY
        packed = (hit.long() << bits).sum(-1)
        prompt[start:end] = torch.where(packed >= 1 << 31, packed - (1 << 32), packed).to(torch.int32)
        seen = torch.rand(end - start, vocab, generator=generator, device=device) < _OUTPUT_DENSITY
        times = torch.randint(1, 8, (end - start, vocab), generator=generator, device=device, dtype=torch.int64)
        counts[start:end] = (seen.long() * times).to(torch.int32)
    logits = 2.0 * torch.randn(n_tokens, vocab, generator=generator, device=device, dtype=torch.float32)
    return {"logits": logits, "expanded_idx_mapping": mapping, "token_ids": token_ids,
            "expanded_local_pos": local_pos, "repetition_penalty": rep, "frequency_penalty": freq,
            "presence_penalty": pres, "prompt_bin_mask": prompt, "output_bin_counts": counts}
