"""Request-state tables of the penalty-state builder.

The tables hold one slot per request; ``idx_mapping`` is a permutation of the slots. Each slot has a prefill length of about
``max_prefill_len`` (up to a quarter shorter; slot ``idx_mapping[0]`` has exactly that length) and a
prompt that is the whole prefill for half of the slots and leaves up to a quarter of it as output
tokens (a request resumed after preemption) for the rest. About a third of the tokens come from a
small set of frequent tokens, so atomics contend; the rest are uniform over the vocabulary. Edge token ids (0, 31, 32, vocab-1) sit at both ends of one row and about 5% of the slots have an empty prompt. Columns
past ``prefill_len`` are filled with valid tokens too and must not be counted.
"""
import torch

HOT_TOKENS = 256
HOT_FRACTION = 0.3


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_reqs = slots = axes["n_reqs"]
    length, vocab = axes["max_prefill_len"], axes["vocab_size"]
    idx = inputs["idx_mapping"]

    tokens = torch.randint(0, vocab, (slots, length), generator=generator, device=device, dtype=torch.int64)
    hot = torch.randint(0, vocab, (HOT_TOKENS,), generator=generator, device=device, dtype=torch.int64)
    pick = torch.randint(0, HOT_TOKENS, (slots, length), generator=generator, device=device, dtype=torch.int64)
    use_hot = torch.rand((slots, length), generator=generator, device=device) < HOT_FRACTION
    tokens = torch.where(use_hot, hot[pick], tokens)

    shrink = (torch.rand(slots, generator=generator, device=device) * (length // 4 + 1)).to(torch.int64)
    prefill = (length - shrink).clamp(min=1)
    prefill[idx[0].to(torch.int64)] = length
    resumed = torch.rand(slots, generator=generator, device=device) < 0.5
    outputs = (torch.rand(slots, generator=generator, device=device) * (prefill // 4 + 1)).to(torch.int64)
    prompt = (prefill - torch.where(resumed, outputs, torch.zeros_like(outputs))).clamp(min=1)
    # Edge tokens: the last vocabulary id (bit 31 of the last word), id 0, and ids around a word
    # boundary, in the slot with the full-length prefill; plus a few requests with an empty prompt.
    first = int(idx[0])
    edge = torch.tensor([vocab - 1, 0, 31, 32], dtype=torch.int64, device=device)
    for k in range(min(4, length)):
        tokens[first, k] = edge[k]
        tokens[first, length - 1 - k] = edge[k]
    empty_prompt = torch.rand(slots, generator=generator, device=device) < 0.05
    prompt = torch.where(empty_prompt & (prefill > 0), torch.zeros_like(prompt), prompt)
    prompt[first] = prompt[first].clamp(min=1)
    return {"all_token_ids": tokens.to(torch.int32),
            "prompt_len": prompt.to(torch.int32), "prefill_len": prefill.to(torch.int32)}
