import torch

_CHUNK = 32


def reference(*, logits, expanded_idx_mapping, token_ids, expanded_local_pos, repetition_penalty,
              frequency_penalty, presence_penalty, prompt_bin_mask, output_bin_counts):
    n_tokens, vocab = logits.shape
    device = logits.device
    out = torch.empty(logits.shape, dtype=torch.float64, device=device)
    bits = torch.arange(32, device=device, dtype=torch.int64)
    for start in range(0, n_tokens, _CHUNK):
        end = min(n_tokens, start + _CHUNK)
        rows = torch.arange(start, end, device=device)
        state = expanded_idx_mapping[start:end].long()
        pos = expanded_local_pos[start:end].long()
        counts = output_bin_counts[state].long()
        for prev in range(int(pos.max())):
            taking = torch.nonzero(pos > prev).squeeze(1)
            draft = token_ids[(rows - pos + prev + 1).clamp(max=n_tokens - 1)].long()
            counts[taking, draft[taking]] += 1
        seen = counts > 0
        packed = prompt_bin_mask[state].long()
        in_prompt = (((packed.unsqueeze(-1) >> bits) & 1) != 0).reshape(end - start, -1)[:, :vocab]
        rep = repetition_penalty[state].double()[:, None]
        freq = frequency_penalty[state].double()[:, None]
        pres = presence_penalty[state].double()[:, None]
        x = logits[start:end].double()
        scale = torch.where(in_prompt | seen, rep, torch.ones_like(rep))
        x = torch.where(x > 0, x / scale, x * scale)
        x = x - freq * counts.double() - pres * seen.double()
        out[start:end] = x
    return out
