import torch

_CHUNK = 32


def reference(*, cu_num_draft_tokens, draft_token_ids, draft_probs, target_probs, inv_q):
    n_tokens = draft_probs.shape[0]
    out = torch.empty(n_tokens, dtype=torch.int32, device=draft_probs.device)
    positions = torch.arange(n_tokens, device=draft_probs.device)
    # Request of each position: the first request whose cumulative count exceeds it.
    request = torch.searchsorted(cu_num_draft_tokens.long(), positions, right=True)
    for start in range(0, n_tokens, _CHUNK):
        end = min(n_tokens, start + _CHUNK)
        prob = (target_probs[start:end].double() - draft_probs[start:end].double()).clamp_min(0)
        score = prob * inv_q[request[start:end]].double()
        out[start:end] = score.argmax(-1).to(torch.int32)
    return out
