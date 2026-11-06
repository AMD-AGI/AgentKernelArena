"""Inputs for sample_recovered_tokens: correlated draft/target distributions and exponential noise."""
import torch

_CHUNK = 32
# Required relative gap between the best and second-best score of a position. The kernel multiplies
# in float32 and the reference in float64; a gap this large (well above float32 rounding) keeps the
# winner unambiguous.
_GAP = 1.0e-4
_BOOST = 3.0e-4


def _probs(rows, vocab, generator, device):
    target_logits = torch.randn(rows, vocab, generator=generator, device=device, dtype=torch.float32) * 2.0
    draft_logits = target_logits + torch.randn(rows, vocab, generator=generator, device=device,
                                               dtype=torch.float32)
    return torch.softmax(draft_logits, -1), torch.softmax(target_logits, -1)


def _best_two(draft, target, inv_q):
    score = (target.double() - draft.double()).clamp_min(0) * inv_q.double()
    top = score.topk(2, dim=-1)
    return top.values[:, 0], top.values[:, 1], top.indices[:, 0]


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_reqs, spec_len, vocab = axes["n_reqs"], axes["spec_len"], axes["vocab_size"]
    n_tokens = n_reqs * spec_len
    noise = torch.empty(n_reqs, vocab, dtype=torch.float32, device=device)
    noise.exponential_(generator=generator)
    inv_q = (1.0 / noise.clamp_min(1.0e-30)).contiguous()
    draft = torch.empty(n_tokens, vocab, dtype=torch.float32, device=device)
    target = torch.empty(n_tokens, vocab, dtype=torch.float32, device=device)
    for start in range(0, n_tokens, _CHUNK):
        end = min(n_tokens, start + _CHUNK)
        d, t = _probs(end - start, vocab, generator, device)
        q = inv_q[torch.arange(start, end, device=device) // spec_len]
        for _ in range(4):
            best, second, winner = _best_two(d, t, q)
            close = (best - second) < _GAP * best
            if not bool(close.any()):
                break
            rows = torch.nonzero(close).flatten()
            cols = winner[rows]
            value = t[rows, cols].double()
            gain = (value - d[rows, cols].double()).clamp_min(0) * _BOOST
            t[rows, cols] = (value + gain).float()
        else:
            best, second, _ = _best_two(d, t, q)
            if bool(((best - second) < 0.5 * _GAP * best).any()):
                raise RuntimeError("could not separate the best score of every position")
        draft[start:end] = d
        target[start:end] = t
    cu = (torch.arange(1, n_reqs + 1, device=device, dtype=torch.int32) * spec_len)
    return {"cu_num_draft_tokens": cu, "draft_probs": draft, "target_probs": target, "inv_q": inv_q}
