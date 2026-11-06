"""Per-request sampling-constraint tables of a speculative-decoding batch.

Every request owns FOUR logit rows (one verified token and three draft tokens) at consecutive
positions, in a randomly permuted request slot. Roughly half of the requests carry an allowed-token
list (2..1024 ids, log-uniform), half a sparse logit-bias list (1..128 ids, normal biases) and 30% a
minimum-length stop-token list (1..16 ids). Lists are combined at random, and bias and stop tokens
often coincide with the allowed list, so the order of the three stages matters. Token ids inside a
list are distinct by construction (an arithmetic progression modulo the vocabulary with a step
coprime to it), as a request's bias map and allowed set are sets upstream.
"""
import math

import torch

SPEC_STRIDE = 4
ALLOWED_MAX, BIAS_MAX, STOP_MAX = 1024, 128, 16


def _randint(low, high, shape, generator, device):
    return torch.randint(low, high, shape, generator=generator, device=device, dtype=torch.int64)


def _uniform(shape, generator, device):
    return torch.rand(shape, generator=generator, device=device)


def _log_count(high, shape, generator, device, low=1):
    exponent = _uniform(shape, generator, device) * (math.log2(high) - math.log2(low)) + math.log2(low)
    return torch.exp2(exponent).round().long().clamp(low, high)


def _progression(start, step, width, vocab):
    return (start[:, None] + torch.arange(width, device=start.device) * step[:, None]) % vocab


def _pad(values, counts, width):
    padded = torch.zeros((values.shape[0], width), dtype=values.dtype, device=values.device)
    padded[:, :values.shape[1]] = torch.where(torch.arange(values.shape[1], device=values.device) < counts[:, None],
                                              values, torch.zeros_like(values))
    return padded


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, vocab, n_reqs = axes["n_tokens"], axes["vocab_size"], axes["n_reqs"]
    allowed_width, bias_width, stop_width = axes["allowed_width"], axes["bias_width"], axes["stop_width"]
    zeros = torch.zeros((n_reqs,), dtype=torch.long, device=device)

    use_allowed = _uniform((n_reqs,), generator, device) < 0.5
    use_bias = _uniform((n_reqs,), generator, device) < 0.5
    use_stop = _uniform((n_reqs,), generator, device) < 0.3
    n_allowed = torch.where(use_allowed, _log_count(allowed_width, (n_reqs,), generator, device, low=2), zeros)
    n_bias = torch.where(use_bias, _log_count(BIAS_MAX, (n_reqs,), generator, device), zeros)
    n_stop = torch.where(use_stop, _randint(1, STOP_MAX + 1, (n_reqs,), generator, device), zeros)

    def stride_parameters():
        step = _randint(1, vocab, (n_reqs,), generator, device) | 1
        step = torch.where(step % 1187 == 0, torch.ones_like(step), step)  # vocab = 128 * 1187
        return _randint(0, vocab, (n_reqs,), generator, device), step

    allowed_start, allowed_step = stride_parameters()
    bias_start, bias_step = stride_parameters()
    stop_start, stop_step = stride_parameters()
    share_bias = (_uniform((n_reqs,), generator, device) < 0.5) & use_allowed
    share_stop = _uniform((n_reqs,), generator, device) < 0.3
    bias_start = torch.where(share_bias, allowed_start, bias_start)
    bias_step = torch.where(share_bias, allowed_step, bias_step)
    stop_start = torch.where(share_stop, allowed_start, stop_start)
    stop_step = torch.where(share_stop, allowed_step, stop_step)

    # Entries past num_logit_bias are never read; like the stale tail of a persistent upstream table they
    # hold ordinary draws rather than zeros, so the 1024-wide table is not 99% exact zeros.
    biases = torch.randn((n_reqs, bias_width), generator=generator, device=device) * 2.0

    rows = torch.arange(n_tokens, device=device)
    request = rows // SPEC_STRIDE
    base = _randint(16, 2048, (n_reqs,), generator, device)
    min_tokens = _randint(-6, 9, (n_reqs,), generator, device)
    slots = torch.randperm(n_reqs, generator=generator, device=device)
    return {
        "expanded_idx_mapping": slots[request].to(torch.int32),
        "pos": base[slots[request]] + rows % SPEC_STRIDE,
        "num_allowed_token_ids": n_allowed.to(torch.int32),
        "allowed_token_ids": _pad(_progression(allowed_start, allowed_step, allowed_width, vocab), n_allowed,
                                  allowed_width).to(torch.int32),
        "num_logit_bias": n_bias.to(torch.int32),
        "logit_bias_token_ids": _pad(_progression(bias_start, bias_step, bias_width, vocab), n_bias,
                                     bias_width).to(torch.int32),
        "logit_bias": biases,
        "min_lens": (base + min_tokens).to(torch.int32),
        "num_stop_token_ids": n_stop.to(torch.int32),
        "stop_token_ids": _pad(_progression(stop_start, stop_step, stop_width, vocab), n_stop,
                               stop_width).to(torch.int32),
    }
