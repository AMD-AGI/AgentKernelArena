"""float64 reference for merging two partial attention states by log-sum-exp."""
import torch


def reference(*, prefix_output, prefix_lse, suffix_output, suffix_lse):
    num_tokens, num_heads, _ = prefix_output.shape
    output = torch.empty(prefix_output.shape, dtype=torch.float64, device=prefix_output.device)
    output_lse = torch.empty(num_heads, num_tokens, dtype=torch.float64, device=prefix_output.device)
    minus_inf = float("-inf")
    chunk = 16384
    for start in range(0, num_tokens, chunk):
        stop = min(num_tokens, start + chunk)
        p = prefix_lse[:, start:stop].double()
        s = suffix_lse[:, start:stop].double()
        p = torch.where(p == float("inf"), torch.full_like(p, minus_inf), p)
        s = torch.where(s == float("inf"), torch.full_like(s, minus_inf), s)
        peak = torch.maximum(p, s)
        w_p = torch.exp(p - peak)
        w_s = torch.exp(s - peak)
        total = w_p + w_s
        scale_p = (w_p / total).T.unsqueeze(-1)
        scale_s = (w_s / total).T.unsqueeze(-1)
        output[start:stop] = (prefix_output[start:stop].double() * scale_p
                              + suffix_output[start:stop].double() * scale_s)
        output_lse[:, start:stop] = torch.log(total) + peak
    return output, output_lse
