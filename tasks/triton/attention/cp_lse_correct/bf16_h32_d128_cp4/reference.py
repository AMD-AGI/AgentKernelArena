"""High-precision reference for the context-parallel LSE correction, used only for correctness."""
import torch


def reference(*, attn_out, lses, lse_idx):
    lse64 = lses.double()
    lse64 = torch.where(torch.isnan(lse64) | (lse64 == float("inf")),
                        torch.full_like(lse64, float("-inf")), lse64)
    lse_max = lse64.max(dim=0).values
    lse_max = torch.where(lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max)
    global_lse = torch.log(torch.exp(lse64 - lse_max).sum(dim=0)) + lse_max
    diff = lses[lse_idx].double() - global_lse
    diff = torch.where(torch.isnan(diff) | (diff == float("inf")), torch.full_like(diff, float("-inf")), diff)
    corrected = attn_out.double()
    corrected.mul_(torch.exp(diff).unsqueeze(-1))
    return {"out_corrected": corrected, "lse_final": global_lse}
