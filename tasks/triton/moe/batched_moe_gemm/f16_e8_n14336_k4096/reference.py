"""float64 batched MoE GEMM reference, used only for correctness."""
import torch


def reference(*, A, B, expert_num_tokens):
    n_experts, max_tokens, _ = A.shape
    out = torch.zeros((n_experts, max_tokens, B.shape[1]), dtype=torch.float64, device=A.device)
    counts = expert_num_tokens.tolist()
    for e in range(n_experts):
        n = counts[e]
        if n:
            out[e, :n] = A[e, :n].double() @ B[e].double().T
    return out
