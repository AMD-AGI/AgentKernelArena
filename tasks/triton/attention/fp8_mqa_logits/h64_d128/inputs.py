"""Window generator: ragged per-request causal windows over a concatenated KV buffer."""
import torch


def _windows(seq_len, seq_len_kv):
    """Two packed requests (one when too small to split); chunked-prefill causal windows.

    Request r owns query rows [q0, q1) and KV columns [k0, k1). Its row j sees
    [k0, k0 + clamp(kv_len - q_len + j + 1, 1, kv_len)).
    """
    if seq_len >= 2 and seq_len_kv >= 2:
        splits = [(0, seq_len // 2, 0, seq_len_kv // 2),
                  (seq_len // 2, seq_len, seq_len_kv // 2, seq_len_kv)]
    else:
        splits = [(0, seq_len, 0, seq_len_kv)]
    ks, ke = [], []
    for q0, q1, k0, k1 in splits:
        q_len, kv_len = q1 - q0, k1 - k0
        j = torch.arange(q_len, dtype=torch.int64)
        end = (kv_len - q_len + j + 1).clamp(1, kv_len) + k0
        ks.append(torch.full((q_len,), k0, dtype=torch.int64))
        ke.append(end)
    return torch.cat(ks).to(torch.int32), torch.cat(ke).to(torch.int32)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    ks, ke = _windows(int(axes["seq_len"]), int(axes["seq_len_kv"]))
    return {"cu_seqlen_ks": ks.to(device), "cu_seqlen_ke": ke.to(device)}
