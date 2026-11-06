"""float64 reference: softmax attention over each request's gathered KV blocks, per head."""
import torch

BLOCK_N = 16


def reference(*, q, k, v, kv_block_tables, batch_num_block_n, sm_scale):
    heads, rows, head_size = q.shape
    batch = batch_num_block_n.shape[0]
    q_per_request = rows // batch
    ends = batch_num_block_n.tolist()
    out = torch.empty(heads, rows, head_size, dtype=torch.float64, device=q.device)
    for head in range(heads):
        k_blocks = k[head].double().reshape(-1, BLOCK_N, head_size)
        v_blocks = v[head].double().reshape(-1, BLOCK_N, head_size)
        table = kv_block_tables[head].long()
        start = 0
        for request, end in enumerate(ends):
            ids = table[start:end]
            keys = k_blocks[ids].reshape(-1, head_size)
            values = v_blocks[ids].reshape(-1, head_size)
            queries = q[head, request * q_per_request:(request + 1) * q_per_request].double()
            probs = torch.softmax(queries @ keys.T * float(sm_scale), dim=-1)
            out[head, request * q_per_request:(request + 1) * q_per_request] = probs @ values
            start = end
    return out
