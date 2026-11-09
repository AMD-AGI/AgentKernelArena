from .hstu_attention_wrapper import triton_hstu_attention_fwd

CAUSAL = True
MAX_ATTN_LEN = 0


def run(*, q, k, v, seq_offsets, num_targets, sort_by_length_indices, max_seq_len,
        contextual_seq_len, alpha, out):
    triton_hstu_attention_fwd(
        N=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=CAUSAL,
        num_targets=num_targets,
        max_attn_len=MAX_ATTN_LEN,
        contextual_seq_len=contextual_seq_len,
        sort_by_length_indices=sort_by_length_indices,
        out=out,
    )
