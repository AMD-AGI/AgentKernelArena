"""Operators whose floor is meaningful from bytes alone (Tier 1: memory-bound).

An operator not listed here and without a registered model gets tier ``none``: a
bytes-only floor of a compute-bound or data-dependent operator looks like large headroom
when the real limit is elsewhere. New operators therefore default to ``none`` until
someone classifies them.
"""
from __future__ import annotations

MEMORY_BOUND = frozenset("""
clamp_silu_and_mul fused_bias_leaky_relu gelu gelu_and_mul gelu_tanh gelu_tanh_and_mul sigmoid silu
silu_and_mul swigluoai_and_mul swiglustep_and_mul tanh
cp_lse_correct merge_attn_states
add copy flip transpose jagged_dense_broadcast_add
index_select_rows learned_position_add
convert_req_to_global_index fused_qk_rope_cache_mla fused_qkv_split_qk_norm_rope_cache
fused_reduce_qk_norm_rope_swa_write fused_rope_reshape_cache write_req_to_token_pool
biased_grouped_topk experts_combine routing_sigmoid_top1 silu_and_mul_fp8_group_quant topk_softplus
zero_expert_identity_combine
fused_add_layer_norm fused_add_rms_norm fused_dual_residual_rms_norm fused_rms_norm_fp8_group_quant
gated_residual_layer_norm_modulate l2_norm layer_norm layer_norm_wide log_softmax rms_norm
rms_norm_backward rms_norm_channel rms_norm_dynamic_quant rms_norm_gated rms_norm_smoothquant softmax
apply_staged_writes batch_memcpy pack_seq prepare_eagle_inputs prepare_pos_seq_lens
prepare_prefill_inputs unpack_seq
fused_qk_rope_concat fused_qkv_split_rope mrope qk_rmsnorm_rope rope_2d_fwd rope_fwd rope_thd_fwd
awq_dequantize dynamic_mxfp4_quant dynamic_mxfp8_quant per_tensor_quant_fp8 per_token_group_quant_fp8
per_token_group_quant_int8 per_token_quant_fp8 per_token_quant_int8 smoothquant_int8
cross_entropy_label_smoothing cross_entropy_loss kd_loss max_pool2d mean_dim row_max sum_dim
apply_grammar_bitmask apply_logit_bias apply_min_p apply_penalties apply_temperature
bincount_penalty_state count_nans token_logprobs token_ranks
topk
causal_conv1d_fwd causal_conv1d_fwd_split_qkv chunk_local_cumsum_vector fused_qkvzba_split kda_gate
ssd_state_passing
""".split())

# Tier 1 operators to audit before trusting G1: they may read only
# part of an input (padding positions, a subset of rows).
AUDIT_G1 = frozenset({"pack_seq", "unpack_seq", "apply_min_p", "token_logprobs", "apply_grammar_bitmask"})

# Inputs a reference reads only in part, per operator: a lower bound on the elements read (an axis
# expression), a fraction of the tensor (a generator-specific margin, noted in the floor), or None
# to count none of it. Applied by the generic byte rules.
PARTIAL_INPUTS: dict[str, dict[str, str | float | None]] = {
    # out_ids[begin:end] = all_token_ids[slot, computed:computed + length]: the scheduled tokens only.
    "prepare_prefill_inputs": {"all_token_ids": "total_tokens"},
    # angles[lo - begin:hi - begin]: positions below the longest sequence, at least T / num_seqs.
    "rope_thd_fwd": {"freqs": "ceil_div(n_tokens, num_seqs) * half_dim"},
    # pos_embedding.index_select(0, arange(seq_len)).
    "learned_position_add": {"pos_embedding": "seq_len * hidden_size"},
    # dA_cumsum[:, :, -1]: the last element of every chunk.
    "ssd_state_passing": {"dA_cumsum": "num_heads * n_chunks"},
    # tokens[:prefill_len] of each row; the row length is data-dependent.
    "bincount_penalty_state": {"all_token_ids": None},
    # v is read only for tokens with a valid slot, and any token may be padding: none is counted
    # without the generated slots (partial_data.fused_rope_reshape_cache counts them exactly).
    "fused_rope_reshape_cache": {"v": None},
    # rsigma = rsqrt(mean(x^2) + eps) is defined by the contract, so a kernel may recompute it from x.
    "rms_norm_backward": {"rsigma": None},
    # dense[b] is read only for non-empty sequences; any sequence may be empty, so none is counted.
    "jagged_dense_broadcast_add": {"dense": None},
}
