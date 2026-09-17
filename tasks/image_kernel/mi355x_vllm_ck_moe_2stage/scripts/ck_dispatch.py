"""Select the declared CK stages while retaining AITER's MoE orchestration.

The protected selector applies equally to baseline and candidate. Image-specific
FlyDSL/ASM tuning defaults must not bypass the editable CK translation unit.
"""
from functools import partial
import importlib


def run_ck_moe(hidden, w1, w2, weights, ids, *, w1_scale, w2_scale,
               quant_type, activation, dtype, activation_dtype):
    aiter = importlib.import_module("aiter")
    moe = importlib.import_module("aiter.fused_moe")
    # This hook belongs to the pinned AITER runtime. Absence is an explicit
    # compatibility failure; there is no installed/alternate backend fallback.
    token = moe.get_padded_M(hidden.shape[0])
    experts, model_dim, inter_dim = moe.get_inter_dim(w1.shape, w2.shape)
    topk = ids.shape[1]
    block_m = ((64 if token > 32 else 16)
               if quant_type == aiter.QuantType.per_1x128
               else moe.get_block_size_M(token, topk, experts, inter_dim))
    split = (moe.get_ksplit(token, topk, experts, inter_dim, model_dim)
             if quant_type in (aiter.QuantType.per_1x128, aiter.QuantType.per_1x32)
             else 0)
    non_temporal = moe.use_nt(token, topk, experts)

    def select_ck(_selected):
        # Use the upstream untuned CK branch's stage constructors and shape
        # heuristics. Empty kernelName requests CK's own specialization dispatch.
        return moe.MOEMetadata(
            stage1=partial(moe.ck_moe_stage1, kernelName="", activation=activation,
                           quant_type=quant_type, dtype=dtype, splitk=int(split),
                           use_non_temporal_load=non_temporal),
            stage2=partial(aiter.ck_moe_stage2_fwd, kernelName="", activation=activation,
                           quant_type=quant_type, use_non_temporal_load=non_temporal),
            block_m=int(block_m), ksplit=int(split), run_1stage=False,
        )

    return moe._fused_moe_impl(
        hidden, w1, w2, weights, ids, w1_scale=w1_scale, w2_scale=w2_scale,
        activation=activation.value, quant_type=quant_type.value, dtype=dtype,
        _q_dtype_a=activation_dtype, _metadata_transform=select_ck,
    )
