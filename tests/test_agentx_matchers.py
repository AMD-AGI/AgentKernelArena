"""AgentX workload matchers: which census rows each operator family selects and which const axes it verifies.

DSV3 holds census rows copied from a DeepSeek-V3 FP8 TP8 capture (MI355X, SGLang with AITER); the other rows are
synthetic, shaped like the ops the matchers expect from AITER and SGLang.
"""
from __future__ import annotations

import pytest

from src.tools.agentx_workloads import matchers


def _row(op, family, phase, dims, dtypes, calls, us, kernels):
    return {"op": op, "family": family, "phase": phase, "dims": dims, "dtypes": dtypes, "calls": calls,
            "total_us": us, "kernels": kernels}


CK = "ck::ELi1ELi128ELi128ELi16ELi128ELi256"
W_FP8 = [[257, 512, 7168], [257, 7168, 256]]
DSV3 = [
    _row("aiter::fmoe_fp8_blockscale_g1u1", "activation", "decode",
         [[17, 7168], [17, 7168], *W_FP8, [8368], [8368], [262], [2], [17, 56], [257, 4, 56], [257, 56, 2]],
         ["BFloat16", "Float8_e4m3fn", "Float8_e4m3fn", "Float8_e4m3fn", "int", "float", "int", "int", "Scalar",
          "float", "float", "float"], 207242.7, 26773610.2, {"fmoe_bf16_blockscaleFp8_g1u1_novs_silu_1tg_32x256": 15428}),
    _row("aiter::fmoe_fp8_blockscale_g1u1", "activation", "outside_step",
         [[8192, 7168], [8192, 7168], *W_FP8, [81943], [81943], [2561], [2], [8192, 56], [257, 4, 56], [257, 56, 2]],
         ["BFloat16", "Float8_e4m3fn", "Float8_e4m3fn", "Float8_e4m3fn", "int", "float", "int", "int", "Scalar",
          "float", "float", "float"], 38810.3, 58055329.94, {"fmoe_bf16_blockscaleFp8_g1u1_novs_silu_1tg_32x256": 3828}),
    _row("aiter::fused_moe_", "moe", "decode", [[17, 7168], *W_FP8, [17, 9], [17, 9], [257, 4, 56], [257, 56, 2]],
         ["BFloat16", "Float8_e4m3fn", "Float8_e4m3fn", "float", "int", "Scalar", "Scalar", "Scalar", "float", "float"],
         207242.7, 3028601.62, {"_moe_sorting_small_kernel_distributed": 15428}),
    _row("aiter::fused_moe_", "moe", "decode", [[16, 7168], *W_FP8, [16, 9], [16, 9], [257, 4, 56], [257, 56, 2]],
         ["BFloat16", "Float8_e4m3fn", "Float8_e4m3fn", "float", "int", "Scalar", "Scalar", "Scalar", "float", "float"],
         2337.3, 34650.76, {"_moe_sorting_small_kernel_distributed": 174}),
    _row("aiter::ck_moe_stage1", "moe", "decode",
         [[16, 7168], *W_FP8, [4247], [266], [2], [16, 9, 256], [257, 4, 56], [16, 56]],
         ["Float8_e4m3fn", "Float8_e4m3fn", "Float8_e4m3fn", "int", "int", "int", "BFloat16", "Scalar", "float",
          "float"], 2337.3, 133849.76, {CK: 174}),
    _row("aiter::ck_moe_stage2", "moe", "decode",
         [[16, 9, 256], *W_FP8, [4247], [266], [2], [16, 7168], [257, 56, 2], [16, 9, 2], [4247]],
         ["Float8_e4m3fn", "Float8_e4m3fn", "Float8_e4m3fn", "int", "int", "int", "BFloat16", "Scalar", "float",
          "float", "Scalar", "float"], 2337.3, 66994.82, {CK: 174}),
    _row("aiter::moe_sorting_opus_fwd", "moe", "outside_step",
         [[8192, 9], [8192, 9], [81943], [81943], [2561], [2], [8192, 7168], [2106624]],
         ["int", "float", "int", "float", "int", "int", "BFloat16", "Scalar", "Scalar", "unsigned char"],
         155241.2, 1881829.11, {"opus_moe_sorting_entry": 15312}),
    _row("aiter::biased_grouped_topk_hip", "moe", "decode", [[17, 256], [256], [17, 8], [17, 8]],
         ["BFloat16", "BFloat16", "float", "int"], 207242.7, 872652.88, {"grouped_topk_opt_sort_kernel": 15428}),
    _row("aiter::biased_grouped_topk_hip", "moe", "outside_step", [[8192, 256], [256], [8192, 8], [8192, 8]],
         ["BFloat16", "BFloat16", "float", "int"], 38810.3, 301808.41, {"grouped_topk_opt_sort_kernel": 3828}),
    _row("aiter::add_rmsnorm", "norm", "decode", [[17, 7168], [17, 7168], [17, 7168], [17, 7168], [7168]],
         ["BFloat16"] * 5, 435924.2, 1798637.86, {"aiter::add_rmsnorm_quant_kernel": 32452}),
    _row("aiter::rmsnorm", "norm", "outside_step", [[640, 512], [640, 512], [512]], ["BFloat16"] * 3,
         618.5, 2858.47, {"aiter::add_rmsnorm_quant_kernel": 61}),
    _row("aiter::_fused_qk_rmsnorm_kernel", "norm", "decode",
         [[17, 1536], [1536], [17, 512], [512], [17, 1536], [17, 512]],
         ["BFloat16", "BFloat16", "Scalar", "BFloat16", "BFloat16", "Scalar", "BFloat16", "BFloat16"],
         217962.1, 901794.96, {"aiter::fused_qk_rmsnorm_kernel": 16226}),
    _row("aiter::fused_qk_rope_cat_and_cache_mla", "rope", "decode",
         [[17, 16, 512], [17, 16, 64], [17, 1, 512], [17, 1, 64], [1659882, 1, 576], [17], [17],
          [163840, 1, 1, 32], [163840, 1, 1, 32]],
         ["BFloat16"] * 5 + ["long int", "long int", "BFloat16", "BFloat16", "float"], 217962.1, 1006567.81,
         {"_fused_qk_rope_cat_and_cache_mla_kernel": 16226}),
    _row("aiter::dynamic_per_token_scaled_quant", "quant", "outside_step", [[8192, 7168], [458752, 128], [8192, 56]],
         ["Float8_e4m3fn", "BFloat16", "float"], 81635.5, 2565835.35,
         {"aiter::dynamic_per_group_scaled_quant_kernel": 8052}),
    _row("aiter::dynamic_per_token_scaled_quant", "quant", "decode", [[17, 7168], [952, 128], [17, 56]],
         ["Float8_e4m3fn", "BFloat16", "float"], 435924.2, 1774116.9,
         {"aiter::dynamic_per_group_scaled_quant_kernel": 32452}),
    _row("aiter::dynamic_per_token_scaled_quant", "quant", "decode", [[17, 2048], [272, 128], [17, 16]],
         ["Float8_e4m3fn", "BFloat16", "float"], 217962.1, 1047401.95,
         {"aiter::dynamic_per_group_scaled_quant_kernel": 16226}),
    _row("aiter::dynamic_per_token_scaled_quant", "quant", "decode", [[16, 9, 256], [288, 128], [16, 9, 2]],
         ["Float8_e4m3fn", "BFloat16", "float"], 2337.3, 10408.28,
         {"aiter::dynamic_per_group_scaled_quant_kernel": 174}),
    _row("", "other", "decode", [], [], 412148.0, 1701589.9, {"_fused_append_shared_experts_kernel": 30682}),
]


def _census(*rows):
    return {"ops": list(rows)}


def _var(lo=1, hi=262144):
    return {"kind": "var", "min": lo, "max": hi}


def _c(value):
    return {"kind": "const", "value": value}


def _tokens(match, axis="n_tokens"):
    return sorted((o.axes[axis], o.phase) for o in match.observations)


X_BF16 = {"x": {"kind": "tensor", "shape": ["n_tokens", "hidden_size"], "dtype": "bfloat16"}}


@pytest.mark.parametrize("operator, name", [
    ("dynamic_mxfp4_quant", "mxfp4_quant"), ("dynamic_mxfp8_quant", "mxfp8_quant"),
    ("per_token_group_quant_fp8", "quant"), ("per_token_quant_int8", "quant"),
    ("silu_and_mul_fp8_group_quant", "quant"), ("biased_grouped_topk", "moe_routing"),
    ("fused_moe_fp8_blockscale", "fused_moe"), ("fused_moe_mxfp4", "fused_moe"),
    ("causal_conv1d_fwd", "causal_conv1d_fwd"), ("gdn_chunk_fwd_h", "gdn_chunk_fwd_h"),
    ("gdn_fused_recurrent_decode", "gdn_fused_recurrent_decode"),
])
def test_registry_routes_the_new_operators(operator, name):
    assert matchers.find(operator)[0] == name


def test_dtype_names_compare_exactly():
    assert matchers.dtype_ok("int8", "signed char")
    assert not matchers.dtype_ok("int8", "unsigned char")
    assert matchers.dtype_ok("uint8", "c10::Float4_e2m1fn_x2")
    assert not matchers.dtype_ok("float8_e4m3fn", "c10::Float8_e4m3fnuz")


# --- Quantization ---------------------------------------------------------------------------------------------

MXFP4_ROWS = [  # AITER's dtype-aware group quant writing fp4x2 codes with e8m0 scales, and SGLang's Triton op
    _row("aiter::dynamic_per_group_scaled_quant", "quant", "prefill", [[4096, 3584], [4096, 7168], [4096, 224]],
         ["Float4_e2m1fn_x2", "BFloat16", "Float8_e8m0fnu", "Scalar"], 61.0, 900.0,
         {"aiter::dynamic_per_group_scaled_quant_kernel": 61}),
    _row("sglang::aiter_dynamic_mxfp4_quant", "quant", "decode", [[24, 7168]], ["BFloat16"], 610.0, 1200.0,
         {"_dynamic_mxfp4_quant_kernel": 610}),
]
MXFP8_ROW = _row("aiter::dynamic_per_group_scaled_quant", "quant", "prefill", [[2048, 7168], [2048, 7168], [2048, 224]],
                 ["Float8_e4m3fn", "BFloat16", "Float8_e8m0fnu", "Scalar"], 30.0, 300.0,
                 {"aiter::dynamic_per_group_scaled_quant_kernel": 30})
GROUP_AXES = {"n_tokens": _var(), "hidden_size": _c(7168), "num_groups": _c(56)}


def test_fp8_group_quant_keeps_the_width_and_group_count():
    fn = matchers.find("per_token_group_quant_fp8")[1]
    match = fn(GROUP_AXES, X_BF16, _census(*DSV3, *MXFP4_ROWS, MXFP8_ROW))
    # the 7168-wide 1x128 rows only: not the 2048-wide rows, the MoE requantization, the bf16 add_rmsnorm whose
    # kernel name contains "quant", nor the MX formats
    assert _tokens(match) == [(17, "decode"), (8192, "outside_step")]
    assert match.unchecked_const_axes == []
    assert fn(dict(GROUP_AXES, num_groups=_c(112)), X_BF16, _census(*DSV3)).observations == []


def test_mxfp4_and_fp8_group_quant_select_disjoint_rows():
    census = _census(*DSV3, *MXFP4_ROWS, MXFP8_ROW)
    mx = matchers.find("dynamic_mxfp4_quant")[1]({"n_tokens": _var(), "hidden_size": _c(7168)}, X_BF16, census)
    assert _tokens(mx) == [(24, "decode"), (4096, "prefill")]
    assert any("record no scale tensor" in n for n in mx.notes)  # the Triton op records only its input
    mx8 = matchers.find("dynamic_mxfp8_quant")[1]({"n_tokens": _var(), "hidden_size": _c(7168)}, X_BF16, census)
    assert _tokens(mx8) == [(2048, "prefill")]


def test_per_token_quant_rejects_group_scales():
    fn = matchers.find("per_token_quant_fp8")[1]
    axes = {"n_tokens": _var(), "hidden_size": _c(7168)}
    assert fn(axes, X_BF16, _census(*DSV3)).observations == []
    per_token = _row("aiter::dynamic_per_token_scaled_quant", "quant", "prefill", [[512, 7168], [512, 7168], [512, 1]],
                     ["Float8_e4m3fn", "BFloat16", "float"], 10.0, 100.0,
                     {"aiter::dynamic_per_token_scaled_quant_kernel": 10})
    assert _tokens(fn(axes, X_BF16, _census(*DSV3, per_token))) == [(512, "prefill")]


def test_generic_quant_needs_a_quantized_type():
    match = matchers.quant({"n_tokens": _var(), "hidden_size": _c(7168)}, X_BF16, _census(*DSV3))
    assert match.observations and not any("add_rmsnorm" in o.source for o in match.observations)


# --- Norm + RoPE ------------------------------------------------------------------------------------------------

QK_AXES = {"n_tokens": _var(1, 16384), "num_q_heads": _c(128), "head_dim": _c(512), "rope_dim": _c(64),
           "max_position": _c(65536)}
QK_INPUTS = {"q": {"kind": "tensor", "shape": ["n_tokens", "num_q_heads", "head_dim"]},
             "kv": {"kind": "tensor", "shape": ["n_tokens", "head_dim"]},
             "kv_weight": {"kind": "tensor", "shape": ["head_dim"], "role": "weight"}}


def _qk_row(tokens, heads):
    return _row("aiter::fused_qk_norm_rope_group_quant", "norm", "prefill",
                [[tokens, heads, 512], [tokens, 512], [512], [65536, 32], [65536, 32], [tokens]],
                ["BFloat16"] * 5 + ["long int"], 5.0, 50.0, {"fused_qk_norm_rope_group_quant_kernel": 5})


def test_qk_norm_needs_the_fused_norm_rope_op_and_the_head_count():
    # MLA's 512-wide kv_a_layernorm, the q_a/kv_a norm pair and the MLA RoPE are not this operator
    assert matchers.qk_norm(QK_AXES, QK_INPUTS, _census(*DSV3)).observations == []
    match = matchers.qk_norm(QK_AXES, QK_INPUTS, _census(*DSV3, _qk_row(1000, 128), _qk_row(64, 16)))
    assert _tokens(match) == [(1000, "prefill")]
    assert match.unchecked_const_axes == []
    no_table = _qk_row(200, 128)
    no_table["dims"] = [d for d in no_table["dims"] if d != [65536, 32]]
    assert matchers.qk_norm(QK_AXES, QK_INPUTS, _census(no_table)).unchecked_const_axes == ["rope_dim",
                                                                                             "max_position"]


# --- Routing and sorting ----------------------------------------------------------------------------------------

ROUTER_INPUTS = {"gating_output": {"kind": "tensor", "dtype": "float32"}}


def test_biased_grouped_topk_checks_experts_top_k_and_the_bias():
    fn = matchers.find("biased_grouped_topk")[1]
    axes = {"n_tokens": _var(), "n_experts": _c(256), "topk": _c(8)}
    unbiased = _row("aiter::grouped_topk", "moe", "decode", [[17, 256], [17, 8], [17, 8]],
                    ["BFloat16", "float", "int"], 10.0, 10.0, {"grouped_topk_kernel": 10})
    match = fn(axes, ROUTER_INPUTS, _census(*DSV3, unbiased))
    assert _tokens(match) == [(17, "decode"), (8192, "outside_step")]
    assert match.unchecked_const_axes == []
    assert any("scalar arguments" in n for n in match.notes)
    assert any("another dtype" in n for n in match.notes)  # served logits are bf16, the task's float32
    assert fn(dict(axes, topk=_c(9)), ROUTER_INPUTS, _census(*DSV3)).observations == []


def test_topk_softmax_finds_the_logits_among_the_inputs():
    row = _row("aiter::topk_softmax_asm", "moe", "decode", [[22, 4], [22, 4], [22, 4], [22, 128]],
               ["float", "int", "int", "BFloat16"], 10.0, 20.0, {"aiter::topksoftmax_4x128x4_bf16": 10})
    match = matchers.moe_routing({"n_tokens": _var(), "n_experts": _c(128), "topk": _c(4)},
                                 {"x": {"kind": "tensor", "dtype": "bfloat16"}}, _census(row))
    assert _tokens(match) == [(22, "decode")] and match.unchecked_const_axes == []


ROUNDED_SORT = _row("aiter::moe_sorting_opus_fwd", "moe", "prefill",  # newer AITER rounds L up to the block
                    [[16384, 4], [16384, 4], [81920], [81920], [640], [2], [16384, 3072], [2097920]],
                    ["int", "float", "int", "float", "int", "int", "BFloat16", "Scalar", "Scalar", "unsigned char"],
                    1152.0, 15000.0, {"opus_moe_sorting_entry": 1152})


def test_sorting_layout_reads_experts_and_block_size():
    assert matchers._sorting_layout(DSV3[6]) == {"tokens": 8192, "top_k": 9, "experts": 257, "block": 32,
                                                 "model_dim": 7168}
    assert matchers._sorting_layout(ROUNDED_SORT) == {"tokens": 16384, "top_k": 4, "experts": 128, "block": 128,
                                                      "model_dim": 3072}
    # sorting kernels launched by the fused MoE op: experts from its weights, no block size
    assert matchers._sorting_layout(DSV3[2]) == {"tokens": 17, "top_k": 9, "experts": 257, "block": None,
                                                 "model_dim": 7168}
    assert matchers._sorting_layout(DSV3[0]) is None  # the expert GEMM is not a sorting op


def test_moe_sorting_checks_experts_and_block_size():
    axes = {"n_tokens": _var(), "top_k": _c(8), "num_experts": _c(256), "block_size": _c(32),
            "model_dim": _c(7168)}
    assert matchers.moe_sorting(axes, {}, _census(*DSV3)).observations == []  # DSV3 sorts 257 experts, top 9
    dsv3 = dict(axes, top_k=_c(9), num_experts=_c(257))
    match = matchers.moe_sorting(dsv3, {}, _census(*DSV3))
    assert _tokens(match) == [(16, "decode"), (17, "decode"), (8192, "outside_step")]
    assert match.unchecked_const_axes == ["block_size"]  # the fused op's sorting kernels do not show it
    sort_only = matchers.moe_sorting(dsv3, {}, _census(DSV3[6]))
    assert sort_only.unchecked_const_axes == []
    assert matchers.moe_sorting(dict(dsv3, block_size=_c(64)), {}, _census(DSV3[6])).observations == []


# --- Fused MoE --------------------------------------------------------------------------------------------------

MOE_AXES = {"num_tokens": _var(1, 32768), "top_k": _c(9), "num_experts": _c(257), "hidden_size": _c(7168),
            "intermediate_size": _c(256)}
MOE_FP8 = {"hidden_states": {"kind": "tensor", "dtype": "bfloat16"},
           "w1": {"kind": "tensor", "dtype": "float8_e4m3fn", "role": "weight"}}


def test_fused_moe_sums_the_layer_ops_per_token_count():
    match = matchers.fused_moe(MOE_AXES, MOE_FP8, _census(*DSV3))
    assert _tokens(match, "num_tokens") == [(16, "decode"), (17, "decode"), (8192, "outside_step")]
    assert match.unchecked_const_axes == []
    by_tokens = {o.axes["num_tokens"]: o for o in match.observations}
    # two-stage CK path: sorting (fused op), both GEMM stages and the requantization between them
    assert by_tokens[16].us == pytest.approx(34650.76 + 133849.76 + 66994.82 + 10408.28)
    assert by_tokens[17].us == pytest.approx(26773610.2 + 3028601.62)  # one-stage asm kernel + sorting
    # calls are layer calls: the multi-kernel sorting op does not multiply them
    assert by_tokens[8192].calls == 38810.3
    assert by_tokens[8192].us == pytest.approx(58055329.94 + 1881829.11)
    assert all(o.pad_axis == "num_tokens" for o in match.observations)


@pytest.mark.parametrize("change", [{"num_experts": _c(256)}, {"intermediate_size": _c(512)}, {"top_k": _c(8)}])
def test_fused_moe_checks_the_const_axes(change):
    assert matchers.fused_moe(dict(MOE_AXES, **change), MOE_FP8, _census(*DSV3)).observations == []


def test_fused_moe_checks_the_weight_format():
    bf16 = dict(MOE_FP8, w1={"kind": "tensor", "dtype": "bfloat16", "role": "weight"})
    assert matchers.fused_moe(MOE_AXES, bf16, _census(*DSV3)).observations == []


def test_fused_moe_mxfp4_uses_packed_weights():
    axes = dict(MOE_AXES, hidden_size=_c(6144), intermediate_size=_c(512))
    fp4 = dict(MOE_FP8, w1={"kind": "tensor", "dtype": "uint8", "role": "weight"})
    row = _row("aiter::fused_moe_", "moe", "prefill",
               [[1024, 6144], [257, 1024, 3072], [257, 6144, 256], [1024, 9], [1024, 9], [257, 1024, 192],
                [257, 6144, 16]],
               ["BFloat16", "Float4_e2m1fn_x2", "Float4_e2m1fn_x2", "float", "int", "Scalar", "unsigned char",
                "unsigned char"], 120.0, 9000.0, {"moe1_a4w4": 60, "moe2_a4w4": 60})
    match = matchers.fused_moe(axes, fp4, _census(row))
    assert _tokens(match, "num_tokens") == [(1024, "prefill")]
    assert match.observations[0].calls == 60.0  # two kernels per layer call
    assert any("activation format" in n for n in match.notes)
    assert matchers.fused_moe(axes, MOE_FP8, _census(row)).observations == []


# --- State space ------------------------------------------------------------------------------------------------

CONV_AXES = {"n_tokens": _var(1, 131072), "dim": _c(8192), "key_dim": _c(2048), "value_dim": _c(4096),
             "width": _c(4)}


def test_causal_conv1d_reads_linked_shapes_or_the_step_tokens():
    linked = _row("sglang::causal_conv1d_fwd", "other", "prefill", [[8192, 300], [8192, 4], [8192], [2]],
                  ["BFloat16", "BFloat16", "BFloat16", "int"], 3.0, 30.0, {"_causal_conv1d_fwd_kernel": 3})
    stepped = dict(_row("", "other", "prefill", [], [], 2.0, 20.0, {"_causal_conv1d_fwd_kernel": 2}),
                   step_tokens=4096)
    bare = _row("", "other", "draft_prefill", [], [], 1.0, 1000.0, {"_causal_conv1d_fwd_kernel": 1})
    update = _row("", "other", "decode", [], [], 1.0, 10.0, {"_causal_conv1d_update_kernel": 1})
    match = matchers.causal_conv1d_fwd(CONV_AXES, {}, _census(linked, stepped, bare, update))
    assert _tokens(match) == [(300, "prefill"), (4096, "prefill")]
    # the step-count row verifies no shape
    assert match.unchecked_const_axes == ["dim", "key_dim", "value_dim", "width"]
    assert any("1.0 ms" in n and "not counted" in n for n in match.notes)
    assert matchers.causal_conv1d_fwd(CONV_AXES, {}, _census(linked)).unchecked_const_axes == ["key_dim",
                                                                                                "value_dim"]


GDN_AXES = {"num_k_heads": _c(16), "num_heads": _c(32), "head_dim_k": _c(128), "head_dim_v": _c(128)}


def test_gdn_chunk_state_pass_checks_heads_and_head_sizes():
    axes = dict(GDN_AXES, n_tokens=_var(1, 65536), chunk_size=_c(64))
    row = _row("sglang::chunk_gated_delta_rule", "other", "prefill",
               [[1, 2500, 16, 128], [1, 2500, 32, 128], [1, 2500, 32, 128], [1, 2500, 32]],
               ["BFloat16", "BFloat16", "BFloat16", "float"], 4.0, 40.0,
               {"chunk_gated_delta_rule_fwd_kernel_h_blockdim64": 4})
    match = matchers.gdn_chunk_fwd_h(axes, {}, _census(row))
    assert _tokens(match) == [(2500, "prefill")] and match.unchecked_const_axes == ["chunk_size"]
    assert matchers.gdn_chunk_fwd_h(dict(axes, num_heads=_c(64)), {}, _census(row)).observations == []
    # an op that also runs the other chunk kernels in the same row: its time is not the state pass's alone
    pipeline = dict(row, kernels={"chunk_gated_delta_rule_fwd_kernel_h_blockdim64": 4, "chunk_fwd_kernel_o": 4})
    assert matchers.gdn_chunk_fwd_h(axes, {}, _census(pipeline)).observations == []


def test_gdn_recurrent_decode_reads_the_packed_qkv_batch():
    axes = dict(GDN_AXES, batch=_var(1, 800))
    row = _row("sglang::gdn_packed_decode", "other", "decode", [[37, 8192], [37, 32], [37, 32], [32], [32]],
               ["BFloat16"] * 4 + ["float"], 5.0, 50.0, {"fused_recurrent_gated_delta_rule_packed_decode_kernel": 5})
    stepped = dict(_row("", "other", "decode", [], [], 2.0, 20.0,
                        {"fused_sigmoid_gating_delta_rule_update_kernel": 2}), step_batch=12)
    match = matchers.gdn_fused_recurrent_decode(axes, {}, _census(row, stepped))
    assert _tokens(match, "batch") == [(12, "decode"), (37, "decode")]
    assert all(o.pad_axis == "batch" for o in match.observations)


# --- RMSNorm ----------------------------------------------------------------------------------------------------

AITER_NORM = {"aiter::add_rmsnorm_quant_kernel": 1}  # AITER's kernel name for plain and residual-add RMSNorm alike
DSV3_NORMS = [
    _row("aiter::rmsnorm", "norm", "decode", [[17, 7168], [17, 7168], [7168]], ["BFloat16"] * 3 + ["Scalar"] * 2,
         3573.1, 14813.63, {"aiter::add_rmsnorm_quant_kernel": 266}),
    _row("aiter::rmsnorm", "norm", "outside_step", [[8192, 7168], [8192, 7168], [7168]],
         ["BFloat16"] * 3 + ["Scalar"] * 2, 669.1, 24616.87, {"aiter::add_rmsnorm_quant_kernel": 66}),
    _row("aiter::rmsnorm", "norm", "outside_step", [[8192, 1536], [8192, 1536], [1536]],
         ["BFloat16"] * 3 + ["Scalar"] * 2, 40817.7, 405086.4, {"aiter::add_rmsnorm_quant_kernel": 4026}),
]
QWEN3_32B_NORMS = [  # Qwen3-32B TP8: hidden 5120 and the per-head q/k norm over head dim 128
    _row("aiter::add_rmsnorm", "norm", "decode", [[25, 5120], [25, 5120], [25, 5120], [25, 5120], [5120]],
         ["BFloat16"] * 5 + ["Scalar"] * 2, 405480.4, 1765090.22, {"aiter::add_rmsnorm_quant_kernel": 8960}),
    _row("aiter::rmsnorm", "norm", "decode", [[25, 5120], [25, 5120], [5120]], ["BFloat16"] * 3 + ["Scalar"] * 2,
         3167.8, 14585.35, {"aiter::add_rmsnorm_quant_kernel": 70}),
    _row("aiter::rmsnorm", "norm", "prefill", [[8192, 5120], [8192, 5120], [5120]],
         ["BFloat16"] * 3 + ["Scalar"] * 2, 560.2, 14458.31, {"aiter::add_rmsnorm_quant_kernel": 27}),
    _row("aiter::rmsnorm", "norm", "decode", [[25, 128], [25, 128], [128]], ["BFloat16"] * 3 + ["Scalar"] * 2,
         202740.2, 858459.19, {"aiter::add_rmsnorm_quant_kernel": 4480}),
    _row("aiter::rmsnorm", "norm", "prefill", [[65536, 128], [65536, 128], [128]],
         ["BFloat16"] * 3 + ["Scalar"] * 2, 35856.0, 624898.02, {"aiter::add_rmsnorm_quant_kernel": 1728}),
]
NORM_QUANT = _row("aiter::add_rmsnorm_quant", "norm", "prefill",  # residual add + RMSNorm writing FP8 codes
                  [[8192, 7168], [8192, 7168], [8192, 7168], [7168], [8192, 56]],
                  ["Float8_e4m3fn", "BFloat16", "BFloat16", "BFloat16", "float"], 10.0, 100.0, AITER_NORM)
X_NORM = {"x": {"kind": "tensor", "shape": ["n_tokens", "hidden_size"], "dtype": "bfloat16"},
          "weight": {"kind": "tensor", "shape": ["hidden_size"], "dtype": "bfloat16", "role": "weight"}}


def _norm_axes(hidden):
    return {"n_tokens": _var(), "hidden_size": _c(hidden)}


def test_rms_norm_matches_aiter_rmsnorm_by_op_name_and_width():
    census = _census(*DSV3, *DSV3_NORMS, NORM_QUANT)
    match = matchers.rms_norm(_norm_axes(7168), X_NORM, census)
    # plain aiter::rmsnorm rows of width 7168 only: not add_rmsnorm, the fused norm + FP8 quant or other widths
    assert _tokens(match) == [(17, "decode"), (8192, "outside_step")]
    assert match.unchecked_const_axes == []
    assert _tokens(matchers.rms_norm(_norm_axes(512), X_NORM, census)) == [(640, "outside_step")]
    # the fused q_a/kv_a norm pair is not a plain RMSNorm of either width
    assert matchers.rms_norm(_norm_axes(1536), X_NORM, _census(DSV3[11])).observations == []


def test_rms_norm_widths_of_qwen3():
    census = _census(*QWEN3_32B_NORMS)
    assert _tokens(matchers.rms_norm(_norm_axes(5120), X_NORM, census)) == [(25, "decode"), (8192, "prefill")]
    assert _tokens(matchers.rms_norm(_norm_axes(128), X_NORM, census)) == [(25, "decode"), (65536, "prefill")]
    fp16 = {"x": dict(X_NORM["x"], dtype="float16")}
    assert matchers.rms_norm(_norm_axes(5120), fp16, census).observations == []


def test_fused_add_rms_norm_takes_only_the_residual_add_ops():
    census = _census(*DSV3, *DSV3_NORMS, *QWEN3_32B_NORMS, NORM_QUANT)
    assert _tokens(matchers.fused_add_rms_norm(_norm_axes(5120), X_NORM, census)) == [(25, "decode")]
    assert _tokens(matchers.fused_add_rms_norm(_norm_axes(7168), X_NORM, census)) == [(17, "decode")]


# --- Gated delta rule in a speculative Qwen3-Next capture (rows copied from a TP1 census) -----------------------

CHUNK_KERNELS = {"l2norm_fwd_kernel": 72, "chunk_local_cumsum_scalar_kernel": 36,
                 "chunk_gated_delta_rule_fwd_kkt_solve_kernel": 36, "recompute_w_u_fwd_kernel": 36}
CHUNK_TYPES = ["BFloat16", "BFloat16", "BFloat16", "float", "float", "Scalar", "BFloat16", "int", "int"]
QWEN3_NEXT = [
    dict(_row("ChunkGatedDeltaRuleFunction", "other", "prefill",
              [[1, 4096, 16, 128], [1, 4096, 16, 128], [1, 4096, 32, 128], [1, 4096, 32], [1, 4096, 32],
               [2307, 32, 128, 128], [2], [3]], CHUNK_TYPES, 427.5, 16484.72, CHUNK_KERNELS), op_calls=61.1),
    dict(_row("ChunkGatedDeltaRuleFunction", "other", "prefill",
              [[1, 2112, 16, 128], [1, 2112, 16, 128], [1, 2112, 32, 128], [1, 2112, 32], [1, 2112, 32],
               [2329, 32, 128, 128], [1], [2]], CHUNK_TYPES, 427.5, 10680.59, CHUNK_KERNELS), op_calls=61.1),
    dict(_row("", "state_space", "verify", [], [], 9485.2, 290639.73, {"gdn_replayssm_spec_circular_kernel": 3816}),
         op_calls=9485.2, step_batch=27),
    dict(_row("", "state_space", "verify", [], [], 179.0, 5377.17, {"gdn_replayssm_spec_circular_kernel": 72}),
         op_calls=179.0, step_batch=24),
    dict(_row("", "state_space", "outside_step", [], [], 709.0, 409370.34,
              {"gdn_replayssm_compact_commit_kernel": 348}), op_calls=709.0),
]
# the chunk-pipeline rows list four kernels; the state pass is in the census's per-kernel totals (half, here)
QWEN3_NEXT_KERNELS = [{"name": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
                       "short_name": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "family": "state_space",
                       "calls": 122.2, "total_us": (16484.72 + 10680.59) / 2}]


def test_gdn_chunk_state_pass_counts_its_share_of_the_chunk_pipeline_op():
    axes = dict(GDN_AXES, n_tokens=_var(1, 65536), chunk_size=_c(64))
    census = dict(_census(*QWEN3_NEXT), kernels=QWEN3_NEXT_KERNELS)
    match = matchers.gdn_chunk_fwd_h(axes, {}, census)
    assert _tokens(match) == [(2112, "prefill"), (4096, "prefill")]
    by_tokens = {o.axes["n_tokens"]: o for o in match.observations}
    assert by_tokens[4096].us == pytest.approx(16484.72 / 2) and by_tokens[4096].calls == 61.1
    assert match.unchecked_const_axes == ["chunk_size"]
    assert any("50.0%" in n for n in match.notes)
    # without the per-kernel totals the pipeline's share is unknown: its rows are not used
    assert matchers.gdn_chunk_fwd_h(axes, {}, _census(*QWEN3_NEXT)).observations == []


def test_gdn_recurrent_update_of_speculative_verify_steps():
    axes = dict(GDN_AXES, batch=_var(1, 800))
    census = dict(_census(*QWEN3_NEXT), deployment={"speculative_num_draft_tokens": 4})
    match = matchers.gdn_fused_recurrent_decode(axes, {}, census)
    # the replay-SSM verify kernel by the step's sequence count; the post-acceptance commit is not the update
    assert _tokens(match, "batch") == [(24, "verify"), (27, "verify")]
    assert match.unchecked_const_axes == ["num_k_heads", "num_heads", "head_dim_k", "head_dim_v"]
    assert any("4 draft tokens" in n for n in match.notes)


def _hooked_replay(tokens, k_heads=16, us=300.0):
    """The verify update run inside a shape-hook scope: q/k [T, Hk, K], v [T, H, V], T = sequences x draft tokens."""
    return dict(_row("pyhook::sglang.gdn_replayssm_spec_decode", "state_space", "verify",
                     [[tokens, k_heads, 128], [tokens, k_heads, 128], [tokens, 32, 128], [tokens, 32], [tokens, 32],
                      [2307, 32, 128, 128], [tokens // 4 or 1]],
                     ["BFloat16", "BFloat16", "BFloat16", "float", "float", "BFloat16", "int"], 10.0, us,
                     {"gdn_replayssm_spec_circular_kernel": 10}), op_calls=10.0)


def test_gdn_recurrent_update_reads_the_shape_hooked_verify_op():
    axes = dict(GDN_AXES, batch=_var(1, 800))
    census = dict(_census(_hooked_replay(48), _hooked_replay(48, k_heads=8), *QWEN3_NEXT[2:]),
                  deployment={"speculative_num_draft_tokens": 4})
    match = matchers.gdn_fused_recurrent_decode(axes, {}, census)
    # hooked rows: 48 tokens / 4 draft tokens = 12 sequences (the 8-head row is another layer); unlinked rows by
    # the step's batch
    assert _tokens(match, "batch") == [(12, "verify"), (24, "verify"), (27, "verify")]
    hooked = dict(census, ops=[_hooked_replay(48)])
    assert matchers.gdn_fused_recurrent_decode(axes, {}, hooked).unchecked_const_axes == []
    # without the draft-token count the hooked row counts tokens, with a note
    unknown = matchers.gdn_fused_recurrent_decode(axes, {}, _census(_hooked_replay(50, us=2000.0)))
    assert _tokens(unknown, "batch") == [(50, "verify")]
    assert any("2.0 ms" in n and "count tokens" in n for n in unknown.notes)


# --- DeepSeek-V4 sparse attention, indexer and top-k (shape-hooked rows of a V4-Flash TP4 speculative capture) ----

SPARSE_MLA = "_sparse_mla_BLOCK_M_16_BLOCK_K_64_HEAD_SIZE_512_SPLIT_K_0_MAIN_FMT_FP8_DSV4_MLA_ROPE_SEPARATE_0"
PA_TYPES = ["BFloat16", "unsigned char", "int", "int", "float", "Scalar"]
V4 = {
    "verify_c128": dict(_row("pyhook::aiter.pa_decode_sparse", "rope", "verify",
                             [[114, 16, 512], [7929, 256, 584], [14592], [115], [16], [79413, 2, 584], [240768],
                              [115]], PA_TYPES, 95384.6, 4006153.2, {SPARSE_MLA: 2140}), op_calls=95384.6),
    "verify_c4": dict(_row("pyhook::aiter.pa_decode_sparse", "rope", "verify",
                           [[114, 16, 512], [7929, 256, 584], [14592], [115], [16], [79289, 64, 584], [58368],
                            [115]], PA_TYPES, 100153.8, 1882891.4, {SPARSE_MLA: 2247}), op_calls=100153.8),
    "verify_swa": dict(_row("pyhook::aiter.pa_decode_sparse", "rope", "verify",
                            [[114, 16, 512], [7929, 256, 584], [14592], [115], [16]], PA_TYPES, 9538.5, 100089.73,
                            {SPARSE_MLA: 214}), op_calls=9538.5),
    "draft_swa": dict(_row("pyhook::aiter.pa_decode_sparse", "rope", "draft",
                           [[95, 64, 512], [7929, 256, 584], [12160], [96], [64]], PA_TYPES, 14307.7, 203744.71,
                           {SPARSE_MLA.replace("M_16", "M_32"): 321}), op_calls=14307.7),
    "mqa_verify": dict(_row("pyhook::sglang.rowgroup_paged_mqa_logits", "sampling", "verify",
                            [[114, 64, 64], [114, 1, 4, 16, 4], [79289, 1, 4, 64, 16], [79289, 1, 4, 64], [114, 64],
                             [19, 1032], [20], [114]],
                            ["Float4_e2m1fn_x2", "unsigned char", "Float4_e2m1fn_x2", "unsigned char", "BFloat16",
                             "int", "int", "int"], 100153.8, 3785813.6, {"pa_mqa_logits_fp4_rowgroup_kernel_0": 2247}),
                       op_calls=100153.8),
    "mqa_prefill": dict(_row("pyhook::sglang.aiter_fp4_paged_mqa_logits", "sampling", "prefill",
                             [[2944, 64, 64], [2944, 1, 4, 16, 4], [79289, 1, 4, 64, 16], [79289, 1, 4, 64],
                              [2944, 64], [2944, 967], [2944]],
                             ["Float4_e2m1fn_x2", "unsigned char", "Float4_e2m1fn_x2", "unsigned char", "BFloat16",
                              "int", "int"], 161.3, 177413.9, {"pa_mqa_logits_fp4_prefill_kernel_0": 21}),
                        op_calls=161.3),
    "topk_verify": dict(_row("pyhook::sglang.topk_transform_paged_v2", "sampling", "verify",
                             [[114, 65792], [114], [114, 1026], [114, 512], [115, 2]],
                             ["float", "int", "int", "int", "Scalar", "int"], 100153.8, 1682583.8,
                             {"topk_main_kernel": 2247}), op_calls=100153.8),
    "topk_prefill": dict(_row("pyhook::sglang.topk_transform_paged_v2", "sampling", "prefill",
                              [[192, 20736], [192], [192, 322], [192, 512], [193, 2]],
                              ["float", "int", "int", "int", "Scalar", "int"], 161.3, 3000.0,
                              {"topk_main_kernel": 21}), op_calls=161.3),
    "prefill_opus": dict(_row("aiter::pa_sparse_prefill_gfx950_opus_fwd", "attention", "prefill",
                              [[10496, 16, 512], [10750, 512], [1343488], [10497], [1, 512], [1], [10497], [16],
                               [10496, 16, 512]],
                              ["BFloat16", "BFloat16", "int", "int", "BFloat16", "int", "int", "float", "BFloat16"],
                              15.4, 3307.9, {"opus_mla_v4_prefill_a16w16_16mx1_16nx4_kernel": 2}), op_calls=15.4),
    "qk_store": dict(_row("pyhook::sglang.fused_qk_norm_rope_swa_store", "rope", "verify",
                          [[114, 8192], [114, 512], [512], [1048576, 1, 1, 32], [1048576, 1, 1, 32], [114],
                           [7929, 149760], [114]],
                          ["BFloat16"] * 5 + ["long int", "unsigned char", "int"], 205076.8, 1374014.6,
                          {"_fused_qk_norm_rope_store_kernel": 4601}), op_calls=205076.8),
}
SPARSE_DECODE_AXES = {"batch": _var(1, 16384), "query_length": _c(1), "heads": _c(64), "head_dim": _c(512),
                      "pages": _c(2019), "page_size": _c(256), "one": _c(1), "packed_width": _c(584),
                      "topk": _c(128)}


def test_sparse_mla_decode_takes_rows_with_the_task_heads_and_pools():
    match = matchers.flash_mla_sparse_decode_fp8(SPARSE_DECODE_AXES, {}, _census(*V4.values()))
    # the target's verify rows have 16 heads per TP4 rank; the draft model's window-only attention has 64
    assert _tokens(match, "batch") == [(95, "draft")]
    assert match.unchecked_const_axes == ["pages"]  # a 7929-page pool, not 2019: it only bounds the indices
    assert any("heads 16 (task 64); compressed pool present" in n for n in match.notes)


def test_sparse_mla_decode_with_a_compressed_pool_needs_the_context_length():
    axes = dict(SPARSE_DECODE_AXES, heads=_c(16), ctx=_var(1, 1048576), extra_page_size=_c(64),
                extra_topk=_c(512))
    del axes["pages"]
    census = _census(*V4.values())
    assert matchers.flash_mla_sparse_decode_fp8(axes, {}, census).observations == []
    with_kv = dict(V4["verify_c4"], kv_tokens_per_seq=40960)
    match = matchers.flash_mla_sparse_decode_fp8(axes, {}, _census(with_kv, V4["verify_c128"]))
    assert [o.axes for o in match.observations] == [{"batch": 114, "ctx": 40960}]
    assert any("extra_page_size 2 (task 64); extra_topk 2112 (task 512)" in n for n in match.notes)


MQA_AXES = {"batch_size": _var(1, 256), "next_n": _var(1, 2), "ctx": _var(128, 131072), "num_heads": _c(64),
            "head_size": _c(128), "block_size": _c(64), "max_seq_len": _c(131072)}


def test_paged_mqa_logits_reads_batch_and_tokens_per_sequence_from_the_block_table():
    census = _census(V4["mqa_verify"], V4["mqa_prefill"])
    match = matchers.paged_mqa_logits(MQA_AXES, {}, census)
    assert match.observations == []
    assert any("max_seq_len 65792 (task 131072); next_n 6 (task 1..2)" in n for n in match.notes)
    assert any("prefill indexer" in n for n in match.notes)
    served = dict(MQA_AXES, next_n=_var(1, 8), max_seq_len=_c(65792))
    assert matchers.paged_mqa_logits(served, {}, census).observations == []  # no context length recorded
    with_kv = dict(V4["mqa_verify"], kv_tokens_per_seq=40960)
    match = matchers.paged_mqa_logits(served, {}, _census(with_kv))
    assert [o.axes for o in match.observations] == [{"batch_size": 19, "next_n": 6, "ctx": 40960}]
    assert match.unchecked_const_axes == []


def test_topk_transform_paged_checks_the_score_width():
    axes = {"batch": _var(1, 2048), "width": _c(262208), "num_pages": _c(4097), "topk": _c(512)}
    census = _census(V4["topk_verify"], V4["topk_prefill"])
    match = matchers.topk_transform_paged(axes, {}, census)
    assert match.observations == []
    assert any("verify rows not used: width 65792 (task 262208); num_pages 1026 (task 4097)" in n
               for n in match.notes)
    served = dict(axes, width=_c(65792), num_pages=_c(1026))
    assert _tokens(matchers.topk_transform_paged(served, {}, census), "batch") == [(114, "verify")]


def test_sparse_mla_prefill_checks_heads_and_top_k():
    axes = {"n_queries": _var(1, 16384), "num_heads": _c(64), "head_dim": _c(512), "num_kv": _c(8192),
            "topk": _c(512)}
    census = _census(V4["prefill_opus"], V4["draft_swa"])
    match = matchers.sparse_mla_prefill(axes, {}, census)
    assert match.observations == []
    assert any("num_heads 16 (task 64); topk 128 (task 512)" in n for n in match.notes)
    tp4 = dict(axes, num_heads=_c(16), topk=_c(128))
    match = matchers.sparse_mla_prefill(tp4, {}, census)
    assert _tokens(match, "n_queries") == [(10496, "prefill")] and match.unchecked_const_axes == ["num_kv"]


def test_qk_norm_reads_the_hooked_fused_store_with_a_flattened_q():
    inputs = dict(QK_INPUTS)
    h16 = dict(QK_AXES, num_q_heads=_c(16))
    match = matchers.qk_norm(h16, inputs, _census(V4["qk_store"]))
    assert _tokens(match) == [(114, "verify")]
    assert match.unchecked_const_axes == ["max_position"]  # a 1M-row cos/sin table, not 65536
    match = matchers.qk_norm(QK_AXES, inputs, _census(V4["qk_store"]))
    assert match.observations == [] and any("num_q_heads 16 (task 128)" in n for n in match.notes)
