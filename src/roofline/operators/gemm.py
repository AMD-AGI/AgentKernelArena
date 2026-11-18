"""Matrix-multiply operators with shapes fixed by the case (Tier 2).

A product of an (M x K) and a (K x N) matrix needs 2*M*N*K FLOPs with the classical
algorithm; every model below counts only those products. Epilogues (bias, activation,
scaling, normalization, softmax) are ignored, which can only lower the floor. Where the
reference admits an algebraic shortcut a correct kernel may take, the model counts the
shortcut (``matmul_avgpool_gelu_scale_max`` pools the weights, ``b2b_gemm`` may reassociate, an
ungated feed-forward block without an activation may merge its weights).

Each product is charged at the matrix peak of its lowest-precision operand: a bfloat16 activation
against fp8 or fp4 weights is charged at the fp8 or fp4 peak. That is the faster peak, so the floor
stays a lower bound whichever format a kernel computes the product in. Where the operator itself
quantizes an operand before the product (a16wfp4 quantizes its activations to MXFP4), the model
gives that operand's bits after quantization. Operands of different widths are marked
``precision_substitution_sensitive``: the measured time may sit well above such a floor, and a
kernel that quantized further and still passed the tolerance could beat it. Float32 products are
marked too; ``floor.py`` charges them at the bf16 or fp16 part products a precision probe saw pass,
or at one bf16 product without a probe result (``precision_policy``), and the note stays because a
kernel may find a cheaper arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..dtypes import matrix_peak_key, operand_bits
from ..interface import CaseShapes
from . import Estimate, OperatorModel, register

INTEGER_DTYPES = ("int8", "int32", "int64", "int16")


@dataclass(frozen=True)
class MatMul:
    m: str
    n: str
    k: str
    # One name per operand; an entry "a|x" takes whichever of the names the task declares.
    operands: tuple[str, str]
    # Bits per value of an operand as it enters the product, where the dtype does not say: a packing
    # the dtype hides (AWQ int32 holds 8 int4), or a quantization the operator itself performs first
    # (bfloat16 activations quantized to MXFP4 or fp8 inside the operator).
    bits: tuple[tuple[str, int], ...] = ()


def _operand(shapes: CaseShapes, spec: str):
    names = {t.name for t in shapes.inputs}
    for name in spec.split("|"):
        if name in names:
            return shapes.tensor(name)
    raise KeyError(f"{shapes.task_id}: none of the operands {spec!r} is declared")


def add_product(shapes: CaseShapes, estimate: Estimate, product: MatMul, flops: int | None = None) -> None:
    """Charge one product (2*M*N*K, or ``flops``) at the peak of its lowest-precision operand."""
    overrides = dict(product.bits)
    operands = [_operand(shapes, spec) for spec in product.operands]
    # A uint8 operand holds packed fp4 codes or int8 codes; the packing is not always visible in the
    # shape (``hidden_bytes: hidden_size // 2``), so it is charged as 4-bit, the faster peak, which
    # keeps the floor a lower bound.
    bits = [overrides.get(t.name, 4 if t.dtype == "uint8" else operand_bits(t.dtype, t.packed_values_per_element))
            for t in operands]
    lowest = min(bits)
    at_lowest = [t for t, b in zip(operands, bits) if b == lowest]
    integer = all(t.dtype in INTEGER_DTYPES and t.name not in overrides for t in at_lowest)
    declared = [t for t in at_lowest if t.name not in overrides]
    half = lowest == 16 and bool(declared) and all(t.dtype == "float16" for t in declared)
    if flops is None:
        flops = 2 * shapes.value(product.m) * shapes.value(product.n) * shapes.value(product.k)
    estimate.add_matrix(matrix_peak_key(lowest, integer=integer, half=half), flops)
    # Mixed-precision operands, or fp32 products a kernel could emulate with bf16 products.
    if (len(set(bits)) > 1 or lowest >= 32) and "precision_substitution_sensitive" not in estimate.notes:
        estimate.notes.append("precision_substitution_sensitive")


def products(*specs: MatMul):
    def estimate(shapes: CaseShapes) -> Estimate:
        result = Estimate()
        for spec in specs:
            add_product(shapes, result, spec)
        return result
    return estimate


def mm(m: str, n: str, k: str, a: str, b: str, **bits: int) -> MatMul:
    return MatMul(m, n, k, (a, b), tuple(bits.items()))


def _b2b(shapes: CaseShapes) -> Estimate:
    """(A Bᵀ) C with A: rows x d, B, C: keys x d. A (Bᵀ C) is the same product in another order.

    With Bᵀ C rounded to the operands' 16-bit format the reordered result fails the evaluator; kept
    as two 16-bit parts it passes every case, so the second product of that order is charged twice.
    """
    rows, keys, dim = (shapes.value(n) for n in ("n_rows", "n_keys", "head_dim"))
    result = Estimate(notes=["cheaper_association_counted: two-part intermediate"])
    left_first = 2 * rows * keys * dim * 2
    right_first = 2 * keys * dim * dim + 2 * 2 * rows * dim * dim
    add_product(shapes, result, mm("n_rows", "head_dim", "n_keys", "a", "b"), min(left_first, right_first))
    return result


def _avgpool_linear(shapes: CaseShapes) -> Estimate:
    """Average pooling of a linear map's outputs is a linear map with pooled weights: summing each
    window of weight rows is vector work, and the product shrinks by the pool size.

    A kernel that takes this shortcut may fail the tolerance on some cases, but it can choose per
    case, and no per-case result is recorded, so the shortcut is counted on every case; the floor
    errs low. A recorded per-case check that the shortcut fails could raise it."""
    result = Estimate(notes=["pooled_weight_shortcut_counted"])
    add_product(shapes, result, mm("batch", "out_features // pool_size", "in_features", "x", "weight"))
    return result


SIMPLE = {
    # Plain and quantized GEMMs: out[M, N] = A[M, K] Wᵀ.
    "gemm_a16w16": [mm("m", "n", "k", "a|x", "b|w")],
    "gemm_f32": [mm("m", "n", "k", "a", "b")],
    "skinny_gemm_a16w16": [mm("tokens", "n_out", "k", "x", "weight")],
    "gemv_f32": [mm("m", "1", "k", "a", "x")],
    "gemm_a8w8": [mm("m", "n", "k", "a|x|xq", "w")],
    "gemm_a8w8_blockscale": [mm("m", "n", "k", "a", "b")],
    "gemm_a16w8_blockscale": [mm("m", "n", "k", "x", "w")],
    "gemm_a16wfp4": [mm("m", "n", "k", "x", "w", x=4)],  # the operator quantizes x to MXFP4
    "gemm_a8wfp4": [mm("m", "n", "k", "a", "w")],
    "gemm_afp4wfp4": [mm("m", "n", "k", "a", "w")],
    "gemm_afp8wfp8": [mm("m", "n", "k", "a", "w")],
    "gemm_mxfp8": [mm("m", "n", "k", "x", "w")],
    "gemm_awq_w4a16": [mm("m", "n", "k", "x", "qweight", qweight=4)],
    "gemm_a16w16_gated": [mm("m", "n", "k", "x", "w")],
    # Batched products.
    "batched_gemm_bf16": [mm("batch * m", "n", "k", "x", "w")],
    "batched_gemm_f32": [mm("batch * m", "n", "k", "a", "b")],
    "batched_gemm_a8w8": [mm("batch * m", "n", "k", "x", "w")],
    "batched_gemm_a8w8_group_prequant": [mm("batch * m", "n", "k", "x", "wq", x=8)],  # x quantized to fp8
    "batched_gemm_afp4wfp4": [mm("batch * m", "n", "k", "x", "w")],
    "batched_gemm_a16wfp4": [mm("batch * m", "n", "k", "x", "w", x=4)],  # x quantized to MXFP4
    "batched_vecmat": [mm("m", "n", "k", "a", "b")],
    "inner_prod": [mm("batch", "height * width", "channels", "feat_img", "feat_sound")],
    # GEMMs with fused prologues or epilogues: only the products count.
    "fused_gemm_a16w16_quant_x": [mm("m", "n", "k", "x", "w")],
    "fused_gemm_a8w8_blockscale_a16w16": [mm("m", "n_fp8", "k", "x_fp8", "w_fp8"),
                                          mm("m", "n_bf16", "k", "x_bf16", "w_bf16")],
    "fused_gemm_a8w8_blockscale_mul_add": [mm("m", "n", "k", "x", "w")],
    "fused_gemm_a8w8_blockscale_split_cat": [mm("m", "n", "k", "x", "w")],
    "fused_gemm_afp4wfp4_a16w16": [mm("m", "n_fp4", "k", "x_fp4", "w_fp4"),
                                   mm("m", "n_bf16", "k", "x_bf16", "w_bf16")],
    "fused_gemm_afp4wfp4_mul_add": [mm("m", "n", "k", "x", "w")],
    "fused_gemm_afp4wfp4_split_cat": [mm("m", "n", "k", "x", "w")],
    "matmul_div_gelu": [mm("batch", "out_features", "in_features", "x", "weight")],
    "matmul_gelu_softmax": [mm("batch", "out_features", "in_features", "x", "weight")],
    "matmul_maxpool_sum_scale": [mm("batch", "out_features", "in_features", "x", "weight")],
    "matmul_scale_residual": [mm("batch", "out_features", "in_features", "x", "weight")],
    "matmul_softmax": [mm("batch", "out_features", "in_features", "x", "weight")],
    "matmul_swish_bias_groupnorm": [mm("batch", "out_features", "in_features", "x", "weight")],
    "matmul_swish_scale": [mm("batch", "out_features", "in_features", "x", "weight")],
    "linear_log_softmax": [mm("n_tokens", "vocab_size", "hidden_size", "x", "weight")],
    "fused_moe_router_softcap": [mm("n_tokens", "n_experts", "hidden_size", "x", "router_weight")],
    # Two or more dependent products (a nonlinearity between them prevents merging).
    "ff_a16w16_fused_gated": [mm("n_tokens", "up_size", "hidden_size", "x", "w_up"),
                              mm("n_tokens", "hidden_size", "intermediate_size", "x", "w_down")],
    "fused_swiglu_ffn": [mm("m", "n", "k", "x", "w_gate"), mm("m", "n", "k", "x", "w_up")],
    "llama_ff_rmsnorm_swiglu": [mm("n_tokens", "ffn_size", "hidden_size", "x", "w1"),
                                mm("n_tokens", "ffn_size", "hidden_size", "x", "w3")],
    "ffn_relu_residual_layernorm": [mm("n_tokens", "d_hidden", "d_model", "x", "w1_weight"),
                                    mm("n_tokens", "d_model", "d_hidden", "x", "w2_weight")],
    "conv1x1_ffn_gelu": [mm("n_tokens", "filter_size", "hidden_size", "x", "conv_weight"),
                         mm("n_tokens", "hidden_size", "filter_size", "x", "linear_weight")],
    "feedforward_vstack_mlp": [mm("2 * n_rows", "hidden_size", "in_features", "x", "fc1_weight"),
                               mm("2 * n_rows", "1", "hidden_size", "x", "fc2_weight")],
    "gate_gru_selection": [mm("rows", "d_model", "d_concat", "x1", "reset_weight"),
                           mm("rows", "d_model", "d_concat", "x1", "update_weight"),
                           mm("rows", "d_model", "d_concat", "x1", "proposal_weight")],
}

def _mixed_cast(shapes: CaseShapes) -> Estimate:
    """out = cast(a) @ cast(b): the operator casts both operands to the output dtype before the product,
    so the product is defined in the output dtype whatever the operands' stored dtypes are. Where the
    first operand is stored narrower than the output (a cast up), the product is charged in that
    narrower dtype, which is what TraceLens's aten::mm charges: the wider peak is not a proven bound."""
    out, first = shapes.outputs[0], shapes.inputs[0]
    charged = min((out, first), key=lambda t: operand_bits(t.dtype))
    bits = operand_bits(charged.dtype)
    result = Estimate(notes=["operands_cast_to_output_dtype"])
    flops = 2 * shapes.value("m") * shapes.value("n") * shapes.value("k")
    result.add_matrix(matrix_peak_key(bits, half=charged.dtype == "float16"), flops)
    if bits >= 32:
        result.notes.append("precision_substitution_sensitive")
    return result


NO_ACTIVATION = "out = (x @ w_up^T) @ w_down^T"


def _ungated_ffn(shapes: CaseShapes) -> Estimate:
    """out = act(x W_upᵀ) W_downᵀ: two products. The activation is not an axis of the contract; the
    scenario whose description gives ``out = (x @ w_up^T) @ w_down^T`` has none, and there W_down W_up
    can be formed once. Rounded to bf16 the merged weight fails the evaluator; kept as two bf16 parts
    it passes every case, so the merged order is charged with its second product twice, and the
    cheaper of the two orders is counted."""
    tokens, hidden, inter = (shapes.value(n) for n in ("n_tokens", "hidden_size", "intermediate_size"))
    flops = 4 * tokens * hidden * inter
    result = Estimate()
    if NO_ACTIVATION in shapes.description:
        flops = min(flops, 2 * hidden * hidden * inter + 2 * 2 * tokens * hidden * hidden)
        result.notes.append("no_activation: merged-weight order considered (two-part merged weight)")
    add_product(shapes, result, mm("1", "1", "1", "x", "w_up"), flops)
    return result


def _grouped_gemm(shapes: CaseShapes) -> Estimate:
    """Group sizes sum to m, so 2*m*n*k FLOPs; a group with no rows needs no weights, so only one
    group's slice of rhs is counted without the generated group sizes. Of group_sizes a kernel may
    read only the entries up to the group of the last row, so it counts zero."""
    result = Estimate(notes=["bytes: one group's weights"])
    add_product(shapes, result, mm("m", "n", "k", "lhs", "rhs"))
    rhs = shapes.tensor("rhs")
    result.bytes_override = (rhs.nbytes // rhs.shape[0]
                             + sum(t.nbytes for t in shapes.inputs if t.name not in ("rhs", "group_sizes"))
                             + sum(t.nbytes for t in shapes.outputs))
    return result


register("ff_a16w16_ungated", OperatorModel("gemm/ff_a16w16_ungated", 2, _ungated_ffn))
register("gemm_mixed_cast", OperatorModel("gemm/gemm_mixed_cast", 1, _mixed_cast))
register("grouped_gemm", OperatorModel("gemm/grouped_gemm", 2, _grouped_gemm))
for _name, _specs in SIMPLE.items():
    register(_name, OperatorModel(f"gemm/{_name}", 1, products(*_specs)))
register("b2b_gemm", OperatorModel("gemm/b2b_gemm", 1, _b2b))
register("matmul_avgpool_gelu_scale_max", OperatorModel("gemm/matmul_avgpool_gelu_scale_max", 1, _avgpool_linear))
