"""Hand-computed byte and FLOP counts of the roofline floor models (CPU only)."""
from __future__ import annotations

import itertools
import random

import pytest

from src.roofline.arch import Peaks, load_peaks
from src.roofline.bytes_model import generic_bytes
from src.roofline.dtypes import BYTES_PER_ELEMENT, matrix_peak_key, tensor_bytes
from src.roofline.interface import CaseShapes, Tensor
from src.roofline.operators import model_for
from src.roofline.operators.attention import causal_pairs, window_pairs
from src.roofline.operators.conv import Conv, conv_flops, in_bounds_taps, stride_and_padding
from src.task_contract import DTYPES, case_symbols, evaluate_dimension


def tensor(name, shape, dtype="bfloat16", direction="input", role="activation", expressions=None, mutable=False):
    expressions = tuple(expressions) if expressions is not None else tuple(shape)
    return Tensor(name, direction, role if direction == "input" else "output", dtype, tuple(shape), expressions,
                  mutable)


def shapes(operator, symbols, tensors, scalars=None, description=""):
    return CaseShapes("synthetic/" + operator, "case", operator, "test", dict(symbols), dict(symbols),
                      tuple(tensors), dict(scalars or {}), description)


def test_every_contract_dtype_has_a_storage_size():
    assert set(BYTES_PER_ELEMENT) == set(DTYPES)
    assert tensor_bytes((3, 5), "bfloat16") == 30
    assert tensor_bytes((4,), "float4_e2m1fn_x2") == 4


@pytest.mark.parametrize("bits,integer,key", [
    (4, False, "matrix_fp4"), (8, False, "matrix_fp8"), (8, True, "matrix_int8"),
    (16, False, "matrix_bf16"), (32, False, "matrix_fp32"), (64, False, "matrix_fp64"),
])
def test_matrix_peak_key_follows_the_operand_width(bits, integer, key):
    assert matrix_peak_key(bits, integer=integer) == key


def test_case_symbols_resolve_const_var_and_derived_axes():
    interface = {"axes": {"m": {"kind": "var"}, "n": {"kind": "const", "value": 7168}},
                 "derived_axes": {"half": "n // 2", "rows": "(m + 31) // 32 * 32"}}
    symbols = case_symbols(interface, {"m": 33})
    assert symbols == {"m": 33, "n": 7168, "half": 3584, "rows": 64}
    assert evaluate_dimension("ceil_div(m, 16) * 16", symbols) == 48


def test_g1_counts_every_input_and_output_once():
    # silu_and_mul, 4096 tokens of width 2 x 4096 in, 4096 out, bf16.
    case = shapes("silu_and_mul", {"n_tokens": 4096},
                  [tensor("x", (4096, 8192)), tensor("out", (4096, 4096), direction="output")])
    counted = generic_bytes(case)
    assert (counted.rule, counted.bytes) == ("G1", 100663296)
    assert counted.bytes / 8000e9 * 1e6 == pytest.approx(12.582912)


def test_g1_matches_tracelens_gemm_bytes():
    # TraceLens GEMM.bytes_func(4096, 4096, 2048, False, 2, 2, 2, 2) == 67108864.
    case = shapes("gemm_a16w16", {"m": 4096, "n": 4096, "k": 2048},
                  [tensor("a", (4096, 2048)), tensor("b", (4096, 2048), role="weight"),
                   tensor("out", (4096, 4096), direction="output")])
    assert generic_bytes(case).bytes == 67108864


def test_g2_drops_weight_state_index_and_mutable_tensors():
    case = shapes("paged", {}, [
        tensor("q", (10,)), tensor("cache", (1000,), role="state", mutable=True),
        tensor("table", (50,), dtype="int32", role="index"), tensor("w", (300,), role="weight"),
        tensor("meta", (4,), dtype="int32", role="metadata"), tensor("out", (10,), direction="output")])
    counted = generic_bytes(case)
    assert counted.rule == "G2"
    assert counted.bytes == 20 + 16 + 20


def test_rounded_up_inputs_are_excluded_but_outputs_are_not():
    case = shapes("gemm_afp4wfp4", {"m": 33, "k": 64}, [
        tensor("a_scales", (64, 2), dtype="uint8", expressions=("(m + 31) // 32 * 32", "k // 32")),
        tensor("out", (64,), direction="output", expressions=("max(m, 64)",))])
    counted = generic_bytes(case)
    assert counted.bytes == 128
    assert counted.notes == ["padded_input_excluded:a_scales"]


def test_packed_fp4_operands_use_the_fp4_peak():
    case = shapes("gemm_afp4wfp4", {"m": 16, "n": 32, "k": 64}, [
        tensor("a", (16, 32), dtype="uint8", expressions=("m", "k // 2")),
        tensor("w", (32, 32), dtype="uint8", role="weight", expressions=("n", "k // 2")),
        tensor("out", (16, 32), direction="output")])
    estimate = model_for("gemm_afp4wfp4").estimate(case)
    assert estimate.flops_matrix == {"matrix_fp4": 2 * 16 * 32 * 64}
    assert estimate.notes == []


def test_a16wfp4_is_charged_at_fp4_because_the_operator_quantizes_its_activations():
    case = shapes("gemm_a16wfp4", {"m": 8, "n": 16, "k": 64}, [
        tensor("x", (8, 64)), tensor("w", (16, 32), dtype="uint8", role="weight", expressions=("n", "k // 2")),
        tensor("out", (8, 16), direction="output")])
    estimate = model_for("gemm_a16wfp4").estimate(case)
    assert estimate.flops_matrix == {"matrix_fp4": 2 * 8 * 16 * 64}
    assert estimate.notes == []


def test_mixed_precision_gemm_is_charged_at_its_lowest_precision_operand_and_flagged():
    # bfloat16 activations against fp8 weights: charged at the fp8 peak, the faster one.
    case = shapes("gemm_a16w8_blockscale", {"m": 8, "n": 16, "k": 64}, [
        tensor("x", (8, 64)), tensor("w", (16, 64), dtype="float8_e4m3fn", role="weight"),
        tensor("out", (8, 16), direction="output")])
    estimate = model_for("gemm_a16w8_blockscale").estimate(case)
    assert estimate.flops_matrix == {"matrix_fp8": 2 * 8 * 16 * 64}
    assert estimate.notes == ["precision_substitution_sensitive"]
    # fp8 activations against packed fp4 weights: charged at the fp4 peak.
    case = shapes("gemm_a8wfp4", {"m": 8, "n": 16, "k": 64}, [
        tensor("a", (8, 64), dtype="float8_e4m3fn"),
        tensor("w", (16, 32), dtype="uint8", role="weight", expressions=("n", "k // 2"))])
    assert model_for("gemm_a8wfp4").estimate(case).flops_matrix == {"matrix_fp4": 2 * 8 * 16 * 64}
    # float16 operands use the fp16 peak, bfloat16 ones the bf16 peak.
    assert matrix_peak_key(16, half=True) == "matrix_fp16" and matrix_peak_key(16) == "matrix_bf16"


def test_mixed_cast_gemm_is_charged_in_the_dtype_the_operator_casts_to():
    # out = cast(a) @ cast(b): both operands are cast to the output dtype before the product.
    def key(a_dtype, b_dtype, out_dtype):
        case = shapes("gemm_mixed_cast", {"m": 8, "n": 16, "k": 64}, [
            tensor("a", (8, 64), dtype=a_dtype), tensor("b", (64, 16), dtype=b_dtype, role="weight"),
            tensor("out", (8, 16), dtype=out_dtype, direction="output")])
        estimate = model_for("gemm_mixed_cast").estimate(case)
        assert sum(estimate.flops_matrix.values()) == 2 * 8 * 16 * 64
        return list(estimate.flops_matrix)
    assert key("float32", "float16", "float16") == ["matrix_fp16"]   # the AKA task: a is rounded to float16
    assert key("float16", "float32", "float16") == ["matrix_fp16"]
    assert key("float16", "bfloat16", "bfloat16") == ["matrix_bf16"]
    # Cast up: defined as a float32 product, but charged in the first operand's narrower dtype, as
    # TraceLens's aten::mm does; the float32 peak is not a proven bound.
    assert key("float16", "float32", "float32") == ["matrix_fp16"]
    assert key("float32", "float32", "float32") == ["matrix_fp32"]


def test_tensors_are_counted_at_their_declared_widths():
    # a float32, b and out float16: 8*64*4 + 64*16*2 + 8*16*2 bytes, not three float32 tensors.
    case = shapes("gemm_mixed_cast", {"m": 8, "n": 16, "k": 64}, [
        tensor("a", (8, 64), dtype="float32"), tensor("b", (64, 16), dtype="float16", role="weight"),
        tensor("out", (8, 16), dtype="float16", direction="output")])
    assert generic_bytes(case).bytes == 8 * 64 * 4 + 64 * 16 * 2 + 8 * 16 * 2


def test_two_precision_fused_gemm_charges_each_product_separately():
    symbols = {"m": 4, "n_fp8": 8, "n_bf16": 16, "k": 32}
    case = shapes("fused_gemm_a8w8_blockscale_a16w16", symbols, [
        tensor("x_fp8", (4, 32), dtype="float8_e4m3fn"), tensor("w_fp8", (8, 32), dtype="float8_e4m3fn"),
        tensor("x_bf16", (4, 32)), tensor("w_bf16", (16, 32))])
    estimate = model_for("fused_gemm_a8w8_blockscale_a16w16").estimate(case)
    assert estimate.flops_matrix == {"matrix_fp8": 2 * 4 * 8 * 32, "matrix_bf16": 2 * 4 * 16 * 32}


def test_b2b_gemm_counts_the_cheaper_association():
    # (A Bᵀ) C: 4 * rows * keys * d. A (Bᵀ C) with Bᵀ C kept as two float16 parts (rounded to one it
    # fails the evaluator): 2 * keys * d^2 + 2 * (2 * rows * d^2).
    symbols = {"n_rows": 4096, "n_keys": 4096, "head_dim": 64}
    case = shapes("b2b_gemm", symbols, [tensor("a", (4096, 64), dtype="float16"),
                                        tensor("b", (4096, 64), dtype="float16")])
    flops = model_for("b2b_gemm").estimate(case).flops_matrix["matrix_fp16"]
    assert flops == 2 * 4096 * 64 * 64 + 4 * 4096 * 64 * 64 < 4 * 4096 * 4096 * 64


def test_avgpool_after_linear_counts_pooled_weights():
    # Pooling 16 outputs of a linear map is a linear map with pooled weights: 2 * 16 * 512 * 8192 FLOPs
    # instead of 2 * 16 * 8192 * 8192. A kernel can choose the order per case, so it is counted everywhere.
    symbols = {"batch": 16, "in_features": 8192, "out_features": 8192, "pool_size": 16, "pooled_features": 512}
    case = shapes("matmul_avgpool_gelu_scale_max", symbols,
                  [tensor("x", (16, 8192), dtype="float32"), tensor("weight", (8192, 8192), dtype="float32")])
    estimate = model_for("matmul_avgpool_gelu_scale_max").estimate(case)
    assert estimate.flops_matrix == {"matrix_fp32": 2 * 16 * 512 * 8192}
    assert "pooled_weight_shortcut_counted" in estimate.notes


def test_causal_and_window_pair_counts_match_brute_force():
    for length, window in itertools.product((1, 2, 7, 64), (0, 1, 5, 63, 100)):
        brute = sum(1 for i in range(length) for j in range(length) if i - window <= j <= i)
        assert window_pairs(length, window) == brute
        assert causal_pairs(length) == sum(i + 1 for i in range(length))


def test_varlen_attention_bound_never_exceeds_any_split():
    rng = random.Random(0)
    tokens, batch = 1000, 7
    case = shapes("flash_attention_fwd", {"n_tokens": tokens, "batch": batch, "n_heads": 1, "head_dim": 1},
                  [tensor("q", (tokens, 1, 1)), tensor("k", (tokens, 1, 1))])
    bound = model_for("flash_attention_fwd").estimate(case).flops_matrix["matrix_bf16"] // 4
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, tokens), batch - 1))
        lengths = [b - a for a, b in zip([0, *cuts], [*cuts, tokens])]
        assert bound <= sum(causal_pairs(n) for n in lengths)


def test_attention_backward_counts_five_products_per_pair():
    symbols = {"batch": 2, "seqlen": 128, "n_heads": 4, "n_kv_heads": 4, "head_dim": 64}
    case = shapes("flash_attention_bwd", symbols, [tensor("q", (2, 128, 4, 64)), tensor("do", (2, 128, 4, 64))])
    flops = model_for("flash_attention_bwd").estimate(case).flops_matrix["matrix_bf16"]
    assert flops == 5 * 2 * 64 * 4 * 2 * causal_pairs(128)


def _classical_conv_taps(extent, kernel, stride, padding):
    out = (extent + 2 * padding - kernel) // stride + 1
    return out, sum(1 for o in range(out) for t in range(kernel) if 0 <= o * stride + t - padding < extent)


def test_conv_counts_exactly_the_taps_inside_the_input():
    # NHWC, weight [out, 9, in]: a 3x3 kernel; stride and padding follow from the extents.
    for extent, (stride, padding) in itertools.product((5, 16, 31), ((1, 1), (1, 0), (2, 1))):
        out, taps = _classical_conv_taps(extent, 3, stride, padding)
        assert in_bounds_taps(extent, out, 3, stride, padding) == taps
        chosen = stride_and_padding(extent, out, 3)
        least = in_bounds_taps(extent, out, 3, *chosen)
        assert least <= taps  # several geometries can fit a small extent: the cheapest is counted
        if extent > 5:
            assert chosen == (stride, padding)
        symbols = {"batch": 2, "in_channels": 3, "out_channels": 4, "height": extent, "width": extent}
        case = shapes("conv2d", symbols, [
            tensor("x", (2, extent, extent, 3), dtype="float32",
                   expressions=("batch", "height", "width", "in_channels")),
            tensor("weight", (4, 9, 3), dtype="float32", expressions=("out_channels", 9, "in_channels")),
            tensor("out", (2, out, out, 4), dtype="float32", direction="output",
                   expressions=("batch", "out_height", "out_width", "out_channels"))])
        flops, notes = conv_flops(case, Conv("batch", "in_channels", "out_channels", "x", "weight", "out",
                                             ("height", "width")))
        assert notes == []
        assert flops == 2 * 2 * 4 * 3 * least * least
        assert flops <= 2 * 2 * 4 * 3 * taps * taps <= 2 * 2 * 4 * 3 * (out * 3) ** 2


def test_conv_with_an_unknown_weight_layout_keeps_the_one_tap_bound():
    for extent, kernel, stride in itertools.product((5, 16, 31), (1, 3, 5, 7), (1, 2, 3)):
        for padding in range(kernel // 2 + 1):
            out, taps = _classical_conv_taps(extent, kernel, stride, padding)
            if out < 1:
                continue
            case = shapes("conv2d", {"batch": 2, "in_channels": 3, "out_channels": 4},
                          [tensor("x", (2, 3, extent, extent)), tensor("weight", (4, 3 * kernel * kernel)),
                           tensor("out", (2, 4, out, out), direction="output")])
            flops, notes = conv_flops(case, Conv("batch", "in_channels", "out_channels", "x", "weight", "out"))
            assert notes == ["kernel_geometry_undeclared_one_tap_counted"]
            assert flops <= 2 * 2 * 4 * 3 * taps * taps


def test_valid_conv_with_declared_kernel_counts_every_tap():
    symbols = {"batch": 1, "in_channels": 8, "out_channels": 16, "height": 34, "width": 34, "kernel_size": 3}
    case = shapes("conv2d", symbols, [tensor("x", (1, 8, 34, 34), dtype="float32"),
                                      tensor("weight", (16, 8, 3, 3), dtype="float32")])
    flops, notes = conv_flops(case, Conv("batch", "in_channels", "out_channels", "x", "weight", "out",
                                         ("height", "width")))
    assert flops == 2 * 1 * 16 * 8 * (32 * 32) * 9
    assert notes == []


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_16_bit_3x3_stride_1_conv_counts_winograd_f4x3_multiplies(dtype):
    # Same padding, 32 x 64 outputs: F(4, 3) needs 6 multiplies per 4 outputs on each axis, 48 x 96 in all
    # per (out, in) channel pair, against (3*32 - 2) x (3*64 - 2) in-bounds direct taps.
    symbols = {"batch": 2, "in_channels": 8, "out_channels": 16, "height": 32, "width": 64}
    def conv(dtype, stride_two=False):
        out_h, out_w = (16, 32) if stride_two else (32, 64)
        case = shapes("conv2d", symbols, [
            tensor("x", (2, 32, 64, 8), dtype=dtype, expressions=("batch", "height", "width", "in_channels")),
            tensor("weight", (16, 9, 8), dtype=dtype, expressions=("out_channels", 9, "in_channels")),
            tensor("out", (2, out_h, out_w, 16), dtype=dtype, direction="output",
                   expressions=("batch", "out_height", "out_width", "out_channels"))])
        return conv_flops(case, Conv("batch", "in_channels", "out_channels", "x", "weight", "out",
                                     ("height", "width")))
    flops, notes = conv(dtype)
    assert flops == 2 * 2 * 16 * 8 * 48 * 96 < 2 * 2 * 16 * 8 * (3 * 32 - 2) * (3 * 64 - 2)
    assert notes == ["winograd_f4x3_counted: 16-bit 3x3 stride-1"]
    # float32 operands and stride-2 axes keep the direct count.
    assert conv("float32") == (2 * 2 * 16 * 8 * (3 * 32 - 2) * (3 * 64 - 2), [])
    assert conv(dtype, stride_two=True)[1] == []


def test_winograd_domain_weights_count_36_products_per_4x4_tile():
    # weight [36, out, in]: the operator is the batched product over the 36 tile elements.
    symbols = {"batch": 2, "in_channels": 8, "out_channels": 16, "height": 32, "width": 64}
    case = shapes("conv2d", symbols, [
        tensor("x", (2, 32, 64, 8), expressions=("batch", "height", "width", "in_channels")),
        tensor("weight", (36, 16, 8), expressions=(36, "out_channels", "in_channels")),
        tensor("out", (2, 32, 64, 16), direction="output",
               expressions=("batch", "height", "width", "out_channels"))])
    flops, notes = conv_flops(case, Conv("batch", "in_channels", "out_channels", "x", "weight", "out",
                                         ("height", "width")))
    tiles = 2 * (32 // 4) * (64 // 4)
    assert flops == 2 * 36 * tiles * 8 * 16
    assert notes == ["winograd_f4x3_domain_weights"]


def test_conv_geometry_of_every_task_matches_its_description():
    """The weight-layout table is hand-written; each task's description must state the same geometry."""
    from pathlib import Path
    from src.roofline.interface import case_shapes, performance_cases
    from src.roofline.operators.conv import SPECS, WEIGHT_LAYOUTS
    from src.task_contract import TASK_FILE, load_task_contract
    tasks = Path(__file__).resolve().parents[2] / "tasks"
    checked = 0
    for task_file in sorted(tasks.glob(f"*/convolution/*/*/{TASK_FILE}")):
        task_id = task_file.parent.relative_to(tasks).as_posix()
        contract = load_task_contract(task_file, task_id=task_id)
        case = case_shapes(contract, performance_cases(contract)[-1])
        spec = SPECS[case.operator]
        _flops, notes = conv_flops(case, spec)
        assert "kernel_geometry_undeclared_one_tap_counted" not in notes, task_id
        if "kernel_size" in case.symbols:
            continue
        kernel, _winograd = WEIGHT_LAYOUTS[tuple(str(e) for e in case.tensor(spec.weight).expressions)]
        description = " ".join(contract.raw["description"].split())
        assert "x".join([str(kernel)] * len(spec.spatial)) in description, task_id
        output = case.tensor(spec.output)
        outs = [extent for extent, expression in zip(output.shape, output.expressions)
                if str(expression) not in (spec.batch, spec.out_channels)]
        strides = {stride_and_padding(case.symbols[axis], out, kernel)[0] for axis, out in zip(spec.spatial, outs)}
        assert len(strides) == 1 and f"stride {strides.pop()}" in description, task_id
        checked += 1
    assert checked == 5


def test_bundled_datasheet_peaks_load():
    peaks = load_peaks("mi355x_datasheet")
    assert isinstance(peaks, Peaks)
    assert peaks.mem_bw_gbps == 8192  # 8192-bit interface at 8 Gbps; the datasheet rounds to 8 TB/s
    assert peaks.matrix_flops_per_s("matrix_bf16") == pytest.approx(2516.6e12)
    assert peaks.matrix_flops_per_s("matrix_int8") == pytest.approx(5033.2e12)
    assert peaks.vector_flops_per_s == pytest.approx(157.3e12)
    assert peaks.unconfirmed == ()


def test_uint8_operands_are_charged_as_fp4_even_when_packing_is_hidden_in_a_derived_axis():
    # fused_moe_mxfp4 declares w1 as [E, gate_up_size, hidden_bytes] with hidden_bytes = hidden_size // 2.
    case = shapes("gemm_afp4wfp4", {"m": 16, "n": 32, "k": 64, "k_bytes": 32}, [
        tensor("a", (16, 32), dtype="uint8", expressions=("m", "k_bytes")),
        tensor("w", (32, 32), dtype="uint8", role="weight", expressions=("n", "k_bytes"))])
    assert set(model_for("gemm_afp4wfp4").estimate(case).flops_matrix) == {"matrix_fp4"}


def test_moe_bounds_hold_for_any_routing():
    symbols = {"num_tokens": 4, "top_k": 2, "hidden_size": 64, "intermediate_size": 32, "num_experts": 8,
               "gate_up_size": 64}
    expert_w1, expert_w2 = 64 * 64 * 2, 64 * 32 * 2
    case = shapes("fused_moe", symbols, [
        tensor("hidden_states", (4, 64)), tensor("w13", (8, 64, 64), role="weight"),
        tensor("w2", (8, 64, 32), role="weight"), tensor("topk_weights", (4, 2), dtype="float32"),
        tensor("topk_ids", (4, 2), dtype="int32", role="index"), tensor("out", (4, 64), direction="output")])
    estimate = model_for("fused_moe").estimate(case)
    assert estimate.flops_matrix == {"matrix_bf16": 2 * 4 * 2 * (64 * 64 + 64 * 32)}
    # Two distinct experts at least: their slices of both weights, plus activations, routing and output.
    assert estimate.bytes_override == 2 * (expert_w1 + expert_w2) + 4 * 64 * 2 + 8 * 4 + 8 * 4 + 4 * 64 * 2


@pytest.mark.parametrize("operator, w1, w2, weight_dtype, key", [
    ("fused_moe_mxfp8", "w13", "w2", "float8_e4m3fn", "matrix_fp8"),         # tokens quantized to MXFP8
    ("fused_moe_fp8_blockscale", "w1", "w2", "float8_e4m3fn", "matrix_fp8"),  # tokens quantized to fp8
    ("fused_moe_mxfp4", "w1", "w2", "uint8", "matrix_fp4"),                  # tokens quantized to MXFP4
    ("fused_moe_a8w4", "w1", "w2", "uint8", "matrix_fp4"),                   # fp8 tokens, MXFP4 weights: TraceLens's peak
    ("fused_moe_a16w4", "w1", "w2", "uint8", "matrix_fp4"),                  # bfloat16 tokens, MXFP4 weights
])
def test_moe_products_are_charged_in_the_format_the_tokens_enter_them(operator, w1, w2, weight_dtype, key):
    symbols = {"num_tokens": 4, "top_k": 2, "hidden_size": 64, "intermediate_size": 32, "num_experts": 8}
    case = shapes(operator, symbols, [
        tensor("hidden_states", (4, 64)), tensor(w1, (8, 64, 64), dtype=weight_dtype, role="weight"),
        tensor(w2, (8, 64, 32), dtype=weight_dtype, role="weight"),
        tensor("topk_ids", (4, 2), dtype="int32", role="index"), tensor("out", (4, 64), direction="output")])
    assert list(model_for(operator).estimate(case).flops_matrix) == [key]


def test_chunk_pair_counts_match_brute_force():
    from src.roofline.operators.chunked import chunk_pairs
    for tokens, chunk in itertools.product((1, 63, 64, 65, 200), (16, 64)):
        rows = [(t, t // chunk) for t in range(tokens)]
        strict = sum(1 for i, ci in rows for j, cj in rows if ci == cj and j < i)
        causal = sum(1 for i, ci in rows for j, cj in rows if ci == cj and j <= i)
        assert chunk_pairs(tokens, chunk, strict=True) == strict
        assert chunk_pairs(tokens, chunk, strict=False) == causal


def test_solve_tril_reads_only_the_strict_lower_triangle_of_a():
    tokens, heads, chunk = 128, 2, 64
    case = shapes("solve_tril", {"n_tokens": tokens, "num_heads": heads, "chunk_size": chunk}, [
        tensor("A", (tokens, heads, chunk), dtype="float32"),
        tensor("out", (tokens, heads, chunk), dtype="float32", direction="output")])
    estimate = model_for("solve_tril").estimate(case)
    assert estimate.bytes_override == heads * 2 * (64 * 63 // 2) * 4 + tokens * heads * chunk * 4


def test_shared_key_heads_count_k_kt_once_per_key_head():
    symbols = {"n_tokens": 64, "chunk_size": 64, "num_heads": 32, "num_key_heads": 16, "head_dim": 128}
    case = shapes("chunk_scaled_dot_kkt", symbols, [tensor("k", (64, 16, 128))])
    flops = model_for("chunk_scaled_dot_kkt").estimate(case).flops_matrix["matrix_bf16"]
    assert flops == 2 * 128 * 16 * (64 * 63 // 2)


class FakeTensor:
    """Stand-in for a generated torch tensor: data models only call .tolist()."""

    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


def test_paged_decode_counts_generated_context_lengths_only():
    from src.roofline.generated import DATA_MODELS
    symbols = {"num_seqs": 3, "num_query_heads": 8, "num_kv_heads": 1, "head_size": 128, "num_blocks": 100,
               "block_size": 16}
    case = shapes("paged_attention_decode", symbols, [
        tensor("query", (3, 8, 128)), tensor("key_cache", (100, 1, 8, 16, 16), dtype="float8_e4m3fn", role="state"),
        tensor("value_cache", (100, 1, 1, 128, 16), dtype="float8_e4m3fn", role="state"),
        tensor("out", (3, 8, 128), direction="output")])
    _, model = DATA_MODELS["paged_attention_decode"]
    lengths = [100, 50, 7]
    estimate = model(case, {"context_lengths": FakeTensor(lengths), "kv_indptr": FakeTensor([0, 7, 11, 12])})
    keys = sum(lengths)
    # bfloat16 queries against fp8 keys and values: charged at the fp8 peak.
    assert estimate.flops_matrix == {"matrix_fp8": 2 * (128 + 128) * 8 * keys}
    assert estimate.bytes_override == keys * 128 * 2 + 3 * 8 * 128 * 2 * 2


def test_sliding_window_decode_caps_each_sequence_at_the_window():
    from src.roofline.generated import DATA_MODELS
    symbols = {"num_query_heads": 1, "num_kv_heads": 1, "head_size": 2}
    case = shapes("paged_attention_decode_swa", symbols, [
        tensor("query", (2, 1, 2)), tensor("key_cache", (10, 2)), tensor("value_cache", (10, 2)),
        tensor("out", (2, 1, 2), direction="output")], scalars={"sliding_window": 256})
    _, model = DATA_MODELS["paged_attention_decode_swa"]
    estimate = model(case, {"context_lengths": FakeTensor([1000, 100])})
    assert estimate.flops_matrix == {"matrix_bf16": 2 * 4 * (256 + 100)}


def test_ragged_decode_uses_token_indptr_differences():
    from src.roofline.operators.attention_data import context_lengths
    assert context_lengths({"kv_indptr": FakeTensor([0, 5, 5, 12])}) == [5, 0, 7]


def test_attention_backward_bytes_exclude_the_saved_output_and_lse():
    symbols = {"batch": 1, "seqlen": 16, "n_heads": 2, "n_kv_heads": 2, "head_dim": 8}
    full = (1, 16, 2, 8)
    case = shapes("flash_attention_bwd", symbols, [
        tensor("q", full), tensor("k", full), tensor("v", full), tensor("o", full), tensor("do", full),
        tensor("lse", (1, 2, 16), dtype="float32"),
        tensor("dq", full, direction="output"), tensor("dk", full, direction="output"),
        tensor("dv", full, direction="output")])
    estimate = model_for("flash_attention_bwd").estimate(case)
    assert estimate.bytes_override == 7 * (16 * 2 * 8 * 2)


def test_jagged_bmm_reads_the_weights_of_groups_that_have_rows():
    from src.roofline.generated import DATA_MODELS
    symbols = {"total_rows": 100, "k": 16, "n": 32, "n_groups": 4}
    case = shapes("jagged_dense_bmm_broadcast_add", symbols, [
        tensor("seq_offsets", (5,), dtype="int64", role="metadata"), tensor("jagged", (100, 16)),
        tensor("dense", (4, 16, 32), role="weight"), tensor("bias", (4, 32), role="weight"),
        tensor("out", (100, 32), direction="output")])
    per_group = 16 * 32 * 2 + 32 * 2
    rest = 5 * 8 + 100 * 16 * 2 + 100 * 32 * 2
    # Without the offsets: the rows are in some group, so one group's dense and bias.
    estimate = model_for("jagged_dense_bmm_broadcast_add").estimate(case)
    assert estimate.flops_matrix == {"matrix_bf16": 2 * 100 * 32 * 16}
    assert estimate.bytes_override == rest + per_group
    # With them: groups 0 and 3 have rows, groups 1 and 2 are empty.
    _, model = DATA_MODELS["jagged_dense_bmm_broadcast_add"]
    estimate = model(case, {"seq_offsets": FakeTensor([0, 60, 60, 60, 100])})
    assert estimate.bytes_override == rest + 2 * per_group


def test_partial_inputs_count_only_the_used_elements():
    from src.roofline.bytes_model import generic_bytes
    # learned_position_add reads seq_len of max_positions embedding rows.
    symbols = {"batch": 2, "seq_len": 4, "hidden_size": 8, "max_positions": 100}
    case = shapes("learned_position_add", symbols, [
        tensor("x", (2, 4, 8)), tensor("pos_embedding", (100, 8), role="weight"),
        tensor("out", (2, 4, 8), direction="output")])
    counted = generic_bytes(case)
    assert counted.bytes == 2 * 4 * 8 * 2 + 4 * 8 * 2 + 2 * 4 * 8 * 2
    assert "partial_input:pos_embedding" in counted.notes


def test_ungated_ffn_merges_the_weights_only_without_an_activation():
    symbols = {"n_tokens": 8192, "hidden_size": 2048, "intermediate_size": 8192}
    tensors = [tensor("x", (8192, 2048)), tensor("w_up", (8192, 2048))]
    two_gemms = 4 * 8192 * 2048 * 8192
    activated = shapes("ff_a16w16_ungated", symbols, tensors, description="out = relu(x @ w_up^T) @ w_down^T")
    estimate = model_for("ff_a16w16_ungated").estimate(activated)
    assert estimate.flops_matrix["matrix_bf16"] == two_gemms and estimate.notes == []
    linear = shapes("ff_a16w16_ungated", symbols, tensors, description="out = (x @ w_up^T) @ w_down^T, hidden 2048")
    estimate = model_for("ff_a16w16_ungated").estimate(linear)
    # The merged weight W_down W_up is kept as two bf16 parts (rounded to one it fails the evaluator), so
    # the product with it is charged twice.
    assert estimate.flops_matrix["matrix_bf16"] == 2 * 2048 * 2048 * 8192 + 4 * 8192 * 2048 * 2048 < two_gemms
    assert estimate.notes == ["no_activation: merged-weight order considered (two-part merged weight)"]


def test_ungated_ffn_tasks_are_read_as_their_descriptions_say():
    from pathlib import Path
    from src.roofline.interface import case_shapes, performance_cases
    from src.task_contract import TASK_FILE, load_task_contract
    tasks = Path(__file__).resolve().parents[2] / "tasks" / "triton" / "gemm" / "ff_a16w16_ungated"
    merged = {}
    for task_file in sorted(tasks.glob(f"*/{TASK_FILE}")):
        contract = load_task_contract(task_file, task_id=f"triton/gemm/ff_a16w16_ungated/{task_file.parent.name}")
        case = case_shapes(contract, performance_cases(contract)[-1])
        merged[task_file.parent.name] = bool(model_for("ff_a16w16_ungated").estimate(case).notes)
    assert merged == {"bf16_h2048_i8192_gelu_tanh": False, "bf16_h2048_i8192_noact": True,
                      "bf16_h2048_i8192_relu": False, "bf16_h2048_i8192_silu": False}


def test_attention_backward_counts_every_pair_of_a_non_causal_scenario():
    symbols = {"batch": 2, "seqlen": 128, "n_heads": 4, "n_kv_heads": 4, "head_dim": 64}
    tensors = [tensor("q", (2, 128, 4, 64)), tensor("do", (2, 128, 4, 64))]
    causal = shapes("flash_attention_bwd", symbols, tensors, description="Backward pass of causal attention")
    dense = shapes("flash_attention_bwd", symbols, tensors, description="Backward pass of non-causal attention")
    model = model_for("flash_attention_bwd")
    assert model.estimate(causal).flops_matrix["matrix_bf16"] == 5 * 2 * 64 * 4 * 2 * causal_pairs(128)
    assert model.estimate(dense).flops_matrix["matrix_bf16"] == 5 * 2 * 64 * 4 * 2 * 128 * 128
    assert model.estimate(dense).notes[0] == "all_pairs_counted: non-causal scenario"


def test_only_the_non_causal_backward_task_counts_every_pair():
    from pathlib import Path
    from src.roofline.interface import case_shapes, performance_cases
    from src.roofline.operators.attention import is_non_causal
    from src.task_contract import TASK_FILE, load_task_contract
    tasks = Path(__file__).resolve().parents[2] / "tasks"
    dense = set()
    for operator in ("flash_attention_fwd", "flash_attention_bwd", "sage_attention_fwd", "sliding_window_attention",
                     "causal_self_attention"):
        for task_file in sorted(tasks.glob(f"*/attention/{operator}/*/{TASK_FILE}")):
            task_id = task_file.parent.relative_to(tasks).as_posix()
            contract = load_task_contract(task_file, task_id=task_id)
            if is_non_causal(case_shapes(contract, performance_cases(contract)[-1])):
                dense.add(task_id)
    assert dense == {"triton/attention/flash_attention_bwd/bf16_noncausal_mha16_d72_dit"}


def test_floor_mode_pooling_drops_trailing_conv_positions_and_input_rows():
    # depth 16, kernel 3: 14 conv outputs, two 2x pools keep 3 -> 12 positions, 14 input rows.
    symbols = {"batch": 1, "in_channels": 1, "out_channels": 1, "depth": 16, "height": 16, "width": 16,
               "kernel_size": 3, "pooled_depth": 3, "pooled_height": 3, "pooled_width": 3}
    case = shapes("conv3d_softmax_maxpool2", symbols, [
        tensor("x", (1, 1, 16, 16, 16), dtype="float32"), tensor("weight", (1, 1, 3, 3, 3), dtype="float32"),
        tensor("out", (1, 1, 3, 3, 3), dtype="float32", direction="output")])
    estimate = model_for("conv3d_softmax_maxpool2").estimate(case)
    assert estimate.flops_matrix["matrix_fp32"] == 2 * (12 * 3) ** 3
    assert estimate.bytes_override == 14 ** 3 * 4 + 27 * 4 + 27 * 4


def test_moe_data_model_reads_only_routed_experts():
    from src.roofline.generated import DATA_MODELS
    symbols = {"num_tokens": 2, "top_k": 2, "hidden_size": 8, "intermediate_size": 4, "num_experts": 8,
               "gate_up_size": 8}
    case = shapes("fused_moe", symbols, [
        tensor("hidden_states", (2, 8)), tensor("w13", (8, 8, 8), role="weight"),
        tensor("w2", (8, 8, 4), role="weight"), tensor("topk_ids", (2, 2), dtype="int32", role="index"),
        tensor("out", (2, 8), direction="output")])
    _, model = DATA_MODELS["fused_moe"]
    estimate = model(case, {"topk_ids": FakeTensor([[0, 3], [3, 5]])})
    per_expert = 8 * 8 * 2 + 8 * 4 * 2
    assert estimate.bytes_override == 3 * per_expert + 2 * 8 * 2 + 4 * 4 + 2 * 8 * 2
    assert estimate.flops_matrix == model_for("fused_moe").estimate(case).flops_matrix


def test_paged_decode_accepts_a_seq_lens_length_tensor():
    from src.roofline.generated import DATA_MODELS
    symbols = {"num_seqs": 2, "num_query_heads": 4, "num_kv_heads": 1, "head_size": 64, "num_blocks": 20,
               "block_size": 16}
    case = shapes("paged_attention_decode", symbols, [
        tensor("query", (2, 4, 64)), tensor("key_cache", (20, 1, 8, 16, 8), role="state"),
        tensor("value_cache", (20, 1, 64, 16), role="state"),
        tensor("out", (2, 4, 64), direction="output")])
    _, model = DATA_MODELS["paged_attention_decode"]
    lengths = [33, 5]
    estimate = model(case, {"seq_lens": FakeTensor(lengths), "block_table": FakeTensor([[0, 1, 2], [3, 0, 0]])})
    keys = sum(lengths)
    assert estimate.flops_matrix == {"matrix_bf16": 2 * (64 + 64) * 4 * keys}
    assert estimate.bytes_override == keys * 64 * 2 * 2 + 2 * 4 * 64 * 2 * 2


def test_generator_imports_task_siblings_without_leaking_or_writing_bytecode(tmp_path):
    import sys
    from src.roofline.generated import _task_function
    for name, scale in (("a", 2), ("b", 3)):
        task = tmp_path / name
        task.mkdir()
        (task / "helper_numerics.py").write_text(f"SCALE = {scale}\n")
        (task / "inputs.py").write_text("from helper_numerics import SCALE\n\n"
                                        "def make_inputs(case, inputs, generator, device):\n"
                                        "    return {'x': SCALE}\n")
    path_before = list(sys.path)
    first = _task_function(tmp_path / "a", "synthetic/sibling_a", {"file": "inputs.py", "symbol": "make_inputs"})
    second = _task_function(tmp_path / "b", "synthetic/sibling_b", {"file": "inputs.py", "symbol": "make_inputs"})
    assert first(None, None, None, None) == {"x": 2}
    assert second(None, None, None, None) == {"x": 3}
    assert sys.path == path_before
    assert "helper_numerics" not in sys.modules
    assert not list(tmp_path.rglob("__pycache__"))
