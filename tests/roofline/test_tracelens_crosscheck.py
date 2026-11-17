"""Cross-check roofline FLOP and byte counts against TraceLens's PerfModel formulas.

TraceLens is not an AKA dependency. Point ``TRACELENS_ROOT`` at a TraceLens checkout (tested
at commit f058ca21) in an environment that can import it; otherwise these tests are skipped.

``CHECKS`` takes one real performance case per Tier 2 operator family and states how our count
relates to TraceLens's: equal, or different by an explained amount that the test recomputes. A
``None`` byte relation means TraceLens's byte model describes a different interface (an unfused
op sequence, or a production op whose outputs differ from the task's), so only the numbers are shown.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import os
from pathlib import Path
import sys
from typing import Callable

import pytest

from src.roofline.arch import load_peaks
from src.roofline.bytes_model import generic_bytes
from src.roofline.floor import case_floor
from src.roofline.interface import CaseShapes, Tensor, case_shapes, performance_cases
from src.roofline.operators import model_for
from src.roofline.operators.attention import causal_pairs, window_pairs
from src.task_contract import TASK_FILE, load_task_contract


def tensor(name, shape, dtype="bfloat16", direction="input", role="activation"):
    return Tensor(name, direction, role if direction == "input" else "output", dtype, tuple(shape), tuple(shape))


def shapes(operator, symbols, tensors):
    return CaseShapes("crosscheck/" + operator, "case", operator, "test", dict(symbols), dict(symbols), tuple(tensors))

ROOT = os.environ.get("TRACELENS_ROOT")
if not ROOT:
    pytest.skip("TRACELENS_ROOT is not set", allow_module_level=True)
sys.path.insert(0, ROOT)
perf_model = pytest.importorskip("TraceLens.PerfModel.perf_model")
extensions = pytest.importorskip("TraceLens.PerfModel.extensions.perf_model_extensions")
GEMM, SDPA, CONV = perf_model.GEMM, perf_model.SDPA, perf_model.CONV

REPO = Path(__file__).resolve().parents[2]
PEAKS = load_peaks("mi355x_datasheet")
BF16, FP8, U8, I8, F32 = "c10::bfloat16", "c10::float8_e4m3fn", "unsigned char", "signed char", "float"


@pytest.mark.parametrize("m,n,k", [(4096, 4096, 2048), (1, 1280, 8192), (8192, 7168, 4608)])
def test_gemm_flops_and_bytes_match(m, n, k):
    case = shapes("gemm_a16w16", {"m": m, "n": n, "k": k},
                  [tensor("a", (m, k)), tensor("b", (n, k), role="weight"),
                   tensor("out", (m, n), direction="output")])
    ours = model_for("gemm_a16w16").estimate(case).flops_matrix["matrix_bf16"]
    assert ours == perf_model.GEMM.flops_func(m, n, k, False)
    assert generic_bytes(case).bytes == perf_model.GEMM.bytes_func(m, n, k, False, 2, 2, 2, 2)


def test_causal_attention_flops_do_not_exceed_tracelens():
    # TraceLens's causal SDPA count may use S^2/2; ours is the exact S(S+1)/2 pair count. Ours must agree
    # to within the diagonal term and never exceed a dense count.
    batch, length, heads, dim = 2, 1024, 32, 128
    case = shapes("flash_attention_fwd", {"batch": batch, "seq_len": length, "n_heads": heads, "head_dim": dim},
                  [tensor("q", (batch, length, heads, dim)), tensor("k", (batch, length, heads, dim))])
    ours = model_for("flash_attention_fwd").estimate(case).flops_matrix["matrix_bf16"]
    theirs = perf_model.SDPA.flops_func(batch, length, heads, length, heads, dim, dim, True)
    dense = perf_model.SDPA.flops_func(batch, length, heads, length, heads, dim, dim, False)
    assert ours == 4 * dim * heads * batch * causal_pairs(length)
    assert ours <= dense
    assert abs(ours - theirs) <= 4 * dim * heads * batch * length


# ----------------------------------------------------------------------------------------------
# One real case per Tier 2 family.

def event(dims, types=()):
    return {"args": {"Input Dims": dims, "Input type": list(types), "Input Strides": [[] for _ in dims],
                     "Concrete Inputs": []}}


@dataclass(frozen=True)
class Check:
    family: str
    task_id: str
    tracelens: str
    # Symbols and scalars of the case -> (TraceLens FLOPs, TraceLens bytes or None).
    theirs: Callable[[dict, dict], tuple[float, float | None]]
    # (symbols, TraceLens FLOPs) -> the FLOPs we expect; (symbols, TraceLens bytes) -> our bytes, or None.
    flops: Callable[[dict, float], float]
    bytes: Callable[[dict, float], float] | None
    why: str


def same(_s, value):
    return value


def gemm(m, n, k):
    return GEMM.flops_func(m, n, k, False)


def sdpa(b, s, h, d, causal, hkv=None):
    return SDPA.flops_func(b, s, h, s, hkv or h, d, d, causal)


def _conv_k3(s, _):
    n, h, w = s["batch"], s["height"], s["width"]
    x, wt, out = (n, 16, h, w), (128, 16, 3, 3), (n, 128, h - 2, w - 2)
    return CONV.flops_func(x, wt, out, False), CONV.bytes_func(x, wt, out, False, 4)


def _conv_nhwc_k3(s, _):
    n, h, w = s["batch"], s["height"], s["width"]
    x, wt, out = (n, 256, h, w), (256, 256, 3, 3), (n, 256, h, w)
    return CONV.flops_func(x, wt, out, False), CONV.bytes_func(x, wt, out, False, 2)


def _bwd(s, _):
    b, t, h, hkv, d = s["batch"], s["seqlen"], s["n_heads"], s["n_kv_heads"], s["head_dim"]
    return (SDPA.flops_bwd_func(b, t, h, t, hkv, d, d, True, True),
            SDPA.bytes_bwd_func(b, t, h, t, hkv, d, d, True, 2))


def _bwd_flops(s, theirs):
    # TraceLens halves the dense count (ours adds the diagonal) and adds the GQA head reduction of dK
    # and dV, B*S*d*(H/Hkv - 1) each, which is vector work we do not count.
    b, t, h, hkv, d = s["batch"], s["seqlen"], s["n_heads"], s["n_kv_heads"], s["head_dim"]
    return theirs - b * t * d * (h // hkv - 1) + 5 * d * h * b * t


def _csa(s, _):
    b, t, e, h = s["batch"], s["seq_len"], s["n_embd"], s["n_head"]
    return gemm(b * t, 3 * e, e) + sdpa(b, t, h, e // h, True) + gemm(b * t, e, e), None


def _vision(s, _):
    b, t, e, h = s["batch"], s["hw"], s["embed_dim"], s["num_heads"]
    return gemm(b * t, 3 * e, e) + sdpa(b, t, h, e // h, False) + gemm(b * t, e, e), None


def _nonlocal(s, _):
    b, n, c, q = s["batch"], s["hw"], s["channels"], s["qk_channels"]
    projections = 2 * gemm(b * n, q, c) + gemm(b * n, c, c)
    return projections + b * gemm(n, n, q) + b * gemm(c, n, n) + gemm(b * n, c, c), None


def _mingpt(s, _):
    b, t, e, h = s["batch"], s["seq_len"], s["n_embd"], s["n_head"]
    rows = b * t
    return (gemm(rows, 3 * e, e) + sdpa(b, t, h, e // h, True) + gemm(rows, e, e) + gemm(rows, 4 * e, e)
            + gemm(rows, e, 4 * e)), None


def _mhc(s, _):
    m, n, c = s["tokens"], s["hc_mult"], s["hidden_size"]
    hc3 = s["mix_dim"]
    op = extensions.mhc_fused_post_pre_gemm_sqrsum(event(
        [[1, m, hc3], [1, m], [m, n, c], [m, c], [m, n, c], [m, n, 1], [m, n, n], [hc3, n * c]],
        [F32, F32, BF16, BF16, BF16, F32, F32, F32]))
    return op.flops(), op.bytes()


def _mhc_flops(s, theirs):
    # The post step (2*M*n*C*(n+1)) mixes the streams elementwise: vector work, not counted.
    m, n, c = s["tokens"], s["hc_mult"], s["hidden_size"]
    return theirs - 2 * m * n * c * (n + 1)


def _ssd_bmm(s, _):
    # MambaSSD step 1a (C B^T), 2*B*G*T*C*N, from TraceLens's MambaSSD.flops.
    return 2 * s["ngroups"] * s["n_tokens"] * s["chunk_size"] * s["dstate"], None


def _ssd_bmm_flops(s, theirs):
    # Ours bounds sum n_c^2 over the n_chunks chunks by T^2/n_chunks; equal to T*C for full chunks.
    t, chunks = s["n_tokens"], s["n_chunks"]
    return theirs * Fraction(t * t // chunks, t * s["chunk_size"])


def _ssd_scan(s, _):
    # MambaSSD steps 1b (M X, full chunk square) and 4 (C h), from TraceLens's MambaSSD.flops.
    h, t, c, p, n = s["num_heads"], s["n_tokens"], s["chunk_size"], s["head_dim"], s["state_dim"]
    return 2 * h * t * c * p + 2 * h * t * n * p, None


def _ssd_scan_flops(s, _theirs):
    # Ours: only the causal half of M X (the mask zeroes the rest), smallest for equal chunks, and no
    # C h term (a chunk that starts a sequence has no incoming state).
    t, chunks = s["n_tokens"], s["n_chunks"]
    return 2 * s["num_heads"] * s["head_dim"] * (t * t // (2 * chunks) + t // 2)


def _bmm_rope(s, _):
    t, qh, kh, k, n, pe = (s[x] for x in ("n_tokens", "num_q_heads", "num_kv_heads", "qk_nope_head_dim",
                                           "kv_lora_dim", "rope_dim"))
    op = extensions.fused_fp8_bmm_rope_cat_and_cache_mla(event(
        [[qh, t, k], [qh, n, k], [], [t, qh, pe], [t, kh, n], [t, kh, pe], [s["num_slots"], kh, n + pe], [t], [t],
         [s["max_position"], pe // 2], [s["max_position"], pe // 2], []],
        [BF16, FP8, F32, BF16, BF16, BF16, BF16, "long int", "long int", BF16, BF16, F32]))
    return op.flops(), op.bytes()


def _bmm_rope_flops(s, theirs):
    t, qh, kh, pe = s["n_tokens"], s["num_q_heads"], s["num_kv_heads"], s["rope_dim"]
    return theirs - 3 * (t * qh * pe + t * kh * pe)  # RoPE is vector work


def _bmm_rope_bytes(s, theirs):
    # The task also returns the rotated k_pe as its own output, which the aiter op only writes into
    # the cache, and ours counts the 4-byte w_k scale.
    t, kh, pe = (s[x] for x in ("n_tokens", "num_kv_heads", "rope_dim"))
    return theirs + t * kh * pe * 2 + 4


CHECKS = [
    Check("GEMM", "triton/gemm/gemm_a16w16/n1280_k8192", "GEMM.flops_func / bytes_func",
          lambda s, _: (gemm(s["m"], s["n"], s["k"]), GEMM.bytes_func(s["m"], s["n"], s["k"], False, 2, 2, 2, 2)),
          same, same, "equal"),
    Check("batched GEMM", "triton/gemm/batched_gemm_bf16/b128_n128_k512", "aten_bmm",
          lambda s, _: (lambda op: (op.flops(), op.bytes()))(perf_model.aten_bmm(event(
              [[s["batch"], s["m"], s["k"]], [s["batch"], s["k"], s["n"]]], [BF16, BF16]))),
          same, same, "equal"),
    Check("fp8 GEMM, per-token scales", "triton/gemm/gemm_a8w8/n1280_k8192", "GEMM (fp8 bytes per element)",
          lambda s, _: (gemm(s["m"], s["n"], s["k"]), GEMM.bytes_func(s["m"], s["n"], s["k"], False, 1, 1, 2, 2)),
          same, lambda s, b: b + 4 * (s["m"] + s["n"]),
          "bytes: ours adds the fp32 x_scale [m] and w_scale [n], which TraceLens's GEMM leaves out"),
    Check("fp8 block-scale GEMM", "gluon/gemm/gemm_a8w8_blockscale/n7168_k2048", "extension gemm_a8w8_blockscale",
          lambda s, _: (lambda op: (op.flops(), op.bytes()))(extensions.gemm_a8w8_blockscale(event(
              [[s["m"], s["k"]], [s["n"], s["k"]], [s["m"], s["k"] // 128], [s["n"] // 128, s["k"] // 128]],
              [FP8, FP8, F32, F32]))),
          same, lambda s, b: b + 4 * (s["m"] * (s["k"] // 128) + (s["n"] // 128) * (s["k"] // 128)),
          "bytes: ours adds the fp32 block scales, which the extension omits"),
    Check("MXFP4 GEMM", "gluon/gemm/gemm_afp4wfp4/n7168_k16384", "extension gemm_afp4wfp4",
          lambda s, _: (lambda op: (op.flops(), op.bytes()))(extensions.gemm_afp4wfp4(event(
              [[s["m"], s["k"] // 2], [s["n"], s["k"] // 2], [s["m"], s["k"] // 32], [s["n"], s["k"] // 32]],
              [U8, U8, U8, U8]))),
          same, same, "equal (both count the e8m0 scales)"),
    Check("a16wfp4 batched GEMM", "triton/gemm/batched_gemm_a16wfp4/b128_n128_k512", "extension batched_gemm_a16wfp4",
          lambda s, _: (lambda op: (op.flops(), op.bytes()))(extensions.batched_gemm_a16wfp4(event(
              [[s["batch"], s["m"], s["k"]], [s["batch"], s["n"], s["k"] // 2], [s["batch"], s["n"], s["k"] // 32]],
              [BF16, U8, U8]))),
          same, lambda s, b: b + s["batch"] * s["n"] * (s["k"] // 32),
          "bytes: ours adds the e8m0 w_scales; FLOPs equal, ours charged at the fp4 peak"),
    Check("int8 batched GEMM", "triton/gemm/batched_gemm_a8w8/int8_b16_n1280_k8192", "extension batched_gemm_a8w8",
          lambda s, _: (lambda op: (op.flops(), op.bytes()))(extensions.batched_gemm_a8w8(event(
              [[s["batch"], s["m"], s["k"]], [s["batch"], s["n"], s["k"]]], [I8, I8]))),
          same, lambda s, b: b + 4 * s["batch"] * (s["m"] + s["n"]),
          "bytes: ours adds the fp32 x_scale and w_scale; FLOPs equal, ours charged at the int8 peak"),
    Check("grouped GEMM", "triton/gemm/grouped_gemm/bf16_g64_k1408_n2048", "GroupedGemm.flops_func / bytes_func",
          lambda s, _: (perf_model.GroupedGemm.flops_func(s["m"], s["k"], s["n"]),
                        perf_model.GroupedGemm.bytes_func(s["m"], s["k"], s["n"], s["num_groups"], 2, 2)),
          same, lambda s, b: b - (s["num_groups"] - 1) * s["k"] * s["n"] * 2,
          "bytes: TraceLens reads all G weight slices; ours one, since a group with no rows needs none"),
    Check("ungated FFN (two GEMMs)", "triton/gemm/ff_a16w16_ungated/bf16_h2048_i8192_noact", "GEMM x 2",
          lambda s, _: (gemm(s["n_tokens"], s["intermediate_size"], s["hidden_size"])
                        + gemm(s["n_tokens"], s["hidden_size"], s["intermediate_size"]), None),
          lambda s, f: min(f, 2 * s["hidden_size"] ** 2 * s["intermediate_size"]
                           + 4 * s["n_tokens"] * s["hidden_size"] ** 2),
          None, "FLOPs: this scenario has no activation, so ours takes the cheaper of the two orders "
                "(W_down W_up formed once and kept as two bf16 parts); bytes: TraceLens's per-op sum includes "
                "the intermediate"),
    Check("conv, declared 3x3 valid", "triton/convolution/conv2d/f32_cin16_cout128_k3", "CONV.flops_func / bytes_func",
          _conv_k3, same, same, "equal"),
    Check("conv, 3x3 same padding (NHWC)", "triton/convolution/conv2d/bf16_nhwc_cin256_cout256_k3",
          "CONV.flops_func / bytes_func", _conv_nhwc_k3,
          lambda s, f: f * (3 * s["height"] - 2) * (3 * s["width"] - 2) // (9 * s["height"] * s["width"]),
          lambda s, b: b + 4 * s["out_channels"],
          "FLOPs: ours leaves out the taps that fall in the zero padding; "
          "bytes: ours adds the fp32 bias that TraceLens's single dtype leaves out"),
    Check("flash attention forward", "flydsl/attention/flash_attention_fwd/bf16_causal_mha_h32_d128",
          "SDPA.flops_func / bytes_func (causal)",
          lambda s, _: (sdpa(s["batch"], s["seq_len"], s["n_heads"], s["head_dim"], True),
                        SDPA.bytes_func(s["batch"], s["seq_len"], s["n_heads"], s["seq_len"], s["n_heads"],
                                        s["head_dim"], s["head_dim"], True, 2)),
          lambda s, f: f + 2 * s["head_dim"] * s["n_heads"] * s["batch"] * s["seq_len"], same,
          "FLOPs: ours counts the exact causal pairs S(S+1)/2, TraceLens S^2/2 (diagonal term)"),
    Check("flash attention backward", "triton/attention/flash_attention_bwd/bf16_causal_gqa32x8_d128",
          "SDPA.flops_bwd_func / bytes_bwd_func (flash, causal)", _bwd, _bwd_flops, same,
          "FLOPs: same five products; ours adds the causal diagonal and leaves out the GQA head reduction"),
    Check("sliding-window attention", "flydsl/attention/sliding_window_attention/bf16_causal_mha_h32_d128_w4096",
          "SDPA (causal; no window)",
          lambda s, _: (sdpa(s["batch"], s["seq_len"], s["n_heads"], s["head_dim"], True),
                        SDPA.bytes_func(s["batch"], s["seq_len"], s["n_heads"], s["seq_len"], s["n_kv_heads"],
                                        s["head_dim"], s["head_dim"], True, 2)),
          lambda s, _f: 4 * s["head_dim"] * s["n_heads"] * s["batch"] * window_pairs(s["seq_len"], s["window_left"]),
          same, "FLOPs: TraceLens has no window; ours counts only the pairs inside it"),
    Check("SageAttention forward", "triton/attention/sage_attention_fwd/int8qk_fp8v_h8_d128", "SDPA (causal)",
          lambda s, _: (sdpa(s["batch"], s["seqlen"], s["n_heads"], s["head_dim"], True),
                        SDPA.bytes_func(s["batch"], s["seqlen"], s["n_heads"], s["seqlen"], s["n_heads"],
                                        s["head_dim"], s["head_dim"], True, 2)),
          lambda s, f: f + 2 * s["head_dim"] * s["n_heads"] * s["batch"] * s["seqlen"], same,
          "FLOPs: causal diagonal as above; ours charges the fp4 peak (in-kernel quantization)"),
    Check("causal self-attention block", "triton/attention/causal_self_attention/f32_e768_h8", "GEMM + SDPA + GEMM",
          _csa, lambda s, f: f + 2 * s["n_embd"] * s["batch"] * s["seq_len"], None,
          "FLOPs: causal diagonal; bytes: the unfused sum moves q, k, v and the attention output"),
    Check("vision attention block", "triton/attention/vision_attention_block/f32_e128_h4", "GEMM + SDPA + GEMM",
          _vision, same, None, "FLOPs equal (unmasked); bytes: unfused sum, not comparable"),
    Check("non-local attention", "hip/attention/nonlocal_attention_dot/f32_c64", "GEMM x 4 + bmm x 2",
          _nonlocal, same, None, "FLOPs equal (the reference's order); bytes: unfused sum"),
    Check("minGPT block", "triton/other/mingpt_block/f32_e768_h8", "GEMM x 4 + SDPA", _mingpt,
          lambda s, f: f + 2 * s["n_embd"] * s["batch"] * s["seq_len"], None,
          "FLOPs: causal diagonal; bytes: unfused sum"),
    Check("mHC post + pre", "tilelang/other/mhc_fused_post_pre/c4_d7168", "extension mhc_fused_post_pre_gemm_sqrsum",
          _mhc, _mhc_flops, None,
          "FLOPs: ours drops the post-mix vector work; bytes: the aiter op writes split-K GEMM partials and "
          "re-reads next_residual, while the task returns the final mixes"),
    Check("SSD C B^T", "triton/state_space/ssd_bmm_chunk/bf16_g8_n128_c256", "MambaSSD step 1a", _ssd_bmm,
          _ssd_bmm_flops, None,
          "FLOPs: equal for full chunks, otherwise ours uses T^2/n_chunks; bytes: MambaSSD models the fused layer"),
    Check("SSD chunk scan", "triton/state_space/ssd_chunk_scan/bf16_h128_p64_g8_n128_c256", "MambaSSD steps 1b + 4",
          _ssd_scan, _ssd_scan_flops, None,
          "FLOPs: ours counts the causal half of M X and no C h term; bytes: fused layer, not comparable"),
    Check("fp8 BMM + RoPE + cache", "triton/kv_cache/fused_bmm_rope_kv_cache/fp8w_bf16_h128_nope128_lora512_rope64",
          "extension fused_fp8_bmm_rope_cat_and_cache_mla", _bmm_rope, _bmm_rope_flops, _bmm_rope_bytes,
          "FLOPs: RoPE is vector work; bytes: ours adds the k_pe output and the w_k scale"),
    Check("gathered latent projection", "triton/kv_cache/gather_kv_b_proj/bf16kv_fp8w_h16_nope128_v128_lora512_rope64_b16",
          "GEMM", lambda s, _: (gemm(s["total_kv"], s["proj_rows"], s["kv_lora_dim"]), None), same, None,
          "FLOPs equal; bytes: the gather and the rope copy are not a GEMM"),
]


def evaluate(check: Check) -> dict:
    """Our floor and TraceLens's counts for the largest performance case of the task."""
    contract = load_task_contract(REPO / "tasks" / check.task_id / TASK_FILE, task_id=check.task_id)
    case = performance_cases(contract)[-1]
    case_shape = case_shapes(contract, case)
    symbols = {**case_shape.symbols, **{k: v for k, v in case_shape.scalars.items() if isinstance(v, (int, float))}}
    floor = case_floor(contract, case, PEAKS)
    their_flops, their_bytes = check.theirs(symbols, case_shape.scalars)
    return {"check": check, "case_id": case["id"], "axes": case["axes"], "floor": floor,
            "flops": (floor["flops_matrix"], their_flops, check.flops(symbols, their_flops)),
            "bytes": (floor["bytes_min"], their_bytes,
                      None if check.bytes is None or their_bytes is None else check.bytes(symbols, their_bytes))}


@pytest.mark.parametrize("check", CHECKS, ids=[c.task_id for c in CHECKS])
def test_family_counts_match_tracelens_up_to_the_explained_difference(check):
    result = evaluate(check)
    ours, theirs, expected = result["flops"]
    assert ours == pytest.approx(expected, rel=1e-12, abs=1), check.why
    ours_bytes, their_bytes, expected_bytes = result["bytes"]
    if check.bytes is not None:
        assert ours_bytes == pytest.approx(expected_bytes, rel=1e-12, abs=1), check.why
