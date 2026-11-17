"""Hand-computed FLOP and byte counts of the composite, attention and grouped models.

Each test evaluates a small case of a real task contract, so the tensor names, roles and dtypes
the model reads are the declared ones. The expected numbers are worked out in the comments from
the shapes alone, independently of the model code.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.roofline.arch import load_peaks
from src.roofline.floor import case_floor
from src.task_contract import TASK_FILE, load_task_contract

ROOT = Path(__file__).resolve().parents[2]
PEAKS = load_peaks("mi355x_datasheet")


def floor(task_id, **axes):
    contract = load_task_contract(ROOT / "tasks" / task_id / TASK_FILE, task_id=task_id)
    return case_floor(contract, {"id": "hand", "axes": axes, "checks": ["performance"]}, PEAKS)


def test_mingpt_block():
    # B = 1, S = 4 (4 rows), E = 768, fp32. QKV 2*4*2304*768 = 14155776; causal pairs 4*5/2 = 10, so
    # scores + PV 4*768*10 = 30720; proj 2*4*768*768 = 4718592; fc and mlp_proj 2*4*3072*768 = 18874368
    # each. Total 56653824. Bytes (G1): x and out 4*768*4 each, every weight and bias: 4 * (2*3072 +
    # 4*768 + 3*768^2 + 3*768 + 768^2 + 768 + 4*768^2 + 4*768 + 4*768^2 + 768) = 28376064.
    f = floor("triton/other/mingpt_block/f32_e768_h8", batch=1, seq_len=4)
    assert f["flops_matrix_by_precision"] == {"matrix_fp32": 56653824}  # float32 task
    assert (f["bytes_rule"], f["bytes_min"]) == ("G1", 28376064)


def test_causal_self_attention():
    # Same QKV (14155776), scores + PV (30720) and output projection (4718592) as mingpt: 18905088.
    # Bytes: 4 * (2*4*768 + 3*768^2 + 3*768 + 768^2 + 768) = 9474048.
    f = floor("triton/attention/causal_self_attention/f32_e768_h8", batch=1, seq_len=4)
    assert f["flops_matrix_by_precision"] == {"matrix_fp32": 18905088}  # float32 task
    assert (f["bytes_rule"], f["bytes_min"]) == ("G1", 9474048)


def test_vision_attention_block():
    # B = 1, hw = 4, E = 128, unmasked: in-proj 2*4*384*128 = 393216, scores + PV 4*128*4*4 = 8192,
    # out-proj 2*4*128*128 = 131072; 532480. Bytes: 4 * (2*128*4 + 3*128^2 + 3*128 + 128^2 + 128 + 2*128)
    # = 269312.
    f = floor("triton/attention/vision_attention_block/f32_e128_h4", batch=1, hw=4)
    assert f["flops_matrix_by_precision"] == {"matrix_fp32": 532480}  # float32 task
    assert f["bytes_min"] == 269312


def test_nonlocal_attention_dot():
    # B = 1, N = hw = 4, C = 64, qk = 16, in the reference's order. Projections: q and k (4*N*C*qk =
    # 16384), v and the output map (4*N*C^2 = 65536). Energy qᵀk: 2*N^2*qk = 512. v E: 2*N^2*C = 2048.
    # Total 84480. Bytes: 4 * (2*C*N + 2*qk*C + 2*qk + 2*C^2 + 2*C) = 43648.
    f = floor("hip/attention/nonlocal_attention_dot/f32_c64", batch=1, hw=4)
    assert f["flops_matrix_by_precision"] == {"matrix_fp32": 84480}  # float32 task
    assert f["bytes_min"] == 43648


@pytest.mark.parametrize("task_id, qk_key", [("triton/attention/sage_attention_fwd/int8qk_fp8v_h8_d128", "matrix_int8"),
                                             ("triton/attention/sage_attention_fwd/mxfp4qk_h8_d128", "matrix_fp4")])
def test_sage_attention_is_charged_in_the_formats_the_operator_quantizes_to(task_id, qk_key):
    # B = 1, S = 4, 8 heads of 128: causal pairs 10. q, k, v are bf16 and the operator quantizes them:
    # QK^T (2*128*8*10 = 20480 FLOPs) in int8 or MXFP4, PV (20480) in fp8.
    # Bytes: q, k, v, out 4*8*128*2 each = 32768.
    f = floor(task_id, batch=1, seqlen=4)
    assert f["flops_matrix_by_precision"] == {qk_key: 20480, "matrix_fp8": 20480}
    assert f["bytes_min"] == 32768


def test_mhc_fused_post_pre():
    # tokens 2, hc 4, hidden 7168: mix_dim 4*6 = 24, hc_hidden 28672; 2*2*24*28672 = 2752512, bf16
    # residual times fp32 fn, charged at bf16. Bytes: x 2*7168*2, residual 2*4*7168*2, post 2*4*4,
    # comb 2*16*4, fn 24*28672*4, hc_scale 12, hc_base 24*4; outputs 2*4*7168*2 + 2*4*4 + 2*16*4 +
    # 2*7168*2. Total 3039660.
    f = floor("tilelang/other/mhc_fused_post_pre/c4_d7168", tokens=2)
    assert f["flops_matrix_by_precision"] == {"matrix_bf16": 2752512}
    assert f["bytes_min"] == 3039660


def test_mhc_fused_post_pre_rmsnorm():
    # hidden 4096: 2*2*24*16384 = 1572864. Bytes as above with hidden 4096 plus norm_weight 4096*2:
    # 1745324.
    f = floor("triton/other/mhc_fused_post_pre_rmsnorm/c4_d4096", tokens=2)
    assert f["flops_matrix_by_precision"] == {"matrix_bf16": 1572864}
    assert f["bytes_min"] == 1745324


def test_ssd_bmm_chunk():
    # 300 tokens in ceil(300/256) = 2 chunks; the full n x n Gram block per chunk and group, smallest
    # for equal chunks: sum n^2 >= 300^2/2 = 45000, so 2*8*128*45000 = 92160000. Bytes: C and B
    # 300*8*128*2 each, out 2*8*256*256*4 (written whole); cu_chunk_seqlens has a rounded-up extent
    # and is left out. 5423104.
    f = floor("triton/state_space/ssd_bmm_chunk/bf16_g8_n128_c256", n_tokens=300)
    assert f["flops_matrix_by_precision"] == {"matrix_bf16": 92160000}
    assert f["bytes_min"] == 5423104


def test_ssd_chunk_scan_and_its_bytes_override():
    # 512 tokens, 2 chunks: causal pairs >= 512^2/4 + 512/2 = 65792; 2*128 heads*64*65792 = 1077936128.
    # Bytes: x and out 512*128*64*2 each, C 512*8*128*2, D 128*4, the causal triangle of cb 8*65792*2,
    # dt and dA_cumsum at the real tokens 128*512*4 each; states and the chunk metadata are left out.
    # 19403264.
    f = floor("triton/state_space/ssd_chunk_scan/bf16_h128_p64_g8_n128_c256", n_tokens=512)
    assert f["flops_matrix_by_precision"] == {"matrix_bf16": 1077936128}
    assert (f["bytes_rule"], f["bytes_min"]) == ("operator", 19403264)


def test_fused_bmm_rope_kv_cache():
    # 4 tokens, 128 q heads: q_nope [128, 4, 128] times w_kᵀ [128 -> 512] per head, fp8 after the
    # in-kernel quantization: 2*128*4*512*128 = 67108864. Bytes: the G2 count (q_nope 128*4*128*2,
    # q_pe 4*128*64*2, k_nope 4*512*2, k_pe 4*64*2, q_out 4*128*576*2, k_pe_out 4*64*2 = 791552), plus
    # the operands and writes G2 leaves out: w_k 128*512*128, its scale 4, one 576*2 cache row per token.
    f = floor("triton/kv_cache/fused_bmm_rope_kv_cache/fp8w_bf16_h128_nope128_lora512_rope64", n_tokens=4)
    assert f["flops_matrix_by_precision"] == {"matrix_fp8": 67108864}
    assert (f["bytes_rule"], f["bytes_min"]) == ("operator", 791552 + 128 * 512 * 128 + 4 + 4 * 576 * 2)


def test_gather_kv_b_proj_reads_only_the_gathered_rows():
    # 10 gathered rows of 512 + 64 latent: projection 2*10*4096*512 = 41943040, bf16 latents against
    # fp8 weights (charged at the fp8 peak). Bytes: 10 rows
    # of k_buffer (576*2 each, not the 262144-block pool), kv_indices 10*4, the fp8 weight 4096*512,
    # its scales 32*4*4, k_prefix 10*16*192*2 and v_prefix 10*16*128*2; per-request offsets are not
    # needed. 2211624.
    f = floor("triton/kv_cache/gather_kv_b_proj/bf16kv_fp8w_h16_nope128_v128_lora512_rope64_b16", total_kv=10)
    assert f["flops_matrix_by_precision"] == {"matrix_fp8": 41943040}
    assert (f["bytes_rule"], f["bytes_min"]) == ("operator", 2211624)


def test_grouped_gemm_counts_one_groups_weights():
    # m = 4 rows, k 1408, n 2048: 2*4*2048*1408 = 23068672. Bytes: one group's rhs slice 1408*2048*2,
    # lhs 4*1408*2, out 4*2048*2; group_sizes is not needed in full. 5794816.
    f = floor("triton/gemm/grouped_gemm/bf16_g64_k1408_n2048", m=4)
    assert f["flops_matrix_by_precision"] == {"matrix_bf16": 23068672}
    assert (f["bytes_rule"], f["bytes_min"]) == ("operator", 5794816)
