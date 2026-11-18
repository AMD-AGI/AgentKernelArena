"""Hand-computed checks of the whole-prefill linear-attention models (KDA and lightning)."""
from __future__ import annotations

from pathlib import Path

from src.roofline.arch import load_peaks
from src.roofline.floor import task_floors
from src.roofline.interface import CaseShapes, Tensor
from src.roofline.operators import model_for
from src.roofline.operators.chunked import chunk_pairs
from src.task_contract import TASK_FILE, load_task_contract

ROOT = Path(__file__).resolve().parents[2]
PEAKS = load_peaks("mi355x_datasheet")


def tensor(name, shape, dtype="bfloat16"):
    return Tensor(name, "input", "activation", dtype, tuple(shape), tuple(shape))


def shapes(operator, symbols, tensors):
    return CaseShapes("synthetic/" + operator, "case", operator, "test", dict(symbols), dict(symbols), tuple(tensors))


def test_kda_chunk_attention_counts_two_dense_state_products_per_token_and_head():
    # 2 sequences x 3 tokens, 12 heads, d = 128: 4 * 6 * 128 * 128 * 12 = 4718592 FLOPs.
    symbols = {"n_seqs": 2, "seq_len": 3, "num_heads": 12, "head_dim": 128}
    case = shapes("kda_chunk_attention", symbols, [tensor("q", (2, 3, 12, 128)), tensor("k", (2, 3, 12, 128))])
    estimate = model_for("kda_chunk_attention").estimate(case)
    assert estimate.flops_matrix == {"matrix_bf16": 4718592}
    assert estimate.bytes_override is None  # the generic rule G2 applies (initial_state is a state input)
    assert "decay_truncation_sensitive" in estimate.notes


def test_lightning_prefill_counts_two_dense_state_products_per_token_and_head():
    # 1000 tokens, 64 heads, d = 128: 4 * 1000 * 128 * 128 * 64 = 4194304000 FLOPs.
    symbols = {"n_tokens": 1000, "num_heads": 64, "head_dim": 128}
    case = shapes("lightning_attention_prefill", symbols,
                  [tensor("q", (64, 1000, 128)), tensor("k", (64, 1000, 128))])
    assert model_for("lightning_attention_prefill").estimate(case).flops_matrix == {"matrix_bf16": 4194304000}


def test_whole_prefill_bound_never_exceeds_the_chunked_pipeline_or_the_recurrence():
    # The chunked KDA pipeline per head (kkt, w/u recompute, state pass, output), for any chunk size,
    # costs at least the whole-prefill bound, which drops every intra-chunk product.
    d = 128
    for tokens in (1, 63, 64, 65, 1000):
        for chunk in (1, 16, 64):
            strict, causal = chunk_pairs(tokens, chunk, strict=True), chunk_pairs(tokens, chunk, strict=False)
            chunked = (2 * d * (strict + causal)                # kda_chunk_kkt
                       + 2 * 2 * d * causal                     # kda_recompute_w_u (w and u)
                       + 2 * tokens * d * d * 2                 # v_new = u - w S^T and S += k^T v_new
                       + 2 * tokens * d * d + 2 * causal * d)   # kda_chunk_fwd_o
            case = shapes("kda_chunk_attention", {"n_seqs": 1, "seq_len": tokens, "num_heads": 1, "head_dim": d},
                          [tensor("q", (1, tokens, 1, d)), tensor("k", (1, tokens, 1, d))])
            bound = model_for("kda_chunk_attention").estimate(case).flops_matrix["matrix_bf16"]
            assert bound <= chunked
            # The plain recurrence: S k, the rank-1 update and S q, 2 d^2 each per token.
            assert bound <= 6 * tokens * d * d


def test_real_tasks_are_memory_bound_estimated_floors():
    for task_id in ("triton/state_space/kda_chunk_attention/bf16_h12_d128",
                    "triton/state_space/lightning_attention_prefill/bf16_h64_d128_b256"):
        floors = task_floors(load_task_contract(ROOT / "tasks" / task_id / TASK_FILE, task_id=task_id), PEAKS)
        assert floors
        for floor in floors:
            assert (floor["tier"], floor["bound"], floor["bytes_rule"]) == ("estimated", "memory", "G2"), floor
            assert floor["t_compute_matrix_us"] < 0.5 * floor["t_memory_us"]


def _floors(task_prefix):
    from src.roofline.__main__ import discover_tasks
    (task_id,) = [t for t in discover_tasks() if t.startswith(task_prefix)]
    return task_floors(load_task_contract(ROOT / "tasks" / task_id / TASK_FILE, task_id=task_id), PEAKS)


def test_gaussian_non_local_block_counts_the_dot_product_blocks_products():
    # B = 1, N = hw = 4, C = 64, qk = 16: the same 84480 FLOPs as nonlocal_attention_dot (see
    # test_roofline_composite_models.py); only the energy nonlinearity differs.
    from src.roofline.floor import case_floor
    task_id = "hip/attention/nonlocal_attention_gaussian/f32_c64"
    contract = load_task_contract(ROOT / "tasks" / task_id / TASK_FILE, task_id=task_id)
    floor = case_floor(contract, {"id": "hand", "axes": {"batch": 1, "hw": 4}, "checks": ["performance"]}, PEAKS)
    assert floor["flops_matrix_by_precision"] == {"matrix_bf16": 84480}  # float32 task, no probe: one bf16 product
    assert floor["tier"] == "estimated"


def test_topk_and_token_ranks_get_byte_floors():
    # Both must read every logit once (any element can enter the top k or change a rank) and do about
    # one compare per element, so the byte rule gives a meaningful floor.
    for prefix, rule in (("triton/sorting/topk/", "G1"), ("triton/sampling/token_ranks/", "G2")):
        floors = _floors(prefix)
        assert {(f["tier"], f["bound"], f["bytes_rule"]) for f in floors} == {("estimated", "memory", rule)}
