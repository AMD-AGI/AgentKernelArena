"""Hand-computed counts of the non-attention partial-access and data-dependent models (CPU only)."""
from __future__ import annotations

import math

import pytest

from src.roofline.bytes_model import generic_bytes
from src.roofline.generated import DATA_MODELS
from src.roofline.operators import model_for
from tests.roofline.test_roofline_counting import shapes, tensor


class Fake:
    """Stand-in for a generated torch tensor: data models only call .tolist()."""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def data(operator):
    return DATA_MODELS[operator][1]


def test_batched_moe_gemm_counts_valid_rows_and_used_experts():
    # 3 experts, max_tokens 4, K 8, N 16; counts 4, 0, 1 (bf16).
    case = shapes("batched_moe_gemm", {}, [
        tensor("A", (3, 4, 8)), tensor("B", (3, 16, 8), role="weight"),
        tensor("expert_num_tokens", (3,), dtype="int32", role="metadata"),
        tensor("C", (3, 4, 16), direction="output")])
    estimate = data("batched_moe_gemm")(case, {"expert_num_tokens": Fake([4, 0, 1])})
    assert estimate.flops_matrix == {"matrix_bf16": 2 * 5 * 16 * 8}
    assert estimate.bytes_override == 5 * 8 * 2 + 2 * 16 * 8 * 2 + 12 + 3 * 4 * 16 * 2


def test_fused_moe_lora_counts_written_routes_and_pairs_only():
    # 3 tokens, top_k 2 (6 routes), block 2, 2 slots (slot 1 disabled), 2 experts, 1 slice, rank 2,
    # H 4, N 3. Slot 0: blocks [e0: routes 0, 2] [e1: routes 1, 6(pad)] [e-1: ...]; padded total 4.
    symbols = {"top_k": 2, "block_size": 2}
    case = shapes("fused_moe_lora", symbols, [
        tensor("hidden_states", (3, 4)), tensor("lora_a", (1, 2, 2, 2, 4), role="weight"),
        tensor("lora_b", (1, 2, 2, 3, 2), role="weight"),
        tensor("lora_delta", (3, 2, 3), direction="output")])
    inputs = {"sorted_token_ids": Fake([[0, 2, 1, 6, 6, 6], [3, 4, 5, 6, 6, 6]]),
              "expert_ids": Fake([[0, 1, -1], [0, 1, 0]]), "num_tokens_post_padded": Fake([4, 6]),
              "lora_ids": Fake([-1, 0, 1]), "adapter_enabled": Fake([1, 0, 0])}
    estimate = data("fused_moe_lora")(case, inputs)
    # Routes 0, 1, 2 (tokens 0, 0, 1) on pairs (0, e0) and (0, e1).
    assert estimate.flops_matrix == {"matrix_bf16": 3 * 1 * 2 * 2 * (4 + 3)}
    assert estimate.bytes_override == 2 * 4 * 2 + 2 * (2 * 4 + 3 * 2) * 2 + 3 * 2 * 3 * 2


def test_fused_moe_lora_counts_two_routes_of_a_token_on_one_pair_once():
    symbols = {"top_k": 2, "block_size": 2}
    case = shapes("fused_moe_lora", symbols, [
        tensor("hidden_states", (1, 4)), tensor("lora_a", (1, 1, 1, 2, 4), role="weight"),
        tensor("lora_b", (1, 1, 1, 3, 2), role="weight"),
        tensor("lora_delta", (1, 2, 3), direction="output")])
    inputs = {"sorted_token_ids": Fake([[0, 1]]), "expert_ids": Fake([[0]]), "num_tokens_post_padded": Fake([2]),
              "lora_ids": Fake([0, -1]), "adapter_enabled": Fake([1, 0])}
    estimate = data("fused_moe_lora")(case, inputs)
    assert estimate.flops_matrix == {"matrix_bf16": 1 * 2 * 2 * (4 + 3)}


def test_lora_counts_routed_tokens_and_used_adapters():
    # 5 tokens; groups: slot 2 -> tokens [3, 0], slot -1 -> [4], slot 0 -> [1, 2]; 3 slots, rank 2, H 8.
    case = shapes("lora_shrink", {}, [
        tensor("x", (5, 8)), tensor("lora_a", (3, 2, 8), role="weight"),
        tensor("out", (5, 2), dtype="float32", direction="output")])
    inputs = {"token_indices_sorted": Fake([3, 0, 4, 1, 2]), "lora_token_start_loc": Fake([0, 2, 3, 5]),
              "num_tokens_per_lora": Fake([2, 1, 2]), "lora_ids": Fake([2, -1, 0])}
    estimate = data("lora_shrink")(case, inputs)
    assert estimate.flops_matrix == {"matrix_bf16": 2 * 4 * 2 * 8}
    assert estimate.bytes_override == 4 * 8 * 2 + 2 * 2 * 8 * 2 + 5 * 2 * 4


def test_ep_gather_reads_only_valid_assignment_rows():
    case = shapes("ep_gather", {}, [
        tensor("input_tensor", (4, 8)), tensor("recv_topk_ids", (2, 2), dtype="int32", role="index"),
        tensor("recv_topk_weight", (2, 2), dtype="float32", role="weight"),
        tensor("input_index", (2, 2), dtype="int32", role="index"),
        tensor("output_tensor", (2, 8), direction="output")])
    inputs = {"recv_topk_ids": Fake([[5, -1], [7, 9]]), "input_index": Fake([[2, 0], [3, 1]])}
    estimate = data("ep_gather")(case, inputs)
    assert estimate.bytes_override == 3 * 8 * 2 + 4 * 4 + 3 * (4 + 4) + 2 * 8 * 2


def test_masked_silu_quant_reads_only_valid_rows():
    case = shapes("silu_and_mul_fp8_quant_masked", {}, [
        tensor("y", (2, 4, 8)), tensor("tokens_per_expert", (2,), dtype="int32", role="metadata"),
        tensor("y_q", (2, 4, 4), dtype="float8_e4m3fn", direction="output"),
        tensor("y_s", (2, 1, 4), dtype="float32", direction="output")])
    estimate = data("silu_and_mul_fp8_quant_masked")(case, {"tokens_per_expert": Fake([4, 1])})
    assert estimate.bytes_override == 5 * 8 * 2 + 8 + 2 * 4 * 4 + 2 * 4 * 4


def test_reshape_and_cache_counts_distinct_written_slots_both_ways():
    case = shapes("reshape_and_cache_flash_diffkv", {}, [
        tensor("key", (4, 2, 6)), tensor("value", (4, 2, 4)),
        tensor("kv_cache", (8, 2, 10), role="state", mutable=True),
        tensor("slot_mapping", (4,), dtype="int64", role="index")])
    estimate = data("reshape_and_cache_flash_diffkv")(case, {"slot_mapping": Fake([3, -1, 5, 3])})
    assert estimate.bytes_override == 2 * 2 * (2 * 6 * 2 + 2 * 4 * 2) + 4 * 8


def test_fused_rope_cache_reads_v_and_writes_the_caches_only_for_written_slots():
    # 4 tokens, 3 query heads, 2 KV heads, head size 4 (bf16); cache 3 blocks of 2 slots (16 bytes a slot).
    case = shapes("fused_rope_reshape_cache", {}, [
        tensor("q", (4, 3, 4)), tensor("k", (4, 2, 4)), tensor("v", (4, 2, 4)),
        tensor("positions", (4,), dtype="int32", role="index"),
        tensor("cos_cache", (8, 2), role="weight"), tensor("sin_cache", (8, 2), role="weight"),
        tensor("slot_mapping", (4,), dtype="int32", role="index"),
        tensor("key_cache", (3, 2, 2, 4), role="state", mutable=True),
        tensor("value_cache", (3, 2, 2, 4), role="state", mutable=True),
        tensor("q_out", (4, 3, 4), direction="output"), tensor("k_out", (4, 2, 4), direction="output")])
    # Slots 3 and 5 are written (token 3 overwrites token 0); token 1 is padding.
    estimate = data("fused_rope_reshape_cache")(case, {"slot_mapping": Fake([3, -1, 5, 3])})
    q, k, positions, slot_mapping, outputs = 96, 64, 16, 16, 96 + 64
    assert estimate.bytes_override == q + k + positions + slot_mapping + 2 * (16 + 16 + 16) + outputs
    # Without the generated slots (G2) no v row is counted, nor the index inputs and the caches.
    counted = generic_bytes(case)
    assert (counted.rule, counted.bytes) == ("G2", q + k + outputs)


def test_recurrent_decode_skips_padded_sequences_and_negligible_state_reads():
    # 3 sequences, sequence 1 padded; 2 heads, state slice 4 x 4 float32 (64 bytes); mixed_qkv width 6.
    # Head 0: A_log 0, a + dt_bias = -30 -> decay ~ 1 (read). Head 1: A_log = log 20; a + dt_bias = 20
    # gives decay exp(-400) (not read), -30 gives decay ~ 1 (read).
    case = shapes("gdn_fused_recurrent_decode", {}, [
        tensor("mixed_qkv", (3, 6)), tensor("a", (3, 2)), tensor("b", (3, 2)),
        tensor("A_log", (2,), role="weight"), tensor("dt_bias", (2,), role="weight"),
        tensor("ssm_state", (5, 2, 4, 4), dtype="float32", role="state", mutable=True),
        tensor("ssm_state_indices", (3,), dtype="int32", role="index"),
        tensor("out", (3, 1, 2, 4), direction="output")])
    inputs = {"ssm_state_indices": Fake([4, -1, 0]), "a": Fake([[-30.0, 20.0], [0.0, 0.0], [-30.0, -30.0]]),
              "dt_bias": Fake([0.0, 0.0]), "A_log": Fake([0.0, math.log(20.0)])}
    estimate = data("gdn_fused_recurrent_decode")(case, inputs)
    slice_bytes = 4 * 4 * 4
    writes, reads = 2 * 2 * slice_bytes, 3 * slice_bytes
    assert estimate.bytes_override == 3 * 4 + 2 * (6 + 2 + 2) * 2 + writes + reads + 3 * 2 * 4 * 2


def test_gdn_decay_matches_the_reference_formula():
    from src.roofline.operators.partial_data import gdn_decays
    (decay,), = gdn_decays([[0.5]], [0.25], [1.0])
    assert decay == pytest.approx(math.exp(-math.e * math.log1p(math.exp(0.75))))
    assert gdn_decays([[30.0]], [0.0], [0.0]) == [[pytest.approx(math.exp(-30.0))]]


def test_ssd_varlen_counts_tokens_of_each_last_chunk():
    # chunk 4, 10 tokens, sequences [0, 6) -> chunk 1 tokens 4, 5 (2); [6, 7) -> 1; [7, 10) -> chunk 2 (2).
    symbols = {"num_heads": 2, "head_dim": 3, "state_dim": 5, "chunk_size": 4}
    case = shapes("ssd_chunk_state_varlen", symbols, [
        tensor("x", (10, 2, 3)), tensor("B", (10, 1, 5)), tensor("dt", (2, 3, 4), dtype="float32"),
        tensor("dA_cumsum", (2, 3, 4), dtype="float32"), tensor("cu_seqlens", (4,), dtype="int32"),
        tensor("chunk_states", (3, 2, 3, 5), dtype="float32"),
        tensor("initial_states", (3, 2, 3, 5), dtype="float32"),
        tensor("states", (3, 2, 3, 5), dtype="float32", direction="output")])
    estimate = data("ssd_chunk_state_varlen")(case, {"cu_seqlens": Fake([0, 6, 7, 10])})
    tokens = 2 + 1 + 2
    assert estimate.flops_matrix == {"matrix_bf16": 2 * tokens * 2 * 3 * 5}
    per_token = 2 * 3 * 2 + 5 * 2 + 2 * (4 + 4)
    assert estimate.bytes_override == tokens * per_token + 3 * 30 * 4 + 16 + 3 * 30 * 4


def test_topk_transform_reads_scores_only_of_rows_longer_than_k():
    symbols = {"topk": 4}
    case = shapes("topk_transform_paged", symbols, [
        tensor("scores", (3, 20), dtype="float32"), tensor("seq_lens", (3,), dtype="int32"),
        tensor("page_tables", (3, 10), dtype="int32", role="index"),
        tensor("out_page_indices", (3, 4), dtype="int32", direction="output")], scalars={"page_size": 2})
    estimate = data("topk_transform_paged")(case, {"seq_lens": Fake([0, 3, 17])})
    # Row 1: no scores, 2 pages; row 2: 17 scores, ceil(4 / 2) = 2 pages.
    assert estimate.bytes_override == 17 * 4 + 4 * 4 + 3 * 4 + 3 * 4 * 4


def test_cross_entropy_counts_logits_of_rows_not_ignored():
    case = shapes("cross_entropy_label_smoothing", {}, [
        tensor("x", (5, 10), dtype="float32"), tensor("target", (5,), dtype="int64", role="index"),
        tensor("weight", (10,), dtype="float32", role="weight"),
        tensor("losses", (5,), dtype="float32", direction="output"),
        tensor("out", (), dtype="float32", direction="output")], scalars={"ignore_index": 0})
    estimate = data("cross_entropy_label_smoothing")(case, {"target": Fake([0, 3, 0, 9, 1])})
    assert estimate.bytes_override == 3 * 10 * 4 + 5 * 8 + 5 * 4 + 4
    static = model_for("cross_entropy_label_smoothing").estimate(case)
    assert static.bytes_override == 5 * 8 + 5 * 4 + 4
    every_row = shapes("cross_entropy_label_smoothing", {}, case.tensors, scalars={"ignore_index": -100})
    assert data("cross_entropy_label_smoothing")(every_row, {"target": Fake([0] * 5)}).bytes_override == 264


def test_state_update_reads_and_writes_every_slot():
    case = shapes("lightning_attention_decode", {}, [
        tensor("q", (2, 3)), tensor("kv_cache", (2, 3, 3), dtype="float32", role="state", mutable=True),
        tensor("slot_idx", (2,), dtype="int32", role="index"), tensor("out", (2, 3), direction="output")])
    assert model_for("lightning_attention_decode").estimate(case).bytes_override == 12 + 2 * 72 + 8 + 12


def test_selective_state_update_counts_90_percent_of_the_state_write():
    case = shapes("selective_state_update", {}, [
        tensor("x", (2, 3)), tensor("state", (2, 3, 10), role="state", mutable=True),
        tensor("state_batch_indices", (2,), dtype="int32", role="index"), tensor("out", (2, 3), direction="output")])
    estimate = model_for("selective_state_update").estimate(case)
    assert estimate.bytes_override == 12 + 120 + 8 + 108 + 12
    assert "negligible_update_margin" in estimate.notes


def test_cpu_generation_emits_no_data_model_floor():
    from pathlib import Path
    from src.roofline.arch import load_peaks
    from src.roofline.floor import task_floors
    from src.task_contract import TASK_FILE, load_task_contract
    root = Path(__file__).resolve().parents[2] / "tasks"
    peaks = load_peaks("mi355x_datasheet")
    for task_id, tier in (("triton/moe/batched_moe_gemm/bf16_e8_n14336_k4096", "none"),
                          ("hip/reduction/cross_entropy_label_smoothing/f32_c100", "estimated"),
                          ("flydsl/kv_cache/fused_rope_reshape_cache/bf16_q32_kv8_d128_p16", "none"),
                          ("triton/moe/fused_moe/bf16_e8_top2_h4096_i14336", "estimated"),
                          ("gluon/attention/paged_attention_decode/bf16_q16_kv1_d128_p16", "none")):
        contract = load_task_contract(root / task_id / TASK_FILE, task_id=task_id)
        for record in task_floors(contract, peaks, root / task_id, generate=True, device="cpu"):
            assert record["tier"] == tier, task_id
            assert "data_model_requires_gpu_generation" in record["notes"]
            assert not record["model"].startswith(("partial_data/", "attention_data/", "routing_data/"))


def test_failed_generation_falls_back_like_a_cpu_run(monkeypatch):
    from pathlib import Path
    from src.roofline import floor as floor_module
    from src.roofline.arch import load_peaks
    from src.task_contract import TASK_FILE, load_task_contract

    def fail(*args, **kwargs):
        raise RuntimeError("no length tensor")

    monkeypatch.setattr(floor_module, "generated_inputs", fail)
    root = Path(__file__).resolve().parents[2] / "tasks"
    peaks = load_peaks("mi355x_datasheet")
    for task_id, tier in (("flydsl/kv_cache/fused_rope_reshape_cache/bf16_q32_kv8_d128_p16", "none"),
                          ("triton/moe/fused_moe/bf16_e8_top2_h4096_i14336", "estimated")):
        contract = load_task_contract(root / task_id / TASK_FILE, task_id=task_id)
        for record in floor_module.task_floors(contract, peaks, root / task_id, generate=True, device="cuda"):
            assert record["tier"] == tier, task_id
            assert "data_model_failed: RuntimeError: no length tensor" in record["notes"]


def test_generated_floors_record_the_runtime_and_package_they_came_from(monkeypatch):
    from pathlib import Path
    from src.roofline import floor as floor_module
    from src.roofline import generated
    from src.roofline.arch import load_peaks
    from src.roofline.precision_policy import package_sha256
    from src.task_contract import TASK_FILE, load_task_contract
    import src.task_runtime

    identity = {"torch": "2.11.0+rocm10.0.0", "hip": "10.0", "gpu_name": "AMD Instinct MI355X", "gpu_arch": "gfx950",
                "AKA_SCORING_IMAGE_REFERENCE": "image@sha256:" + "1" * 64}
    monkeypatch.setattr(src.task_runtime, "runtime_identity", lambda: dict(identity))
    root = Path(__file__).resolve().parents[2] / "tasks"
    task_id = "triton/moe/fused_moe/bf16_e8_top2_h4096_i14336"
    provenance = generated.generation_provenance(root / task_id)
    assert provenance == {**identity, "package_sha256": package_sha256(root / task_id)}

    seeds = []
    def fake_case_floor(contract, case, peaks, inputs=None, task_dir=None, probe=None):
        seeds.append(inputs["seed"])
        return {"t_floor_us": 1.0 + inputs["seed"], "seed": inputs["seed"], "notes": []}
    monkeypatch.setattr(floor_module, "generated_inputs", lambda *args: {"seed": args[3]})
    monkeypatch.setattr(floor_module, "case_floor", fake_case_floor)
    contract = load_task_contract(root / task_id / TASK_FILE, task_id=task_id)
    records = floor_module.task_floors(contract, load_peaks("mi355x_datasheet"), root / task_id, generate=True,
                                       device="cuda")
    assert records and {r["seed"] for r in records} == {0}  # the smaller of the two seeds' floors
    assert all(r["generated_on"] == "cuda" and r["generation"] == provenance for r in records)


def test_sorting_and_counting_read_every_routing_entry():
    ids = tensor("topk_ids", (4, 2), dtype="int32", role="index")
    case = shapes("count_expert_tokens", {}, [ids, tensor("expert_num_tokens", (8,), dtype="int32",
                                                          direction="output")])
    assert model_for("count_expert_tokens").estimate(case).bytes_override == 32 + 32
    case = shapes("moe_sorting", {}, [ids, tensor("topk_weights", (4, 2), dtype="float32"),
                                      tensor("sorted_ids", (64,), dtype="int32", direction="output"),
                                      tensor("moe_buf", (4, 16), direction="output")])
    assert model_for("moe_sorting").estimate(case).bytes_override == 32 + 32 + 256 + 128


def test_index_select_reads_each_distinct_row_once():
    # 6 selections of 3 distinct rows of a [5, 4] float32 table.
    case = shapes("index_select_rows", {}, [
        tensor("table", (5, 4), dtype="float32", role="weight"), tensor("idx", (6,), dtype="int64", role="index"),
        tensor("out", (6, 4), dtype="float32", direction="output")])
    estimate = data("index_select_rows")(case, {"idx": Fake([4, 0, 4, 2, 0, 4])})
    assert estimate.bytes_override == 3 * 16 + 6 * 8 + 6 * 16
    assert estimate.notes == ["data_dependent: 3 distinct rows among 6 selections"]


def test_temperature_scaling_touches_only_the_rows_it_modifies():
    # 5 rows of 8 float32 logits; rows map to requests 0, 1, 1, 2, 3 with temperatures 0.5, 1, 0 and 2:
    # rows 0 and 4 are scaled, rows 1 to 3 (temperature 1 or 0) are left as they are.
    case = shapes("apply_temperature", {}, [
        tensor("logits", (5, 8), dtype="float32", mutable=True),
        tensor("expanded_idx_mapping", (5,), dtype="int32", role="index"),
        tensor("temperature", (4,), dtype="float32", role="metadata")])
    estimate = data("apply_temperature")(case, {"expanded_idx_mapping": Fake([0, 1, 1, 2, 3]),
                                               "temperature": Fake([0.5, 1.0, 0.0, 2.0])})
    assert estimate.bytes_override == 2 * 2 * 32 + 5 * 4 + 4 * 4
    assert estimate.notes == ["data_dependent: 2 of 5 rows modified"]


def test_jagged_broadcast_add_does_not_count_dense_rows():
    case = shapes("jagged_dense_broadcast_add", {}, [
        tensor("seq_offsets", (3,), dtype="int64", role="metadata"), tensor("jagged", (10, 4)),
        tensor("dense", (2, 4)), tensor("out", (10, 4), direction="output")])
    counted = generic_bytes(case)
    assert (counted.rule, counted.bytes) == ("G1", 24 + 80 + 80)


def test_rms_norm_backward_does_not_count_the_recomputable_rsigma():
    case = shapes("rms_norm_backward", {}, [
        tensor("dy", (2, 4)), tensor("x", (2, 4)), tensor("gamma", (4,), role="weight"),
        tensor("rsigma", (2,), dtype="float32"),
        tensor("dx", (2, 4), direction="output"), tensor("dgamma", (4,), direction="output")])
    counted = generic_bytes(case)
    assert (counted.rule, counted.bytes) == ("G1", 16 + 16 + 8 + 16 + 8)
