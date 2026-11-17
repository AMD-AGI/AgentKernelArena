"""Counting rules of the prefill, extend and sparse attention data models, checked by hand or brute force."""
from __future__ import annotations

import itertools

import pytest

from src.roofline.generated import DATA_MODELS
from src.roofline.interface import CaseShapes, Tensor
from src.roofline.operators import attention_data
from src.roofline.operators.attention_data import (bottom_right_keys, bottom_right_pairs, contract_window,
                                                   hstu_pairs, merged_length)


def tensor(name, shape, dtype="bfloat16", direction="input", role="activation", expressions=None):
    expressions = tuple(expressions) if expressions is not None else tuple(shape)
    return Tensor(name, direction, role if direction == "input" else "output", dtype, tuple(shape), expressions)


def shapes(operator, symbols, tensors, scalars=None, task_id=None):
    return CaseShapes(task_id or "synthetic/" + operator, "case", operator, "attention", dict(symbols),
                      dict(symbols), tuple(tensors), dict(scalars or {}))


class FakeTensor:
    """Stand-in for a generated torch tensor: data models only call .tolist()."""

    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


def run(operator, case, inputs):
    return DATA_MODELS[operator][1](case, {name: FakeTensor(value) for name, value in inputs.items()})


def test_bottom_right_counts_match_brute_force():
    for q_len, kv_len, window in itertools.product(range(0, 7), range(0, 9), (None, 1, 2, 3, 5)):
        visible = [(r, c) for r in range(q_len) for c in range(kv_len)
                   if c <= kv_len - q_len + r and (window is None or kv_len - q_len + r - c < window)]
        assert bottom_right_pairs(q_len, kv_len, window) == len(visible)
        if q_len <= kv_len:  # every contract here has at least as many keys as queries
            assert bottom_right_keys(q_len, kv_len, window) == len({c for _, c in visible})


def test_merged_length_counts_the_union_of_intervals():
    assert merged_length([(0, 4), (2, 6), (8, 9), (3, 3), (5, 2)]) == 7


def _fake_task(root, task_id, description, reference="def reference():\n    pass\n"):
    task = root / task_id
    task.mkdir(parents=True)
    (task / "task.yaml").write_text(f"description: {description}\nreference: {{file: reference.py}}\n")
    (task / "reference.py").write_text(reference)


def test_contract_window_comes_from_the_description_and_task_id(tmp_path, monkeypatch):
    assert contract_window("triton/attention/unified_attention/bf16_q8_kv4_d256_p16_swa1024") == 1024
    assert contract_window("triton/attention/unified_attention/bf16_q32_kv8_d128_p16") is None
    assert contract_window("triton/attention/unified_attention/bf16_q12_kv2_d128_p16") is None
    monkeypatch.setattr(attention_data, "TASKS_DIR", tmp_path)
    window = contract_window.__wrapped__
    _fake_task(tmp_path, "a/b_swa512", "causal attention")
    assert window("a/b_swa512") == 512
    _fake_task(tmp_path, "a/c_swa512", "causal attention (sliding window 512)")
    assert window("a/c_swa512") == 512
    _fake_task(tmp_path, "a/d_swa256", "causal attention (sliding window 512)")
    _fake_task(tmp_path, "a/e", "attends within a local window of keys")
    _fake_task(tmp_path, "a/f", "causal attention", "WINDOW = 64\n")
    _fake_task(tmp_path, "a/g", "causal attention", "# A query sees the last 64 tokens\n")
    for task_id in ("a/d_swa256", "a/e", "a/f", "a/g"):
        with pytest.raises(ValueError):
            window(task_id)
    _fake_task(tmp_path, "a/h", "plain causal attention")
    assert window("a/h") is None


def test_unified_attention_counts_windowed_causal_pairs_and_used_keys():
    symbols = {"num_q_heads": 8, "num_kv_heads": 4, "head_dim": 256}
    case = shapes("unified_attention", symbols, [
        tensor("q", (3, 8, 256)), tensor("kv_cache", (300, 16, 4, 512), role="state"),
        tensor("out", (3, 8, 256), direction="output")],
        task_id="triton/attention/unified_attention/bf16_q8_kv4_d256_p16_swa1024")
    # Two decode sequences (2000 and 500 keys) and one of 1 query row: window 1024.
    estimate = run("unified_attention", case, {"cu_seqlens_q": [0, 1, 2, 3], "seqused_k": [2000, 500, 1]})
    keys = 1024 + 500 + 1
    assert estimate.flops_matrix == {"matrix_bf16": 4 * 256 * 8 * keys}
    assert estimate.bytes_override == keys * 4 * 512 * 2 + 2 * 3 * 8 * 256 * 2


def test_unified_attention_without_window_counts_causal_prefill_pairs():
    symbols = {"num_q_heads": 2, "num_kv_heads": 1, "head_dim": 4}
    case = shapes("unified_attention", symbols, [
        tensor("q", (4, 2, 4)), tensor("k_cache", (2, 16, 1, 4), role="state"),
        tensor("v_cache", (2, 16, 1, 4), role="state"), tensor("out", (4, 2, 4), direction="output")],
        task_id="triton/attention/unified_attention/bf16_q12_kv2_d128_p16")
    # One sequence: 4 query rows at positions 6..9 of 10 keys -> 7 + 8 + 9 + 10 pairs.
    estimate = run("unified_attention", case, {"cu_seqlens_q": [0, 4], "seqused_k": [10]})
    assert estimate.flops_matrix == {"matrix_bf16": 4 * 4 * 2 * 34}
    assert estimate.bytes_override == 10 * 1 * 2 * 4 * 2 + 2 * 4 * 2 * 4 * 2


def test_batch_prefill_reads_kv_lengths_from_pages_and_last_page_lens():
    symbols = {"num_q_heads": 8, "num_kv_heads": 1, "head_size": 128, "page_size": 16}
    case = shapes("batch_prefill_paged", symbols, [
        tensor("q", (7, 8, 128)), tensor("k_cache", (10, 16, 1, 128)), tensor("v_cache", (10, 16, 1, 128)),
        tensor("out", (7, 8, 128), direction="output")])
    # Sequence 0: 3 queries over 2 pages, last holding 4 -> 20 keys; sequence 1: 4 queries, 1 page of 4.
    estimate = run("batch_prefill_paged", case, {"cu_seqlens_q": [0, 3, 7], "kv_indptr": [0, 2, 3],
                                                 "kv_last_page_lens": [4, 4]})
    pairs = (18 + 19 + 20) + (1 + 2 + 3 + 4)
    assert estimate.flops_matrix == {"matrix_bf16": 4 * 128 * 8 * pairs}
    assert estimate.bytes_override == (20 + 4) * 128 * 2 * 2 + 2 * 7 * 8 * 128 * 2


def test_prefix_prefill_counts_cached_and_causal_new_pairs():
    case = shapes("prefix_prefill_attention", {}, [
        tensor("q", (5, 4, 8)), tensor("k", (5, 2, 8)), tensor("v", (5, 2, 8)),
        tensor("k_cache", (4, 2, 1, 16, 8)), tensor("v_cache", (4, 2, 8, 16)),
        tensor("out", (5, 4, 8), direction="output")])
    # Sequence 0: 2 new tokens after 10 cached; sequence 1: 3 new after 20 cached.
    estimate = run("prefix_prefill_attention", case, {"b_start_loc": [0, 2, 5], "b_seq_len": [12, 23]})
    pairs = (2 * 10 + 3) + (3 * 20 + 6)
    assert estimate.flops_matrix == {"matrix_bf16": 4 * 8 * 4 * pairs}
    dense = 5 * 4 * 8 * 2 + 2 * 5 * 2 * 8 * 2
    assert estimate.bytes_override == dense + 30 * 2 * 8 * 2 * 2 + 5 * 4 * 8 * 2


def test_prefix_prefill_alibi_counts_every_causal_pair_whatever_the_slopes():
    case = shapes("prefix_prefill_attention_alibi", {}, [
        tensor("q", (3, 2, 8), dtype="float16"), tensor("k", (3, 2, 8), dtype="float16"),
        tensor("v", (3, 2, 8), dtype="float16"), tensor("k_cache", (1, 2, 1, 16, 8), dtype="float16"),
        tensor("v_cache", (1, 2, 8, 16), dtype="float16"),
        tensor("alibi_slopes", (2,), dtype="float32", role="weight"),
        tensor("out", (3, 2, 8), dtype="float16", direction="output")])
    # One sequence: 3 new tokens (positions 10..12) after 10 cached; both heads see 11 + 12 + 13 pairs.
    estimate = run("prefix_prefill_attention_alibi", case,
                   {"b_start_loc": [0, 3], "b_seq_len": [13], "alibi_slopes": [2.0, 0.5]})
    assert estimate.flops_matrix == {"matrix_fp16": 4 * 8 * 2 * (11 + 12 + 13)}
    # q, k, v whole, the slopes, 10 cached keys and values per KV head, out.
    assert estimate.bytes_override == 3 * (3 * 2 * 8 * 2) + 2 * 4 + 10 * 2 * 8 * (2 + 2) + 3 * 2 * 8 * 2


def test_lean_decode_counts_distinct_blocks_per_head_and_request():
    case = shapes("lean_attention_paged_decode", {}, [
        tensor("q", (2, 32, 8), dtype="float16"), tensor("k", (2, 96, 8), dtype="float16"),
        tensor("v", (2, 96, 8), dtype="float16"), tensor("out", (2, 32, 8), dtype="float16", direction="output")])
    # Two requests of 16 query rows: blocks [0, 2) and [2, 5); head 1 repeats block 4 in request 1.
    estimate = run("lean_attention_paged_decode", case,
                   {"batch_num_block_n": [2, 5], "kv_block_tables": [[0, 1, 2, 3, 4, 5], [5, 3, 4, 4, 1, 0]]})
    pairs = 16 * 16 * (2 + 3) + 16 * 16 * (2 + 2)
    assert estimate.flops_matrix == {"matrix_fp16": 4 * 8 * pairs}
    assert estimate.bytes_override == (5 + 4) * 16 * 8 * 2 * 2 + 2 * 2 * 32 * 8 * 2


def _hstu_reference_pairs(length, contextual, targets):
    """Brute force of the reference mask (reference.py of the hstu_attention task)."""
    max_ids = length
    ids = list(range(length))
    if contextual > 0:
        ids = [max(i - contextual + 1, 0) for i in ids]
        max_ids = max_ids - contextual + 1
    max_ids -= targets
    ids = [min(i, max_ids) for i in ids]
    return sum(1 for i in range(length) for j in range(length)
               if i == j or ids[i] - ids[j] > 0 or (contextual > 0 and ids[i] == 0 and ids[j] < max_ids))


def test_hstu_pairs_match_the_reference_mask():
    for length in range(1, 12):
        for contextual in range(0, 4):
            for targets in range(0, max(1, length - contextual)):
                assert hstu_pairs(length, contextual, targets) == _hstu_reference_pairs(length, contextual, targets)


def test_hstu_counts_visible_pairs_of_every_sequence():
    case = shapes("hstu_attention", {"n_heads": 4, "head_dim": 128}, [
        tensor("q", (9, 4, 128)), tensor("k", (9, 4, 128)), tensor("v", (9, 4, 128)),
        tensor("out", (9, 4, 128), direction="output")])
    # Sequence 0: 5 tokens, last 2 targets -> 1+2+3 history pairs + 2 * (3 + 1); sequence 1: 4, none.
    estimate = run("hstu_attention", case, {"seq_offsets": [0, 5, 9], "num_targets": [2, 0]})
    pairs = (1 + 2 + 3 + 4 + 4) + (1 + 2 + 3 + 4)
    assert estimate.flops_matrix == {"matrix_bf16": 4 * 128 * 4 * pairs}
    assert estimate.bytes_override == 4 * 9 * 4 * 128 * 2


def test_paged_mqa_logits_counts_scored_keys_and_whole_logits():
    symbols = {"next_n": 2, "num_heads": 2, "head_size": 4, "num_blocks": 10, "block_size": 1, "num_rows": 4}
    case = shapes("paged_mqa_logits", symbols, [
        tensor("q", (2, 2, 2, 4), dtype="float8_e4m3fn"), tensor("kv_cache", (10, 1, 1, 8), dtype="uint8"),
        tensor("weights", (4, 2), dtype="float32", role="weight"),
        tensor("logits", (4, 32), dtype="float32", direction="output")])
    # Lengths 5 and 3: rows see 4, 5 and 2, 3 keys.
    estimate = run("paged_mqa_logits", case, {"context_lens": [5, 3]})
    assert estimate.flops_matrix == {"matrix_fp8": 2 * 4 * 2 * 14}
    assert estimate.bytes_override == 8 * 8 + 4 * (2 * 4 + 2 * 4) + 4 * 32 * 4


def test_paged_mqa_logits_fp4_reads_key_scales_and_skips_padded_query_scales():
    symbols = {"next_n": 1, "num_heads": 16, "head_size": 128, "num_blocks": 2, "block_size": 64, "num_rows": 1}
    case = shapes("paged_mqa_logits", symbols, [
        tensor("q_fp4", (1, 1, 16, 64), dtype="uint8"),
        tensor("q_scale", (1, 1, 1, 4, 16, 4), dtype="uint8", expressions=(1, 1, 1, 4, 16, "scale_pad")),
        tensor("kv_cache", (2, 1, 4, 64, 16), dtype="uint8"), tensor("kv_scale", (2, 1, 4, 64), dtype="uint8"),
        tensor("weights", (1, 16), role="weight"), tensor("logits", (1, 256), dtype="float32", direction="output")])
    case = CaseShapes(case.task_id, case.case_id, case.operator, case.category, case.axes, case.symbols,
                      tuple(Tensor(t.name, t.direction, t.role, t.dtype, t.shape, t.expressions,
                                   rounded_axes=frozenset({"scale_pad"})) for t in case.tensors))
    estimate = run("paged_mqa_logits", case, {"context_lens": [100]})
    assert estimate.flops_matrix == {"matrix_fp4": 2 * 128 * 16 * 100}
    assert estimate.bytes_override == 100 * (64 + 4) + (16 * 64 + 16 * 2) + 256 * 4


def test_fp8_mqa_logits_counts_window_pairs_and_the_union_of_windows():
    case = shapes("fp8_mqa_logits", {"num_heads": 2, "head_size": 8}, [
        tensor("q", (3, 2, 8), dtype="float8_e4m3fn"), tensor("kv", (10, 8), dtype="float8_e4m3fn"),
        tensor("kv_scales", (10,), dtype="float32", role="weight"),
        tensor("weights", (3, 2), dtype="float32", role="weight"),
        tensor("logits", (3, 10), dtype="float32", direction="output")])
    estimate = run("fp8_mqa_logits", case, {"cu_seqlen_ks": [0, 0, 6], "cu_seqlen_ke": [2, 3, 6]})
    assert estimate.flops_matrix == {"matrix_fp8": 2 * 8 * 2 * 5}
    assert estimate.bytes_override == 3 * (8 + 4) + 2 * (2 * 8 + 2 * 4) + 3 * 10 * 4


def test_mla_decode_reads_each_latent_row_once_and_takes_values_from_its_lora_part():
    # 2 heads, a 12-wide latent row [kv_lora 8 | rope 4] that is both the key and (its first 8) the value.
    symbols = {"num_heads": 2, "head_size": 12, "kv_lora_rank": 8}
    case = shapes("mla_decode", symbols, [
        tensor("q", (3, 2, 12)), tensor("kv_cache", (40, 1, 1, 12), role="state"),
        tensor("out", (3, 2, 8), direction="output")])
    estimate = run("mla_decode", case, {"kv_indptr": [0, 5, 5, 12]})
    keys = 5 + 0 + 7
    # QK^T over the 12-wide row, PV over its 8-wide value part.
    assert estimate.flops_matrix == {"matrix_bf16": 2 * (12 + 8) * 2 * keys}
    # One read of each latent row serves both: keys * 12 * 2 bytes, not twice that.
    assert estimate.bytes_override == keys * 12 * 2 + 3 * 2 * 12 * 2 + 3 * 2 * 8 * 2


def test_sparse_mla_prefill_counts_distinct_valid_rows():
    case = shapes("sparse_mla_prefill", {"head_dim": 16, "num_heads": 2}, [
        tensor("q", (3, 2, 16)), tensor("kv", (8, 16)), tensor("attn_sink", (2,), dtype="float32", role="weight"),
        tensor("out", (3, 2, 16), direction="output")])
    # Row 0: {1, 3} (3 repeated), row 1 empty, row 2: {3, 7} (-1 and 9 are skipped).
    estimate = run("sparse_mla_prefill", case, {"indptr": [0, 3, 3, 7], "indices": [1, 3, 3, 3, 7, -1, 9, -1]})
    assert estimate.flops_matrix == {"matrix_bf16": 2 * 32 * 2 * 4}
    assert estimate.bytes_override == 3 * 16 * 2 + 2 * (2 * 16 * 2) + 8 + 3 * 2 * 16 * 2


def test_sparse_mla_attention_reads_full_rows_and_multiplies_by_the_latent_values():
    symbols = {"head_size": 576, "kv_lora_rank": 512, "num_heads": 16}
    case = shapes("sparse_mla_attention", symbols, [
        tensor("q", (2, 16, 576)), tensor("kv", (1, 64, 1, 576)), tensor("out", (2, 16, 512), direction="output")])
    estimate = run("sparse_mla_attention", case, {"topk_indices": [[0, -1, -1], [0, 1, 0]]})
    assert estimate.flops_matrix == {"matrix_bf16": 2 * (576 + 512) * 16 * 3}
    assert estimate.bytes_override == 2 * 576 * 2 + 2 * 16 * 576 * 2 + 2 * 16 * 512 * 2


def test_flash_mla_sparse_decode_adds_both_pools_and_drops_the_pad_byte():
    symbols = {"query_length": 1, "heads": 2, "head_dim": 512}
    case = shapes("flash_mla_sparse_decode_fp8", symbols, [
        tensor("q", (3, 1, 2, 512)), tensor("kv_cache", (1, 4, 1, 584), dtype="float8_e4m3fn", role="state"),
        tensor("extra_kv_cache", (2, 2, 1, 584), dtype="float8_e4m3fn", role="state"),
        tensor("sinks", (2,), dtype="float32", role="weight"),
        tensor("output", (3, 1, 2, 512), direction="output"), tensor("lse", (3, 1, 2), dtype="float32",
                                                                     direction="output")])
    # Main: row 0 {0, 1}, row 1 empty (length 0), row 2 {1} (the slot past its prefix is ignored).
    # Extra: row 0 {3}, row 1 empty (only -1), row 2 {3, 2}.
    estimate = run("flash_mla_sparse_decode_fp8", case, {
        "sparse_indices": [[[0, 1, 1]], [[2, 3, 0]], [[1, -1, 2]]], "sparse_lens": [3, 0, 2],
        "extra_sparse_indices": [[[3, -1]], [[-1, 0]], [[3, 2]]], "extra_sparse_lens": [2, 1, 2]})
    # bfloat16 queries against fp8 keys: charged at the fp8 peak.
    assert estimate.flops_matrix == {"matrix_fp8": 2 * 1024 * 2 * (3 + 0 + 3)}
    assert "precision_substitution_sensitive" in estimate.notes
    assert estimate.bytes_override == (2 + 2) * 583 + 2 * (2 * 512 * 2) + 8 + 3 * 2 * 512 * 2 + 3 * 2 * 4
