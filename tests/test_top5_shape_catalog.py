"""Static shape inventory checks: never import task code or deserialize fixtures."""

import hashlib
import json
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1] / "tasks" / "head_kernels"


def catalogs():
    return {
        data["task_id"]: (path.parent, data)
        for path in TASK_ROOT.rglob("SHAPES.json")
        for data in [json.loads(path.read_text())]
    }


def test_catalogs_preserve_all_source_case_inventories():
    found = catalogs()
    assert len(found) == 18
    for task, data in found.values():
        meta = json.loads((task / "ut" / "meta.json").read_text())
        for key, inventory in data["source_inventories"].items():
            expected = meta
            for part in key.split("/"):
                expected = expected[part]
            assert inventory["records"] == expected
        source = next(x for x in data["sources"] if x["path"] == "ut/meta.json")
        assert source["sha256"] == hashlib.sha256(
            (task / source["path"]).read_bytes()
        ).hexdigest()


def test_benchmark_ids_are_unique_and_resolve_to_shape_records():
    for task, data in catalogs().values():
        records = {row["case_id"]: row for row in data["cases"]}
        assert len(records) == len(data["cases"])
        ids = [row["case_id"] for row in data["benchmark_cases"]]
        assert len(ids) == len(set(ids))
        assert data["case_inventory"]["benchmark_case_ids"] == ids
        assert data["case_inventory"]["benchmark_case_count"] == len(ids)
        assert data["case_inventory"]["shape_record_count"] == len(records)
        for row in data["benchmark_cases"]:
            record = records[row["shape_record_id"]]
            assert row["case_id"] in record["benchmark_case_ids"]
        assert (task / "SHAPES.md").is_file()


def test_evidence_paths_move_with_tasks_and_unknowns_remain_explicit():
    for task, data in catalogs().values():
        assert data["path_base"] == "task directory"
        assert data["task_path"] == "."
        for source in data["sources"]:
            assert not Path(source["path"]).is_absolute()
            assert (task / source["path"]).is_file()
            assert len(source["sha256"]) == 64
        actual_unknowns = []
        for record in data["cases"]:
            for tensor in record["tensors"] + record["outputs"]:
                unknown = [
                    key for key in ("shape", "dtype", "strides")
                    if tensor[key] is None
                ]
                if unknown:
                    actual_unknowns.append({
                        "case_id": record["case_id"],
                        "tensor": tensor["name"],
                        "unknown_fields": unknown,
                        "evidence": tensor["evidence"],
                    })
        assert data["unknown_tensor_metadata"] == actual_unknowns
        for artifact in data["opaque_artifacts"]:
            assert len(artifact["sha256"]) == 64
            if artifact["inspected"]:
                evidence = next(x for x in data["fixture_metadata"] if x["artifact_path"] == artifact["path"])
                assert evidence["sha256"] == artifact["sha256"]
                assert evidence["metadata_only"] is True
                assert evidence["tensor_values_read"] is False
                assert evidence["parser"]["weights_only"] is True
                assert evidence["parser"]["map_location"] == "cpu"
                assert evidence["parser"]["mmap"] is True


def test_full_glm_correctness_coverage_and_observed_scored_cases():
    found = catalogs()
    bf16 = found["glm-5.3-flash__gemm_a16w16_bf16_cijk"][1]
    fp8 = found["glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle"][1]
    assert bf16["case_inventory"]["correctness_case_count"] == 27
    assert bf16["case_inventory"]["benchmark_case_count"] == 7
    assert len(bf16["unscored_correctness_cases"]) == 20
    assert fp8["case_inventory"]["correctness_case_count"] == 21
    assert fp8["case_inventory"]["benchmark_case_count"] == 7
    assert len(fp8["unscored_correctness_cases"]) == 14
    for record in fp8["cases"]:
        operands = {x["name"]: x for x in record["tensors"]}
        m, k = record["metadata"]["m"], record["metadata"]["k"]
        assert operands["x_scale"]["shape"] == [m, k // 128]
        assert operands["x_scale"]["strides"] == (
            [1, m] if m > 1 else [k // 128, 1]
        )
        assert operands["WQ"]["attributes"]["is_shuffled"] is True


def test_attention_capture_geometry_and_timing_geometry_are_distinct():
    found = catalogs()
    kimi = found["kimi-k3__fwd_grouped_kernel_stage1"][1]
    assert kimi["case_inventory"]["diagnostic_case_ids"] == [
        "decode_bs1_ctx8704", "decode_bs64_ctx8704",
    ]
    by_id = {x["case_id"]: x for x in kimi["cases"]}
    for case_id in kimi["case_inventory"]["diagnostic_case_ids"]:
        record = by_id[case_id]
        operands = {x["name"]: x for x in record["tensors"]}
        assert operands["k_buffer"]["shape"][0] == (
            record["metadata"]["total_kv"] + 64
        )
        assert operands["v_buffer"]["strides"] == [576, 576, 1]
    dsa = found["deepseek-v4-pro__dsa_sparse_mla_attn"][1]
    assert dsa["case_inventory"]["shape_record_count"] == 6
    assert dsa["case_inventory"]["benchmark_case_count"] == 4
    for record in dsa["cases"]:
        operands = {x["name"]: x for x in record["tensors"]}
        width = 64 if record["metadata"]["family_topk"] == 1024 else 2
        assert operands["extra_k_cache"]["shape"] == [40128, width, 1, 584]


def test_hash_verified_fixtures_close_physical_layout_unknowns():
    found = catalogs()
    assert sum(len(data["cases"]) for _, data in found.values()) == 121
    assert sum(data["case_inventory"]["benchmark_case_count"] for _, data in found.values()) == 49
    for _, data in found.values():
        assert data["unknown_tensor_metadata"] == []
        assert data["unknown_benchmark_tensor_overrides"] == []
    copy = found["glm-5.3-flash__elementwise_copy_cluster"][1]
    captures = [r for r in copy["cases"] if r["metadata"].get("source") == "hash_verified_capture"]
    assert len(captures) == 7
    assert all(r["benchmark_case_ids"] == [] for r in captures)
    dsa = found["deepseek-v4-pro__dsa_sparse_mla_attn"][1]
    for record in dsa["cases"]:
        cache = next(t for t in record["tensors"] if t["name"] == "k_cache")
        assert cache["strides"] == [149760, 584, 584, 1]


def test_kimi_disabled_scores_preserve_all_diagnostics_and_source_hashes():
    expected = {
        "fwd_grouped_kernel_stage1": (4, ["decode_bs1_ctx8704", "decode_bs64_ctx8704"]),
        "moe_gemm1_stage1": (3, ["prefill_M8192", "decode_M1", "decode_M64"]),
        "moe_gemm2_stage2": (4, ["prefill_M16384", "prefill_M8192", "decode_M1", "decode_M64"]),
    }
    found = catalogs()
    for name, (shape_count, ids) in expected.items():
        task, data = found["kimi-k3__" + name]
        meta = json.loads((task / "ut/meta.json").read_text())
        assert data["workload_scoring"] == meta["workload_scoring"]
        assert data["workload_scoring"]["enabled"] is False
        assert data["benchmark_cases"] == []
        assert data["case_inventory"]["benchmark_case_count"] == 0
        assert data["case_inventory"]["shape_record_count"] == shape_count
        assert data["case_inventory"]["diagnostic_case_ids"] == ids
        assert data["case_inventory"]["diagnostic_case_count"] == len(ids)
        diagnostics = data["unscored_diagnostic_cases"]
        assert [row["case_id"] for row in diagnostics] == ids
        records = {row["case_id"]: row for row in data["cases"]}
        assert all(row["benchmark_case_ids"] == [] for row in records.values())
        for row in diagnostics:
            assert row["scoring_enabled"] is False
            assert row["scoring_reason"] == meta["workload_scoring"]["reason"]
            assert row["case_id"] in records[row["shape_record_id"]]["diagnostic_case_ids"]
        for source in data["sources"]:
            assert source["sha256"] == hashlib.sha256((task / source["path"]).read_bytes()).hexdigest()


def test_kimi_historical_prefill_chunks_are_distinct_from_shared_isl():
    found = catalogs()
    expected_chunks = {
        "fwd_grouped_kernel_stage1": (None, 16384),
        "moe_gemm1_stage1": (8192, 8192),
        "moe_gemm2_stage2": (16384, 16384),
    }
    for name, (regime_chunk, weight_chunk) in expected_chunks.items():
        task, data = found["kimi-k3__" + name]
        workload = json.loads((task / "ut/workload.json").read_text())
        group = json.loads((task.parent / "workload.json").read_text())
        scenario = data["serving_capture_context"]["historical_scenario"]
        assert group["serving_capture"]["input_sequence_length"] == 8192
        assert scenario["regime_prefill_chunk"] == workload["regime"]["prefill_chunk"] == regime_chunk
        assert scenario["analytic_weight_prefill_chunk"] == workload["serving_weight_model"]["prefill_chunk"] == weight_chunk
        assert group["historical_scenarios"][name]["regime_prefill_chunk"] == regime_chunk
        kernel = next(row for row in group["kernels"] if row["kernel_slug"] == name)
        assert kernel["workload_scoring"]["enabled"] is False
        assert kernel["workload_scoring"]["benchmark_case_count"] == 0
