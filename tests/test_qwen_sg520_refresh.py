"""CPU checks for the audited Qwen image refresh and preserved capture ABI."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
import yaml

TASKS = Path(__file__).resolve().parents[1] / "tasks/headkernel"
IMAGE = "docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96"
COUNTS = {
    "dense_bf16_gemm_cluster": 5,
    "fused_moe_2stage_mxfp4": 2,
    "fused_recurrent_gated_delta_rule_decode": 2,
    "gemma_fused_add_rmsnorm": 2,
    "paged_attention_decode": 1,
}


@pytest.mark.parametrize("name,count", COUNTS.items())
def test_native_source_and_existing_callable_contract(name, count):
    task = TASKS / ("qwen3.8-2.4t__" + name)
    cfg = yaml.safe_load((task / "config.yaml").read_text())
    meta = json.loads((task / "ut/meta.json").read_text())
    refresh = json.loads((task / "ut/runtime_refresh.json").read_text())
    source = task / cfg["source_file_path"][0]
    native = task / "ut" / meta["source_provenance"]["baseline_ref"]
    assert hashlib.sha256(native.read_bytes()).hexdigest() == refresh["native_source_sha256"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == refresh["candidate_sha256"]
    assert cfg["headkernel"]["docker"] == meta["source_provenance"]["runtime_image"] == IMAGE
    assert meta["num_cases"] == refresh["case_count"] == count
    assert refresh["current_runtime_GPU_validation"] == "PENDING_ROOT_ASSIGNED_GPU_REPLAY"
    assert not refresh["current_runtime_workload_capture_claimed"]
    assert meta["capture_source_provenance"]["runtime_image"] != IMAGE
    candidate_functions = {n.name: n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)}
    native_functions = {n.name: n for n in ast.parse(native.read_text()).body if isinstance(n, ast.FunctionDef)}
    for symbol in cfg["target_kernel_functions"]:
        assert ast.dump(candidate_functions[symbol]) == ast.dump(native_functions[symbol])


def test_captured_qwen_shapes_are_not_replaced_by_model_size_guesses():
    def meta(name):
        return json.loads((TASKS / ("qwen3.8-2.4t__" + name) / "ut/meta.json").read_text())
    moe = meta("fused_moe_2stage_mxfp4")
    assert {r["m"] for r in moe["workload"]["cases"]} == {64, 8192}
    assert moe["geometry"]["local_experts"] == 64 and moe["geometry"]["topk"] == 10
    attn = meta("paged_attention_decode")
    assert attn["geometry"]["query_shape"] == [64, 8, 256]
    assert attn["geometry"]["distinct_page_count"] == 521351
    recurrent = meta("fused_recurrent_gated_delta_rule_decode")
    assert {r["B"] for r in recurrent["workload"]["cases"]} == {1, 64}
    assert recurrent["geometry"]["H"] == 2 and recurrent["geometry"]["HV"] == 16
