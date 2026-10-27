# SPDX-License-Identifier: MIT
"""Protected, CPU-readable configuration and fixed batching derivation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_contract(root=ROOT):
    import yaml

    root = Path(root)
    data = json.loads((root / "workloads.json").read_text())
    identity = json.loads((root / "source_identity.json").read_text())
    model_bytes = (root / "model_config.json").read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != identity["model_config_sha256"]:
        raise ValueError("Checkpoint configuration content differs from the source binding")
    model = json.loads(model_bytes)
    text = model["text_config"]
    workload = data["workload"]
    if (workload["model"] != "amd/MiniMax-M3-MXFP4"
            or workload["model_revision"] != "b83d14e3d64bf373a207f3c2a7e9f0b0f1e7fc3a"
            or workload["model_config_sha256"] != identity["model_config_sha256"]):
        raise ValueError("Recorded checkpoint identity differs from the task binding")
    if workload["tp"] != 8 or workload["concurrency"] != 64 or workload["num_prompts"] != 64:
        raise ValueError("Task requires the declared TP8/concurrency64 request configuration")
    if (workload["input_tokens"], workload["output_tokens"]) != (8192, 1024):
        raise ValueError("Task requires input8192/output1024")
    site = "language_model.model.layers.0.mlp.down_proj"
    if site not in model["quantization_config"]["exclude"]:
        raise ValueError("The selected dense projection is not exempt from Quark quantization")
    if text["moe_layer_freq"][0] != 0 or model["torch_dtype"] != "bfloat16":
        raise ValueError("The selected site must be a BF16 dense MLP")
    n = text["hidden_size"]
    full_k = text["dense_intermediate_size"]
    if full_k % workload["tp"]:
        raise ValueError("Dense intermediate dimension must divide by TP")
    k = full_k // workload["tp"]
    if (n, k) != (6144, 1536):
        raise ValueError("Checkpoint-derived rank-local dimensions changed")
    if data["operation"] != {"site": site, "n": n, "k": k, "bias": False,
                              "output_dtype": "bfloat16", "includes_all_reduce": False}:
        raise ValueError("Operation differs from the selected rank-local projection")
    expected = [("prefill_m8192", 8192), ("decode_m64", 64)]
    cases = data["cases"]
    if len(cases) != len(expected):
        raise ValueError("The complete fixed-scenario case set is required")
    for case, (case_id, m) in zip(cases, expected):
        if (case["test_case_id"] != case_id or case["shape"] != [m, n, k]
                or case["dtype"] != "bfloat16"
                or case["checks"] != ["correctness", "performance"]
                or case["params"] != {"m": m, "n": n, "k": k, "bias": False,
                                       "transpose_weight": True, "benchmark_method": "cuda_graph"}):
            raise ValueError("Case identity, geometry, or measurement policy changed")
    config = yaml.safe_load((root / "config.yaml").read_text())
    aggregate = config["evaluation"]["aggregation"]
    source = aggregate["source"]
    if source["path"] != "frequency_provenance.json":
        raise ValueError("Unexpected frequency provenance path")
    raw = (root / source["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Fixed batching policy hash changed")
    policy = json.loads(raw)
    weights = {"prefill_m8192": workload["num_prompts"],
               "decode_m64": workload["output_tokens"]}
    if (aggregate["method"] != "weighted_total_time"
            or aggregate["case_weights"] != weights or policy["case_weights"] != weights
            or policy["workload"] != workload
            or policy["kind"] != "config_derived_fixed_batching_scenario"
            or policy["observed_scheduler_frequencies"] is not False):
        raise ValueError("Scoring must preserve the explicitly config-derived batching scenario")
    generation = data["input_generation"]
    if generation != {"distribution": "upstream_normal_bf16", "seeds": [42, 1042],
                       "checkpoint_weights": False, "prepared_tensor_values_may_be_cached": False}:
        raise ValueError("Protected upstream-style input policy changed")
    if data["tolerance"] != {"rtol": 0.01, "atol": 0.01}:
        raise ValueError("Protected BF16 numerical tolerance changed")
    if data["timing"] != {"warmup": 10, "repetition": 100, "target_ms": 1.0,
                          "max_graph_repeats": 1000}:
        raise ValueError("Protected timing policy changed")
    return data
