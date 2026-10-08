"""Preserve the original prefill state sampler and bind its measured receipt."""
from collections import Counter
import math
from evaluation_contract import fingerprint, require, canonical


def state_sampling(case, seed, policy, indices, variants, samples_ms):
    require(type(seed) is int and seed >= 0, "private prefill challenge is missing")
    warm, count = policy["warmup_iterations"], policy["benchmark_iterations"]
    require(warm == 10 and count == 100, "prefill sample policy changed")
    states = case["states"]
    expected_indices = [(seed+i) % len(states) for i in range(warm+count)]
    expected_variants = [fingerprint(states[index]["tensor_controls"]) for index in expected_indices]
    require(indices == expected_indices and variants == expected_variants, "prefill original state schedule changed")
    require(len(samples_ms) == count and all(type(t) in (int,float) and math.isfinite(t) and t > 0 for t in samples_ms),
            "prefill raw timing samples are incomplete")
    return {"schema":"minimax-recorded-state-sampling-v1", "sampling":"original_seed_modulo_recorded_state_count",
        "private_challenge_seed":seed,"recorded_state_count":len(states),
        "warmup_state_indices":indices[:warm],"measured_state_indices":indices[warm:],
        "warmup_variant_ids":variants[:warm],"measured_variant_ids":variants[warm:],
        "schedule_fingerprint":fingerprint({"case_id":case["case_id"],"seed":seed,"state_indices":indices,"variant_ids":variants}),
        "mean_gpu_cost_ms":math.fsum(samples_ms)/count,
        "aggregation":"arithmetic_mean_from_all_100_raw_device_samples",
        "timing_guarantee":"original_recorded_state_sampled_inputs"}


def validate_state_sampling(case, request, policy, row, signatures):
    receipt = row.get("workload_state_sampling",{})
    indices = receipt.get("warmup_state_indices",[]) + receipt.get("measured_state_indices",[])
    variants = receipt.get("warmup_variant_ids",[]) + receipt.get("measured_variant_ids",[])
    expected = state_sampling(case,request["challenge_seed"],policy,indices,variants,row["samples_ms"])
    require(canonical(receipt) == canonical(expected), "prefill state receipt differs from protected schedule/raw samples")
    require(len(signatures) == 110 and all(s["input_seed"] == request["challenge_seed"]+i
            and s["variant_id"] == s["actual_tensor_controls_sha256"] == variants[i]
            for i,s in enumerate(signatures)), "prefill realized controls contradict the protected states")
