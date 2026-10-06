"""Keep the editable boundary on the three actual BF16 split/reduce captures."""

TARGETS = ("_paged_decode_split_kernel", "_paged_decode_reduce_kernel")
QUERY_SHAPE = (64, 16, 512)


def validate_dispatch(manifest):
    if manifest.get("seam") != "mla_decode":
        raise ValueError("Unsupported captured dispatch seam")
    for case in manifest["cases"]:
        tensors = case["tensors"]
        if (tuple(tensors["arg.q"]["shape"]) != QUERY_SHAPE
                or tensors["arg.q"]["dtype"] != "bfloat16"
                or tensors["arg.unified_kv"]["dtype"] != "bfloat16"
                or case["scalars"]["arguments"]["kv_scales"] is not None):
            raise ValueError("Case dispatches outside the captured BF16 T=64 H=16 decode specialization")
    return True


def validate_runtime_dispatch(module, inputs):
    if (tuple(inputs["q"].shape) != QUERY_SHAPE or inputs.get("kv_scales") is not None
            or not module._is_hip or module._is_gfx1250_supported):
        raise ValueError("Runtime inputs/platform differ from the captured decode specialization")
    T, H, D = QUERY_SHAPE
    block_h = max(16, 1 << (min(H, 64) - 1).bit_length())
    num_cu = module._cu_count()
    kv_splits = module._kv_splits_heuristic(T, H, block_h, num_cu=num_cu)
    if kv_splits != 4:
        raise ValueError("Runtime heuristic differs from the captured kv_splits=4 dispatch")
    return {"T": T, "H": H, "D": D, "block_h": block_h, "num_cu": num_cu,
            "kv_splits": kv_splits, "quant_kv": False, "selected_targets": list(TARGETS)}
