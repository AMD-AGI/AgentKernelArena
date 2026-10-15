"""Unscored checks for the scaled-GEMM public optional arguments."""
import copy

API_CASES = [
    {"name": "api_fp16_matrix_scales", "M": 3, "N": 5, "K": 2,
     "output_dtype": "float16", "vector_scales": False, "with_bias": False,
     "preallocated_output": False, "strided_output": False, "config_override": False},
    {"name": "api_fp16_vector_preallocated", "M": 17, "N": 33, "K": 65,
     "output_dtype": "float16", "vector_scales": True, "with_bias": True,
     "preallocated_output": True, "strided_output": False, "config_override": True},
    {"name": "api_bf16_vector_preallocated", "M": 7, "N": 19, "K": 33,
     "output_dtype": "bfloat16", "vector_scales": True, "with_bias": False,
     "preallocated_output": True, "strided_output": False, "config_override": True},
    {"name": "api_bf16_strided_output", "M": 3, "N": 5, "K": 65,
     "output_dtype": "bfloat16", "vector_scales": False, "with_bias": True,
     "preallocated_output": True, "strided_output": True, "config_override": False},
]


def run_api_controls(module, make_inputs, reference, checked_output, in_dtype, *, device="cuda", verbose=True):
    import torch
    from .replay_checks import require_unchanged

    failures = []
    for case in API_CASES:
        name = case["name"]
        try:
            m, n, k = (case[key] for key in ("M", "N", "K"))
            x, w, xs, ws, bias = make_inputs(m, n, k, in_dtype, case["with_bias"], device=device)
            out_dtype = getattr(torch, case["output_dtype"])
            # The independent oracle retains explicit row/column broadcast.
            expected = reference(x, w, xs.reshape(m, 1), ws.reshape(1, n), bias, out_dtype)
            if case["vector_scales"]:
                xs, ws = xs.reshape(m), ws.reshape(n)
            if bias is not None:
                bias = bias.reshape(n)
            inputs = tuple(value for value in (x, w, xs, ws, bias) if value is not None)
            originals = tuple(value.clone() for value in inputs)
            destination = None
            storage = None
            if case["preallocated_output"]:
                if case["strided_output"]:
                    storage = torch.full((m, n * 2), float("nan"), dtype=out_dtype, device=device)
                    destination = storage[:, ::2]
                else:
                    destination = torch.full((m, n), float("nan"), dtype=out_dtype, device=device)
            config = None
            if case["config_override"]:
                config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32,
                          "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 128,
                          "cache_modifier": "", "num_warps": 4, "num_stages": 2, "waves_per_eu": 0}
            original_config = copy.deepcopy(config)
            actual = module.gemm_a8w8(x, w, xs, ws, bias, out_dtype, y=destination, config=config)
            require_unchanged(inputs, originals)
            if config != original_config:
                raise AssertionError("Scaled GEMM changed caller-owned configuration")
            checked_output(actual, x, w, out_dtype)
            if destination is not None and actual is not destination:
                raise AssertionError("Scaled GEMM must return the supplied output tensor")
            if storage is not None and not bool(torch.isnan(storage[:, 1::2]).all().item()):
                raise AssertionError("Scaled GEMM wrote outside strided output elements")
            if not bool(torch.isfinite(actual).all().item()) or not torch.allclose(actual, expected, atol=0.02, rtol=0.01):
                raise AssertionError("Scaled GEMM optional-argument output differs from FP32 reference")
            if verbose:
                print(f"  PASS: {name}")
        except Exception as error:
            failures.append(name)
            if verbose:
                print(f"  FAIL: {name} - {type(error).__name__}: {error}")
    return failures
