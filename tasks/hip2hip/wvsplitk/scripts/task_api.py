"""Protected skinny-GEMM inputs, independent FP32 oracle and launch ABI."""
from __future__ import annotations

import torch


def validate_params(p):
    if (set(p) != {"tokens", "n", "k", "bias", "dtype"}
            or p["tokens"] not in (1, 2, 3, 4) or p["n"] <= 0
            or p["k"] <= 0 or p["k"] % 8 or type(p["bias"]) is not bool
            or p["dtype"] not in ("bfloat16", "float16")):
        raise ValueError(f"Unsupported skinny-GEMM parameters: {p}")


def make_inputs(p, seed=42, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = getattr(torch, p["dtype"])
    values = {
        "weight": torch.randn((p["n"], p["k"]), generator=generator, device=device, dtype=dtype),
        "activation": torch.randn((p["tokens"], p["k"]), generator=generator, device=device, dtype=dtype),
    }
    if p["bias"]:
        values["bias"] = torch.randn(p["n"], generator=generator, device=device, dtype=dtype)
    return values


def readonly(values):
    return values


def draw(values, p, seed):
    activation = values["activation"]
    generator = torch.Generator(device=activation.device).manual_seed(seed)
    return {"activation": torch.randn(activation.shape, generator=generator,
                                       device=activation.device, dtype=activation.dtype)}


def reference(values, p):
    result = values["activation"].float() @ values["weight"].float().T
    if p["bias"]:
        result = result + values["bias"].float()
    return result.to(getattr(torch, p["dtype"]))


def check_output_contract(got, expected):
    if (not isinstance(got, torch.Tensor) or got.shape != expected.shape
            or got.dtype != expected.dtype or got.device != expected.device):
        raise AssertionError("GEMM output shape, dtype or device mismatch")
    if not torch.isfinite(got).all() or not torch.isfinite(expected).all():
        raise AssertionError("Non-finite GEMM output or reference")


def compare(got, expected, p):
    check_output_contract(got, expected)
    # Preserve the original HIP task's FP16/BF16 numerical gate.
    if not torch.allclose(got.float(), expected.float(), atol=5e-2, rtol=5e-2):
        error = (got.float() - expected.float()).abs().max().item()
        raise AssertionError(f"Skinny-GEMM differs from FP32 oracle: max_abs={error}")


def extra_negative_checks(expected, p):
    pass


def poison_output(output, p):
    output.fill_(float("nan"))


def timing_options():
    from build_kernel import timing_options as options
    return options()


def load_candidate():
    from build_kernel import load_extension
    op = load_extension("extracted_wvsplitk").wvSplitK
    cu_count = torch.cuda.get_device_properties(0).multi_processor_count

    def invoke(values, p):
        return op(values["weight"], values["activation"], values.get("bias"), cu_count)

    return invoke
