"""All upstream GLM GEMM geometries with deterministic synthetic operands.

The package deliberately has no captured tensor oracle. Independent FP32
references use the same BF16/FP8 operands, before the weight-only permutation.
"""
import json
from pathlib import Path

import harness_lib as h

META = json.loads((Path(__file__).resolve().parent / "meta.json").read_text())


def correctness_cases(meta):
    """Return every fixed case, including unscored robustness/generalization."""
    cases = meta["cases"]
    by_id = {case["sig"]: case for case in cases}
    expected_count = 27 if meta["kind"] == "bf16" else 21
    if (len(cases) != expected_count or len(by_id) != expected_count
            or meta["correctness_case_ids"] != [case["sig"] for case in cases]):
        raise RuntimeError("the complete fixed GLM correctness case set is required")
    return cases


def selected_cases(meta, case_ids):
    """The unchanged Arena adapter requests only the observed scored cases."""
    cases = correctness_cases(meta)
    observed = [case for case in cases if case["scenario_evidence"]["scored"]]
    expected_ids = [case["sig"] for case in observed]
    if (len(observed) != 7 or list(case_ids) != meta["ledger_ids"]
            or meta["ledger_ids"] != expected_ids
            or any(case["m"] != 64
                   or case["scenario_evidence"]["classification"] != "observed_profile_shape"
                   for case in observed)):
        raise RuntimeError("the complete fixed observed GLM benchmark set is required")
    return observed


def shuffle_weight(weight):
    """Exact gfx950 AITER shuffle_weight(weight, (16,16)) byte permutation."""
    import torch
    n, k = weight.shape
    raw = weight.view(torch.uint8)
    shuffled = (raw.view(n // 16, 16, k // 32, 2, 16)
                .permute(0, 2, 3, 1, 4).contiguous().view(n, k).view(weight.dtype))
    shuffled.is_shuffled = True
    return shuffled


def make_args(case, *, seed, device="cuda", with_reference=False):
    import torch
    m, n, k = case["m"], case["n"], case["k"]
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(m, k, generator=generator, dtype=torch.float32, device=device) * 0.1
    w = torch.randn(n, k, generator=generator, dtype=torch.float32, device=device) * 0.05
    if META["kind"] == "bf16":
        x, w = x.to(torch.bfloat16), w.to(torch.bfloat16)
        args = {"A": x, "B": w}
        ref = (x.float() @ w.float().t()).to(torch.bfloat16) if with_reference else None
    else:
        dtype = torch.float8_e4m3fn  # gfx950 production FP8 format
        maximum = float(torch.finfo(dtype).max)
        sk, sn = k // 128, (n + 127) // 128
        xb = x.reshape(m, sk, 128)
        xs = xb.abs().amax(dim=2).clamp_min(1e-8) / maximum
        xq = (xb / xs[:, :, None]).clamp(-maximum, maximum).reshape(m, k).to(dtype)
        padded = torch.zeros(sn * 128, k, device=device, dtype=torch.float32)
        padded[:n] = w
        wb = padded.reshape(sn, 128, sk, 128)
        ws = wb.abs().amax(dim=(1, 3)).clamp_min(1e-8) / maximum
        wq = (wb / ws[:, None, :, None]).clamp(-maximum, maximum)
        wq = wq.reshape(sn * 128, k)[:n].contiguous().to(dtype)
        ref = None
        if with_reference:
            xdeq = xq.float() * xs.repeat_interleave(128, dim=1)
            wdeq = wq.float() * ws.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:n]
            ref = (xdeq @ wdeq.t()).to(torch.bfloat16)
        # "x" identifies the activation to the shared timed-replay probe. The
        # candidate receives this tensor in the production XQ argument position.
        args = {"x": xq, "WQ": shuffle_weight(wq),
                "x_scale": xs.t().contiguous().t(), "w_scale": ws}
    return (args, ref) if with_reference else args


def _invoke(function, args):
    import torch
    if META["kind"] == "bf16":
        # The historically selected inner launcher takes solution index 0.
        return function(args["A"], args["B"], 0, bias=None, otype=torch.bfloat16,
                        scale_a=None, scale_b=None, scale_c=None, bpreshuffle=False,
                        config=None)
    return function(args["x"], args["WQ"], args["x_scale"], args["w_scale"],
                    dtype=torch.bfloat16)


def baseline_call(args):
    return _invoke(h.native_function(), args)


def candidate_call(args):
    # The binder also updates the native serving aliases. Call through those
    # aliases, so a dead rebind cannot pass simply by calling the local object.
    import importlib
    h.candidate_function()
    if META["kind"] == "bf16":
        function = importlib.import_module("aiter.tuned_gemm").solMap["torch"]
    else:
        function = importlib.import_module("aiter").gemm_a8w8_blockscale_bpreshuffle
    return _invoke(function, args)


def timing_case(case):
    if not case["scenario_evidence"]["scored"]:
        raise RuntimeError("unscored robustness/generalization case cannot enter timing")
    return {"sig": case["sig"], "regime": case["regime"], "m": case["m"],
            "args": make_args(case, seed=3000)}
