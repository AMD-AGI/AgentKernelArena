"""Independent known answers and comparator controls for task validation only.

These small fixtures are validation evidence, never additional scored cases.
They do not call the production baseline, initializer or candidate. Expected
answers are literal arithmetic/format facts, not outputs of task_reference.
"""
from __future__ import annotations

import math

import torch


def _exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def gemm_known_answer(reference, device):
    # Integer arithmetic is exact in both FP32 accumulation and BF16 output.
    a = torch.tensor([[1, 2, -1], [0, -2, 3]], dtype=torch.bfloat16, device=device)
    b = torch.tensor([[2, 0, 1], [-1, 3, 2], [1, -1, -1], [0, 2, 0]],
                     dtype=torch.bfloat16, device=device)
    expected = torch.tensor([[1, 3, 0, 4], [3, 0, -1, -4]],
                            dtype=torch.bfloat16, device=device)
    got = reference.run(a=a, b=b)
    _exact(got, expected)
    return {"oracle": "literal integer dot products A @ B.T; asymmetric 2x3 and 4x3 inputs",
            "expected": expected.cpu().tolist(), "observed": got.cpu().tolist()}


def moe_format_answers(reference, device):
    # E2M1 is [0, .5, 1, 1.5, 2, 3, 4, 6] plus its sign bit;
    # each byte contains low-nibble then high-nibble values.
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]],
                          dtype=torch.uint8, device=device)
    decoded = reference._mxfp4_to_f32(packed)
    expected = torch.tensor([[0, .5, 1, 1.5, 2, 3, 4, 6,
                              -0., -.5, -1, -1.5, -2, -3, -4, -6]],
                            dtype=torch.float32, device=device)
    _exact(decoded, expected)
    _exact(torch.signbit(decoded), torch.signbit(expected))
    scales = torch.tensor([125, 126, 127, 128, 129], dtype=torch.uint8, device=device)
    _exact(reference._e8m0_to_f32(scales),
           torch.tensor([.25, .5, 1., 2., 4.], dtype=torch.float32, device=device))

    # At maximum 6 the scale is 2**0 (biased exponent 127). These are midpoint
    # ties between adjacent representable values: the even code must win.
    ties = [.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.,
            -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5., -6.] * 2
    got_codes, got_scales = reference._quantize_mxfp4(
        torch.tensor([ties], dtype=torch.float32, device=device))
    expected_codes = torch.tensor([[0x20, 0x42, 0x64, 0x76, 0xA8, 0xCA, 0xEC, 0xFE] * 2],
                                  dtype=torch.uint8, device=device)
    _exact(got_codes, expected_codes)
    _exact(got_scales, torch.tensor([[127]], dtype=torch.uint8, device=device))
    return {"oracle": "literal E2M1 codebook, biased E8M0 powers, and nearest-even midpoint codes",
            "decoded": decoded.cpu().tolist(), "midpoint_packed_bytes": got_codes.cpu().tolist(),
            "midpoint_scale_byte": got_scales.item()}


def moe_gate_answer(reference, device):
    # Different gate and up halves expose swapped halves and a missing product.
    values = torch.tensor([[1., 2., 3., 4.], [-1., -2., 5., 6.]], dtype=torch.float32, device=device)
    expected = torch.tensor([[3 / (1 + math.exp(-1)), 8 / (1 + math.exp(-2))],
                             [-5 / (1 + math.exp(1)), -12 / (1 + math.exp(2))]],
                            dtype=torch.float32, device=device)
    got = reference._apply_gated_activation(values, activation=0)
    # Scalar Python/FP64 exp is independent of the callback's PyTorch SiLU.
    # This bound covers FP32 rounding in this fixture, not candidate tolerance.
    torch.testing.assert_close(got, expected, rtol=2e-6, atol=1e-6)
    return {"oracle": "gate * up / (1 + exp(-gate)) using scalar Python math",
            "expected": expected.cpu().tolist(), "observed": got.cpu().tolist()}


def moe_weight_layout_answer(reference, device):
    # Independent scalar indexing for the documented 16x32-byte tile packing;
    # nonconstant bytes make a wrong/no-op unshuffle observable.
    rows, cols = 32, 64
    physical = [0] * (rows * cols)
    logical = [[(row * cols + col) % 251 for col in range(cols)] for row in range(rows)]
    for row in range(rows):
        for col in range(cols):
            position = ((((row // 16) * (cols // 32) + col // 32) * 2
                         + (col % 32) // 16) * 16 + row % 16) * 16 + col % 16
            physical[position] = logical[row][col]
    got = reference._unshuffle_weight(
        torch.tensor(physical, dtype=torch.uint8, device=device).reshape(1, rows, cols))
    _exact(got, torch.tensor([logical], dtype=torch.uint8, device=device))
    return {"oracle": "scalar 16x32-byte tile address calculation", "checked_bytes": rows * cols}


def moe_known_answer(reference, device):
    # D=I=256 satisfies the reference's packed weight/scale layout constraints.
    # Per-expert constants are invariant under shuffling, avoiding reliance on
    # either callback's layout/quantizer to manufacture this fixture.
    d = i = 256
    experts = 3
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is None:
        raise RuntimeError("Native FP4 dtype required for the quantized MoE known answer")
    hidden = torch.zeros((2, d), dtype=torch.bfloat16, device=device)
    hidden[:, 0] = torch.tensor([1., 2.], dtype=torch.bfloat16, device=device)
    w1 = torch.empty((experts, 2 * i, d // 2), dtype=torch.uint8, device=device)
    w2 = torch.empty((experts, d, i // 2), dtype=torch.uint8, device=device)
    # Expert 0: W1=+1, W2=+1/4. Expert 1: W1=-1/2, W2=-1/2.
    # Expert 2: deliberately large weights; it must not contribute (not routed).
    for expert, byte in enumerate([0x22, 0x99, 0x77]):
        w1[expert].fill_(byte)
    for expert, byte in enumerate([0x22, 0xAA, 0x77]):
        w2[expert].fill_(byte)
    s1 = torch.full((experts, 2 * i, d // 32), 127, dtype=torch.uint8, device=device)
    s2 = torch.empty((experts, d, i // 32), dtype=torch.uint8, device=device)
    for expert, scale in enumerate([125, 126, 127]):
        s2[expert].fill_(scale)
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    weights = torch.tensor([[.25, .75], [.75, .25]], dtype=torch.float32, device=device)
    # Scalar derivation, after BF16 intermediate rounding + block FP4 rounding:
    # token 1: SiLU(1)*1 -> .75; SiLU(-.5)*(-.5) -> .09375.
    #          Expert outputs 256*.75/4=48 and -256*.09375/2=-12.
    #          .25*48 + .75*(-12) = 3.
    # token 2: SiLU(2)*2 -> 4; SiLU(-1)*(-1) -> .25.
    #          Expert outputs 256 and -32. .75*(-32) + .25*256 = 40.
    expected = torch.tensor([[3.] * d, [40.] * d], dtype=torch.bfloat16, device=device)
    got = reference.run(hidden_states=hidden, w1=w1.view(fp4), w2=w2.view(fp4),
                        topk_weights=weights, topk_ids=ids, activation=0,
                        doweight_stage1=False, w1_scale=s1, w2_scale=s2)
    _exact(got, expected)
    return {"oracle": "hand-computed quantized three-expert fixture with two routed experts",
            "shape": list(got.shape), "expected_row_values": [3., 40.],
            "observed_row_min": got.float().amin(dim=1).cpu().tolist(),
            "observed_row_max": got.float().amax(dim=1).cpu().tolist(),
            "routing_ids": ids.cpu().tolist(), "unselected_expert": 2}


def comparator_controls(compare, family, device):
    expected = torch.ones((2, 4), dtype=torch.bfloat16, device=device)
    # Positive controls include nonzero error; this is not just a self-compare.
    accepted_value = 1.0078125 if family == "gemm" else 1.125
    rejected_value = 1.03125 if family == "gemm" else 1.5
    compare.run(torch.full_like(expected, accepted_value), expected)
    rejected = {
        "outside_numerical_gate": torch.full_like(expected, rejected_value),
        "wrong_sign": -expected,
        "wrong_shape": expected[:, :1],
        "wrong_dtype": expected.float(),
        "nonfinite_output": torch.full_like(expected, float("nan")),
    }
    observations = {}
    for name, wrong in rejected.items():
        try:
            compare.run(wrong, expected)
        except AssertionError as error:
            observations[name] = str(error)
        else:
            raise RuntimeError(f"Comparator accepted negative control: {name}")
    return {"oracle": ("BF16 0.01 either tolerance" if family == "gemm" else "output-dtype SQNR >= 13 dB"),
            "reference_value": 1., "accepted_inexact_value": accepted_value,
            "rejected_inexact_value": rejected_value, "rejected_controls": observations}


def run_controls(measure, *, device="cuda", records=None):
    records = [] if records is None else records
    family = measure.task_inputs.WORKLOAD["op_type"]
    checks = [("gemm_integer_matrix", gemm_known_answer)] if family == "gemm" else [
        ("moe_fp4_format_and_rounding", moe_format_answers),
        ("moe_gate_up_split", moe_gate_answer),
        ("moe_weight_layout", moe_weight_layout_answer),
        ("moe_quantized_routing", moe_known_answer),
    ]
    for name, check in checks + [("comparator_positive_and_negative", None)]:
        record = {"name": name, "device": str(device), "status": "FAIL"}
        records.append(record)
        try:
            record["evidence"] = (check(measure.task_reference, device) if check is not None
                                  else comparator_controls(measure.task_compare, family, device))
        except Exception as error:
            record["reason"] = f"{type(error).__name__}: {error}"
            raise RuntimeError(f"Task validation control failed: {name}: {error}") from error
        record["status"] = "PASS"
    return records
