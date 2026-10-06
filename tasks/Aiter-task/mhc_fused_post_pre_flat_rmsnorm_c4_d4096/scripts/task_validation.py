"""Independent known answers and comparator controls for task validation only.

These small fixtures are validation evidence, never additional scored cases.
They do not call the production baseline, initializer or candidate. Expected
answers come from scalar float64 Python arithmetic written from the operator's
definition, not from outputs of task_reference.
"""
from __future__ import annotations

import math

import torch

TOKENS, STREAMS, HIDDEN = 2, 4, 4
MIXES = 2 * STREAMS + STREAMS * STREAMS


def _fixture():
    """Small exactly representable operands with asymmetric structure.

    Token 0 combines streams through a permutation, token 1 through an average
    of two permutations, so a missing or extra transpose of the combination
    matrix changes the new residual. Projection weights, mix scales and biases
    differ per mix, so a wrong scale group, bias offset or reshape changes the
    derived mixes. All residual-path values are multiples of 1/8 below 16, which
    BF16 represents exactly.
    """
    residual = [[[((t * 7 + i * 3 + h * 5) % 7) - 3 for h in range(HIDDEN)]
                 for i in range(STREAMS)] for t in range(TOKENS)]
    x = [[((t * 3 + h * 2) % 5) - 2 for h in range(HIDDEN)] for t in range(TOKENS)]
    gates = [0.25, 0.5, 0.75, 1.0]
    post = [[gates[(i + t) % STREAMS] for i in range(STREAMS)] for t in range(TOKENS)]

    def permutation(target):
        return [[1.0 if target[i] == j else 0.0 for j in range(STREAMS)] for i in range(STREAMS)]

    first, second = permutation([1, 2, 3, 0]), permutation([2, 0, 3, 1])
    comb = [first, [[0.5 * (a + b) for a, b in zip(ra, rb)] for ra, rb in zip(first, second)]]
    weight = [[(((k * 5 + f * 3) % 9) - 4) / 16 for f in range(STREAMS * HIDDEN)]
              for k in range(MIXES)]
    scale = [1.0, 0.5, 0.25]
    bias = [(((k * 3) % 7) - 3) / 8 for k in range(MIXES)]
    norm = [0.5, 0.75, 1.25, 1.5]
    return residual, x, post, comb, weight, scale, bias, norm


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def _oracle(residual, x, post, comb, weight, scale, bias, norm, scalars):
    """Scalar float64 evaluation of post followed by the next pre with RMSNorm."""
    outputs = {"next_post_mix": [], "next_comb_mix": [], "layer_input": [], "next_residual": []}
    for t in range(TOKENS):
        # Post: stream j gathers sum_i comb[i][j] * residual[i] plus the gated layer output.
        new = [[sum(comb[t][i][j] * residual[t][i][h] for i in range(STREAMS)) + x[t][h] * post[t][j]
                for h in range(HIDDEN)] for j in range(STREAMS)]
        flat = [value for stream in new for value in stream]
        inverse_rms = 1.0 / math.sqrt(sum(v * v for v in flat) / len(flat) + scalars["rms_eps"])
        mixes = [sum(w * v for w, v in zip(weight[k], flat)) * inverse_rms for k in range(MIXES)]
        pre = [_sigmoid(mixes[i] * scale[0] + bias[i]) + scalars["pre_eps"] for i in range(STREAMS)]
        gate = [_sigmoid(mixes[STREAMS + i] * scale[1] + bias[STREAMS + i]) * scalars["post_multiplier"]
                for i in range(STREAMS)]
        logits = [[mixes[2 * STREAMS + STREAMS * i + j] * scale[2] + bias[2 * STREAMS + STREAMS * i + j]
                   for j in range(STREAMS)] for i in range(STREAMS)]
        matrix = []
        for row in logits:
            peak = max(row)
            total = sum(math.exp(v - peak) for v in row)
            matrix.append([math.exp(v - peak) / total + scalars["sinkhorn_eps"] for v in row])

        def normalize_columns(m):
            sums = [sum(m[i][j] for i in range(STREAMS)) + scalars["sinkhorn_eps"] for j in range(STREAMS)]
            return [[m[i][j] / sums[j] for j in range(STREAMS)] for i in range(STREAMS)]

        def normalize_rows(m):
            return [[v / (sum(row) + scalars["sinkhorn_eps"]) for v in row] for row in m]

        matrix = normalize_columns(matrix)
        for _ in range(scalars["sinkhorn_iters"] - 1):
            matrix = normalize_columns(normalize_rows(matrix))
        layer = [sum(new[i][h] * pre[i] for i in range(STREAMS)) for h in range(HIDDEN)]
        inverse_norm = 1.0 / math.sqrt(sum(v * v for v in layer) / HIDDEN + scalars["norm_eps"])
        outputs["next_post_mix"].append([[g] for g in gate])
        outputs["next_comb_mix"].append(matrix)
        outputs["layer_input"].append([v * inverse_norm * n for v, n in zip(layer, norm)])
        outputs["next_residual"].append(new)
    return outputs


def mhc_known_answer(reference, inputs_module, device):
    residual, x, post, comb, weight, scale, bias, norm = _fixture()
    scalars = inputs_module.SCALARS
    expected = _oracle(residual, x, post, comb, weight, scale, bias, norm, scalars)
    tensor = lambda value, dtype: torch.tensor(value, dtype=dtype, device=device)
    operands = {
        "x": tensor(x, torch.bfloat16), "residual": tensor(residual, torch.bfloat16),
        "post_mix": tensor(post, torch.float32), "comb_mix": tensor(comb, torch.float32),
        "proj_weight": tensor(weight, torch.float32), "mix_scale": tensor(scale, torch.float32),
        "mix_bias": tensor(bias, torch.float32), "norm_weight": tensor(norm, torch.bfloat16),
        **scalars,
    }
    got = inputs_module.named_outputs(reference.run(**{name: operands[name] for name in inputs_module.INPUTS}))
    # The residual path is exact by construction. FP32 mixes may differ from the
    # float64 oracle in the last bits; the BF16 layer input may additionally
    # round to an adjacent BF16 value. These bounds cover this fixture's
    # arithmetic only and are unrelated to the candidate tolerance.
    torch.testing.assert_close(got["next_residual"], tensor(expected["next_residual"], torch.bfloat16),
                               rtol=0, atol=0)
    for name in ("next_post_mix", "next_comb_mix"):
        torch.testing.assert_close(got[name], tensor(expected[name], torch.float32), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got["layer_input"].float(), tensor(expected["layer_input"], torch.float64).float(),
                               rtol=2**-7, atol=1e-6)
    return {"oracle": ("scalar float64 post (comb^T residual + post * x), RMS-scaled projection, "
                       "sigmoid/softmax/Sinkhorn mixes and RMSNorm, written from the definition"),
            "tokens": TOKENS, "streams": STREAMS, "hidden_size": HIDDEN, "scalars": dict(scalars),
            "expected_next_post_mix": expected["next_post_mix"],
            "observed_next_post_mix": got["next_post_mix"].cpu().tolist(),
            "expected_layer_input": expected["layer_input"],
            "observed_layer_input": got["layer_input"].float().cpu().tolist()}


def comparator_controls(compare, inputs_module, device):
    shapes = {"next_post_mix": (TOKENS, STREAMS, 1), "next_comb_mix": (TOKENS, STREAMS, STREAMS),
              "layer_input": (TOKENS, HIDDEN), "next_residual": (TOKENS, STREAMS, HIDDEN)}
    dtypes = {name: inputs_module.DTYPES[spec["dtype"]] for name, spec in inputs_module.OUTPUTS.items()}
    if set(shapes) != set(dtypes):
        raise RuntimeError("Comparator controls do not cover the declared outputs")
    expected = {name: torch.ones(shapes[name], dtype=dtypes[name], device=device) for name in dtypes}
    filled = lambda value: {name: torch.full_like(t, value) for name, t in expected.items()}
    # Positive control with nonzero error in every output; not just a self-compare.
    accepted_value, rejected_value = 1.015625, 1.03125
    compare.run(filled(accepted_value), expected)
    rejected = {f"outside_numerical_gate:{name}": {**expected, name: torch.full_like(expected[name], rejected_value)}
                for name in expected}
    rejected.update({
        "wrong_sign": {name: -t for name, t in expected.items()},
        "wrong_shape": {**expected, "layer_input": expected["layer_input"][:, :1]},
        "wrong_dtype": {**expected, "layer_input": expected["layer_input"].float()},
        "nonfinite_output": {**expected, "next_comb_mix": torch.full_like(expected["next_comb_mix"], float("nan"))},
        "missing_output": {name: t for name, t in expected.items() if name != "next_residual"},
    })
    observations = {}
    for name, wrong in rejected.items():
        try:
            compare.run(wrong, expected)
        except AssertionError as error:
            observations[name] = str(error)
        else:
            raise RuntimeError(f"Comparator accepted negative control: {name}")
    return {"oracle": "additive atol=rtol=1e-2 on every named output",
            "reference_value": 1., "accepted_inexact_value": accepted_value,
            "rejected_inexact_value": rejected_value, "rejected_controls": observations}


def run_controls(measure, *, device="cuda", records=None):
    records = [] if records is None else records
    checks = [("mhc_fused_post_pre_fixture",
               lambda: mhc_known_answer(measure.task_reference, measure.task_inputs, device)),
              ("comparator_positive_and_negative",
               lambda: comparator_controls(measure.task_compare, measure.task_inputs, device))]
    for name, check in checks:
        record = {"name": name, "device": str(device), "status": "FAIL"}
        records.append(record)
        try:
            record["evidence"] = check()
        except Exception as error:
            record["reason"] = f"{type(error).__name__}: {error}"
            raise RuntimeError(f"Task validation control failed: {name}: {error}") from error
        record["status"] = "PASS"
    return records
