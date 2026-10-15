"""Independent CPU known answers for protected reference functions.

These additional small controls do not replace or resize any GPU workload or
change its numerical gate. Full reference/candidate evaluation remains in the
original harness. A deliberately wrong output must fail each control comparator.
"""
import math
import torch
import torch.nn.functional as F
from reference_support import references, control, close


def run():
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    rows = []
    def check(actual, expected, label):
        rows.append(control(actual, expected, close(1e-5, 1e-5), label))
    r = references(['_torch_reference'], {"F": F, "SQRT2": math.sqrt(2)})
    x=t([[3.,-3.,4.,-4.]])
    check(r._torch_reference(x,2.,t([[2.,3.]])),t([[8/(1+math.exp(-2)),18/(1+math.exp(3))]]),"one-sided gate clamp, two-sided up clamp, weights")
    from semantic_controls import reference
    gelu_input = t([[-1., 1., 2., -2.]])
    for activation in ("gelu", "gelu_tanh"):
        def phi(value):
            if activation == "gelu":
                return .5 * value * (1. + math.erf(value / math.sqrt(2.)))
            return .5 * value * (1. + math.tanh(math.sqrt(2. / math.pi) * (value + .044715 * value ** 3)))
        expected = t([[2. * phi(-1.), -2. * phi(1.)]])
        check(reference(gelu_input, 0., None, activation), expected,
              "independent scalar " + activation + " known answer")
    return rows
