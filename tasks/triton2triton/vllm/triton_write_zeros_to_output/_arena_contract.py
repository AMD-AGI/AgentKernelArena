"""Protected CPU oracle and additional operator-domain controls."""

import torch

def i(values):
    return torch.tensor(values, dtype=torch.int32)

def l(values):
    return torch.tensor(values, dtype=torch.int64)

def f(values):
    return torch.tensor(values, dtype=torch.float32)

def controls(harness, function, device):
    from _arena_replay import to_device
    for args in control_inputs(harness):
        function(*to_device(args, device))

FUNCTION = 'write_zeros'
MUTABLE = (0,)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return torch.zeros_like(args[0])

def fresh(args):
    args[0].fill_(1)

def control_inputs(harness):
    yield (f([[1,-2,3],[4,5,-6]]),)
    yield (torch.full((7,), torch.finfo(torch.float32).tiny),)

def observe(result, args):
    return args[MUTABLE[0]]
