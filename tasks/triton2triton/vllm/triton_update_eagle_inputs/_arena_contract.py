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

FUNCTION = 'update_eagle_inputs'
MUTABLE = (2, 3, 4, 5)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference(*args)

def fresh(args):
    args[0].add_(1); args[1].add_(0.25)

def control_inputs(harness):
    args = list(harness.make_inputs(3,8,16))
    args[3] = i([14,15,16]); args[5] = i([15,16,17])
    yield (*args,16)

def observe(result, args):
    return tuple(args[index] for index in MUTABLE)
