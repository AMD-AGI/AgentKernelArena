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

FUNCTION = 'prepare_eagle_inputs'
MUTABLE = ()
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference(*args)

def fresh(args):
    args[3].add_(1); args[4].add_(2)

def control_inputs(harness):
    args = list(harness.make_inputs(3,4,2))
    args[2] = i([4,0,2]); args[5] = i([0,1,0]); args[6] = i([0,1,2])
    yield tuple(args)

def observe(result, args):
    return result
