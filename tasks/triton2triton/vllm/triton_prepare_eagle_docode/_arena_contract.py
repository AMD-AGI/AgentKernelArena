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

FUNCTION = 'prepare_eagle_decode'
MUTABLE = (5, 6, 7, 8, 9)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference(*args)

def fresh(args):
    args[0].add_(1); args[1].add_(0.25)

def control_inputs(harness):
    args = list(harness.make_inputs(3,6,8,32,5))
    args[5][:3] = i([30,31,32]); args[3] = i([31,32,33]); args[4] = i([0,0,1])
    yield (*args,32,5)

def observe(result, args):
    return tuple(args[index] for index in MUTABLE)
