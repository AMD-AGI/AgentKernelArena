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

FUNCTION = 'prepare_mrope_positions'
MUTABLE = (0,)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference_prepare_mrope(*args)

def fresh(args):
    args[1].add_(1); args[3].add_(2)

def control_inputs(harness):
    yield (torch.full((3,7),-9,dtype=torch.int64), torch.arange(120,dtype=torch.int32).reshape(12,10),
           10, i([1,2,-3,4]), i([3,0,2]), i([0,1,4,6]), i([2,8,9,8]), i([2,0,1,4]))

def observe(result, args):
    return args[MUTABLE[0]]
