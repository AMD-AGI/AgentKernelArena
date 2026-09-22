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

FUNCTION = 'combine_sampled_and_draft_tokens'
MUTABLE = (0,)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference_combine(*args[:8])

def fresh(args):
    args[2].add_(1); args[6].add_(2)

def control_inputs(harness):
    yield (i([-9]*7), i([3,0,2]), l([10,11,12,13]), i([0,1,4,7]), i([3,8,7]),
           i([4,0,2,3]), i([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), i([0,1,3,6]), 6)

def observe(result, args):
    return args[0], result
