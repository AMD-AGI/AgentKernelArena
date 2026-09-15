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

FUNCTION = 'post_update'
MUTABLE = (1, 2, 3, 8, 9)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference_post_update(*args)

def fresh(args):
    args[4].add_(1).remainder_(args[3].shape[1])

def control_inputs(harness):
    yield (i([3,0,2]), i([3,4,5,6]), i([10,11,12,13]), torch.zeros((4,16),dtype=torch.int32),
           i([[1,2,3],[4,4,6],[7,8,9]]), i([0,1,2]), i([1,2,1]), i([0,1,4,7]),
           torch.full((4,16),-9,dtype=torch.int32), i([2,3,4,5]))

def observe(result, args):
    return tuple(args[index] for index in MUTABLE)
