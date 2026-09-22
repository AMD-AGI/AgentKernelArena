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

FUNCTION = 'apply_min_p'
MUTABLE = (0,)
ATOL = 0.01
RTOL = 0.01
CONTROL_INDEX = 5

def reference(harness, args):
    logits, mapping, p = args
    out = logits.clone()
    for row, req in enumerate(mapping.tolist()):
        if p[req] > 0:
            out[row, logits[row] < logits[row].max() + p[req].log()] = -torch.inf
    return out

def fresh(args):
    args[2].fill_(0.75)

def control_inputs(harness):
    yield (f([[0,1,2,3]]).repeat(5,1), i([2,0,3,1,2]), f([0,1,0.1,0.5]))

def observe(result, args):
    return args[MUTABLE[0]]
