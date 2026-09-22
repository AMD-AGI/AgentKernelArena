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

FUNCTION = 'apply_temperature'
MUTABLE = (0,)
ATOL = 0.01
RTOL = 0.01
CONTROL_INDEX = 5

def reference(harness, args):
    out, mapping, temperature = args
    out = out.clone()
    for row, req in enumerate(mapping.tolist()):
        t = float(temperature[req])
        if t not in (0., 1.):
            out[row] /= t
    return out

def fresh(args):
    args[2].add_(0.25)

def control_inputs(harness):
    yield (f([[1,-2,3,4], [2,1,-1,3], [4,6,-2,0], [2,3,4,5]]),
           i([2,0,3,1]), f([0,1,0.5,2]))

def observe(result, args):
    return args[MUTABLE[0]]
