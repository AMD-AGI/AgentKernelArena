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

FUNCTION = 'pack_seq'
MUTABLE = ()
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference_pack_seq(args[0], args[1].tolist(), args[2])

def fresh(args):
    args[0].add_(0.25)

def control_inputs(harness):
    yield (torch.arange(15, dtype=torch.float16).reshape(5,3), i([1,3,1]), -float('inf'), 64, 64)
    yield (torch.arange(15, dtype=torch.float16).reshape(5,3), i([1,3,1]), 7., 64, 64)

def observe(result, args):
    return result

def direct(state):
    return (state['x'], state['lengths'], 0., 64, 64), state['out']
