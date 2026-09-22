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

FUNCTION = 'compute_ranks'
MUTABLE = ()
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    logits, token_ids = args
    return (logits >= logits.gather(1, token_ids.long()[:, None])).sum(-1)

def fresh(args):
    args[1].add_(1).remainder_(args[0].shape[1])

def control_inputs(harness):
    yield (f([[2,2,1,3],[1,1,1,1],[3,2,1,0]]), l([0,2,3]))

def observe(result, args):
    return result
