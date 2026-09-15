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

FUNCTION = 'prepare_pos_seq_lens'
MUTABLE = (3, 4)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    mapping, qsl, computed, pos, seq = args
    pos, seq = pos.clone(), seq.clone()
    for row, req in enumerate(mapping.tolist()):
        start, end = int(qsl[row]), int(qsl[row+1])
        pos[start:end] = torch.arange(int(computed[req]), int(computed[req])+end-start, dtype=pos.dtype)
        seq[row] = computed[req]+end-start
    seq[len(mapping):] = 0
    return pos, seq

def fresh(args):
    args[2].add_(1)

def control_inputs(harness):
    yield (i([3,0,2]), i([0,1,4,6]), i([2,0,7,4]),
           torch.full((6,), -9, dtype=torch.int64), i([-8,-8,-8,91,92]))

def observe(result, args):
    return tuple(args[index] for index in MUTABLE)
