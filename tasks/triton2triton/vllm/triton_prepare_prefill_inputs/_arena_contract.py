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

FUNCTION = 'prepare_prefill_inputs'
MUTABLE = (0, 1)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    ids, next_tokens, mapping, qsl, tokens, lengths, computed = args
    ids, next_tokens = ids.clone(), next_tokens.clone()
    for row, req in enumerate(mapping.tolist()):
        nc, length = int(computed[req]), int(lengths[req])
        if nc >= length:
            continue
        start, end = int(qsl[row]), int(qsl[row+1])
        ids[start:end] = tokens[req, nc:nc+end-start]
        if nc+end-start < length:
            next_tokens[req] = tokens[req,nc+end-start]
    return ids, next_tokens

def fresh(args):
    args[4].add_(1)

def control_inputs(harness):
    yield (i([-9]*6), i([81,82,83,84]), i([3,0,2]), i([0,1,4,6]),
           torch.arange(40, dtype=torch.int32).reshape(4,10), i([5,0,2,8]), i([2,0,2,4]))

def observe(result, args):
    return tuple(args[index] for index in MUTABLE)
