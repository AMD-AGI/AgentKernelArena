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

FUNCTION = 'get_prompt_logprobs_token_ids'
MUTABLE = ()
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    count, qsl, mapping, computed, tokens = args
    out = torch.empty(count, dtype=torch.int64)
    for row, req in enumerate(mapping.tolist()):
        start, end = int(qsl[row]), int(qsl[row+1])
        offset = int(computed[req]) + 1
        out[start:end] = tokens[req, offset:offset + end-start]
    return out

def fresh(args):
    args[4].add_(1)

def control_inputs(harness):
    yield (6, i([0,1,4,6]), i([3,0,2]), i([2,0,1,4]),
           torch.arange(40, dtype=torch.int32).reshape(4,10))

def observe(result, args):
    return result
