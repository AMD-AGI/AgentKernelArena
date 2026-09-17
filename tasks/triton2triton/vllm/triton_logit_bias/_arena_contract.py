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

FUNCTION = 'apply_logit_bias'
MUTABLE = (0,)
ATOL = 0.01
RTOL = 0.01
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference_apply_logit_bias(*args)

def fresh(args):
    args[7].add_(0.25)

def control_inputs(harness):
    yield (torch.arange(32,dtype=torch.float32).reshape(4,8)/10, i([2,0,3,1]), i([0,4,2,9]),
           i([0,2,3,0]), i([[0,0,0],[1,3,0],[0,2,4],[0,0,0]]), i([1,1,2,0]),
           i([[1,0],[3,0],[2,4],[0,0]]), f([[0.5,0],[-1,0],[2,-2],[0,0]]),
           i([2,3,1,4]), i([1,1,1,1]), i([[1],[3],[4],[2]]))

def observe(result, args):
    return args[MUTABLE[0]]


SCORED_CASE_IDS = ('perf_combined_filtering',)
SCORED_TARGET_MS = 1.0

def scored_inputs(harness):
    args = list(next(control_inputs(harness)))
    args[0] = torch.linspace(-2,2,32*1024).reshape(32,1024)
    args[1] = args[1].repeat(8)
    args[2] = args[2].repeat(8)
    yield SCORED_CASE_IDS[0], tuple(args)
