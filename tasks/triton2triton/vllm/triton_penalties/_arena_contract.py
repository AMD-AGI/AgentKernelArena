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

FUNCTION = 'apply_penalties'
MUTABLE = (0,)
ATOL = 0.01
RTOL = 0.01
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference_apply_penalties(*args)

def fresh(args):
    args[5].add_(0.5)

def control_inputs(harness):
    prompt = torch.zeros((3,1),dtype=torch.int32)
    prompt[0,0]=1<<2; prompt[1,0]=1<<4
    counts = torch.zeros((3,16),dtype=torch.int32); counts[2,3]=2; counts[0,5]=1
    yield (torch.arange(96,dtype=torch.float32).reshape(6,16)/20-2, i([2,2,0,0,1,1]),
           i([3,4,5,6,7,8]), i([0,1,0,1,0,1]), f([1,1.2,0.8]),
           f([0,0.5,-0.5]), f([0,1,-1]), prompt, counts, 1)

def observe(result, args):
    return args[MUTABLE[0]]


SCORED_CASE_IDS = ('perf_speculative_penalties',)
SCORED_TARGET_MS = 20.0

def scored_inputs(harness):
    args = list(next(control_inputs(harness)))
    args[0] = torch.linspace(-2,2,48*1024).reshape(48,1024)
    for index in (1,2,3):
        args[index] = args[index].repeat(8)
    prompt = torch.zeros((3,32),dtype=torch.int32); prompt[:,0] = args[7][:,0]
    counts = torch.zeros((3,1024),dtype=torch.int32); counts[:,:16] = args[8]
    args[7], args[8] = prompt, counts
    yield SCORED_CASE_IDS[0], tuple(args)
