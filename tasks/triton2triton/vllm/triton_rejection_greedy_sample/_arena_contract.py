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

FUNCTION = 'rejection_greedy_sample'
MUTABLE = (0,)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    output, ends, draft, target, bonus, greedy, max_spec = args
    output = output.clone()
    start = 0
    for row, end in enumerate(ends.tolist()):
        if greedy is None or bool(greedy[row]):
            for index in range(start,end):
                output[row,index-start] = target[index]
                if target[index] != draft[index]:
                    break
            else:
                output[row,end-start] = bonus[row]
        start = end
    return output

def fresh(args):
    args[3].add_(1).remainder_(100); args[4].add_(1).remainder_(100)

def control_inputs(harness):
    yield (torch.full((4,4),-1,dtype=torch.int64), i([0,1,4,6]), l([7,10,11,12,20,21]),
           l([7,10,11,12,20,99]), l([80,81,82,83]), torch.tensor([True,False,True,True]), 3)
    yield (torch.full((2,3),-1,dtype=torch.int64), i([1,3]), l([5,6,7]), l([9,6,7]), l([80,81]), None, 2)

def observe(result, args):
    return args[MUTABLE[0]]
