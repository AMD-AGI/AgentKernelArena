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

FUNCTION = 'rejection_sample'
MUTABLE = ()
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference(*args)

def fresh(args):
    args[0].add_(1).remainder_(100)

def control_inputs(harness):
    yield (l([10,11,12,13,20,21,22,23,30,31,32,33]),
           l([0,10,11,12,0,99,21,22,0,30,99,32]), i([0,4,8,12]), 3)
    yield (l([8,9]), l([0,0]), i([0,1,2]), 0)

def observe(result, args):
    return result

def check(harness, actual, answer, args):
    if not isinstance(actual,(tuple,list)) or len(actual)!=2:
        raise AssertionError('Expected sampled tokens and counts')
    for got, ref in zip(actual,answer):
        if not isinstance(got,torch.Tensor) or (got.shape,got.dtype,got.device)!=(ref.shape,ref.dtype,ref.device):
            raise AssertionError('Sample output shape/dtype/device mismatch')
    if not torch.equal(actual[1],answer[1]):
        raise AssertionError('Incorrect accepted token counts')
    for row, count in enumerate(answer[1].tolist()):
        if not torch.equal(actual[0][row,:count],answer[0][row,:count]):
            raise AssertionError('Incorrect accepted token prefix')
