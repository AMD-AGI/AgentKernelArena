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

FUNCTION = 'apply_top_k_top_p_triton'
MUTABLE = (0,)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference_apply_top_k_top_p(*args[:3])

def fresh(args):
    if args[1] is not None:
        args[1].div_(2, rounding_mode='floor').clamp_(min=1)
    if args[2] is not None:
        args[2].mul_(0.5)

def control_inputs(harness):
    logits = torch.linspace(-4,4,256).repeat(3,1)
    yield (logits.clone(), None, f([0.5,0.8,1.]), -float('inf'))
    yield (logits.clone(), i([8,32,128]), f([0.5,0.8,1.]), -float('inf'))

def observe(result, args):
    return args[MUTABLE[0]]

def direct(state):
    values, meta = state['kernel_args'], state['kernel_meta']
    return (values[0], values[4] if meta['TOPK_ENABLED'] else None,
            values[5] if meta['TOPP_ENABLED'] else None, meta['MASK_VALUE']), values[0]

def check(harness, actual, answer, args):
    if not isinstance(actual,torch.Tensor) or (actual.shape,actual.dtype,actual.device)!=(answer.shape,answer.dtype,answer.device):
        raise AssertionError('Logits output shape/dtype/device mismatch')
    if torch.isnan(actual).any() or torch.isposinf(actual).any():
        raise AssertionError('Only negative infinity is a valid masked logit')
    limit = 1 if args[2] is None else max(4,args[0].shape[1]//500)
    ok, reason = harness.compare_masked_logits(actual,answer,args[0].shape[1],limit)
    if not ok:
        raise AssertionError(reason)
