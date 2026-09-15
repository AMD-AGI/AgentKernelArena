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
    verify_subnormal_rejection(device)
    for args in control_inputs(harness):
        function(*to_device(args, device))

FUNCTION = 'write_zeros'
MUTABLE = (0,)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return torch.zeros_like(args[0])

def fresh(args):
    args[0].fill_(1)

def control_inputs(harness):
    yield (f([[1,-2,3],[4,5,-6]]),)
    for dtype in (torch.float32, torch.float16):
        yield (residual_input(dtype, 'cpu'),)


def residual_input(dtype, device):
    """Construct exact IEEE bits, without floating arithmetic that can flush."""
    if dtype == torch.float32:
        integer, sign, largest_subnormal, smallest_normal = torch.int32, -(1 << 31), 0x7fffff, 0x800000
    elif dtype == torch.float16:
        integer, sign, largest_subnormal, smallest_normal = torch.int16, -(1 << 15), 0x3ff, 0x400
    else:
        raise ValueError('Residual control supports the declared float16/float32 inputs')
    return torch.tensor([[1, sign + 1, 2, sign + 2, largest_subnormal,
                          sign + largest_subnormal, smallest_normal]],
                        dtype=integer, device=device).view(dtype)


def check(harness, output, answer, args):
    if not isinstance(output, torch.Tensor) or (output.shape != answer.shape or
            output.dtype != answer.dtype or output.device != answer.device):
        raise AssertionError('Output shape/dtype/device violates the operator contract')
    if output.dtype == torch.float32:
        integer, mask = torch.int32, 0x7fffffff
    elif output.dtype == torch.float16:
        integer, mask = torch.int16, 0x7fff
    else:
        raise AssertionError('Zero output requires the declared float16/float32 dtype')
    # Ignore only the sign of zero. Integer comparisons reject nonzero IEEE
    # magnitudes (including NaN, infinity and subnormals) even under GPU FTZ.
    if torch.any((output.contiguous().view(integer) & mask) != 0):
        raise AssertionError('Output contains a nonzero bit pattern')


def verify_subnormal_rejection(device):
    """Exercise the real checker on-device as part of contract_controls."""
    for dtype in (torch.float16, torch.float32):
        residuals = residual_input(dtype, device)
        answer = torch.zeros_like(residuals)
        check(None, answer, answer, (answer,))
        # Test both signs independently; a normal value cannot mask acceptance
        # of a subnormal. Creating the tensor through integer storage preserves
        # its bits on the GPU even when floating comparisons flush subnormals.
        for index in (0, 1):
            bad = residuals[:, index:index + 1].contiguous()
            try:
                check(None, bad, torch.zeros_like(bad), (bad,))
            except AssertionError:
                continue
            raise AssertionError('Exact-zero checker accepted a subnormal residual')

def observe(result, args):
    return args[MUTABLE[0]]
