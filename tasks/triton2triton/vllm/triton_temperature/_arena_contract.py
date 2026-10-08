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

    # Build the view after allocating on the target device: transferring a CPU
    # slice can make it contiguous and silently erase the row-stride control.
    storage = torch.arange(4 * 264, dtype=torch.float32, device=device).reshape(4, 264) + 1
    logits = storage[:, :256]
    mapping = i([2, 0, 3, 1]).to(device)
    temperature = f([0, 1, 0.5, 2]).to(device)
    assert logits.stride() == (264, 1) and logits.stride(0) > logits.shape[1]
    padding = storage[:, 256:].clone()
    function(logits, mapping, temperature)
    torch.testing.assert_close(storage[:, 256:], padding, atol=0, rtol=0)

FUNCTION = 'apply_temperature'
MUTABLE = (0,)
ATOL = 0.01
RTOL = 0.01
CONTROL_INDEX = 5

def reference(harness, args):
    out, mapping, temperature = args
    out = out.clone()
    for row, req in enumerate(mapping.tolist()):
        t = float(temperature[req])
        if t not in (0., 1.):
            out[row] /= t
    return out

def fresh(args):
    args[2].add_(0.25)
    args[1].copy_(args[1].roll(1))

def control_inputs(harness):
    yield (f([[1,-2,3,4], [2,1,-1,3], [4,6,-2,0], [2,3,4,5]]),
           i([2,0,3,1]), f([0,1,0.5,2]))
    rows, vocab = 64, 32768
    yield (torch.linspace(-4, 4, rows * vocab, dtype=torch.float32).reshape(rows, vocab),
           i([(row + 17) % rows for row in range(rows)]),
           f([0, 1, 0.5, 2, 1.5, 0.75, 0.25, 3] * (rows // 8)))

def observe(result, args):
    return args[MUTABLE[0]]
