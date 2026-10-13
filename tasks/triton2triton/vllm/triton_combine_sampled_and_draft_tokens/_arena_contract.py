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

    args, storage = padded_draft_inputs(harness, device)
    draft_tokens = args[6]
    assert draft_tokens.stride() == (5, 1)
    drafts_before = draft_tokens.clone()
    padding_before = storage[:, 2:].clone()
    function(*args)
    torch.testing.assert_close(draft_tokens, drafts_before, atol=0, rtol=0)
    torch.testing.assert_close(storage[:, 2:], padding_before, atol=0, rtol=0)


def padded_draft_inputs(harness, device):
    from _arena_replay import to_device
    # Construct the view on the target device; moving a CPU slice could erase
    # its padded row stride. Reuse the remapped 0/1/2-draft decode inputs.
    args = list(to_device(tuple(control_inputs(harness))[1], device))
    storage = torch.full((3, 5), -777, dtype=torch.int32, device=device)
    storage[:, :2].copy_(args[6])
    args[6] = storage[:, :2]
    return tuple(args), storage

FUNCTION = 'combine_sampled_and_draft_tokens'
MUTABLE = (0,)
ATOL = 0.0
RTOL = 0.0
CONTROL_INDEX = 5

def reference(harness, args):
    return harness.reference_combine(*args[:8])

def fresh(args):
    args[2].add_(1); args[6].add_(2)

def control_inputs(harness):
    # Mixed prefill/decode requests with different numbers of draft tokens.
    yield (i([-9]*7), i([3,0,2]), l([10,11,12,13]), i([0,1,4,7]), i([3,8,7]),
           i([4,0,2,3]), i([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]), i([0,1,3,6]), 6)
    # The first decode request has no drafts. The other decode requests use
    # remapped state rows and contribute two and one drafts, respectively.
    yield (i([-9]*6), i([2,0,1]), l([101,202,303]), i([0,1,4,6]),
           i([10,20,30]), i([0,0,0]), i([[111,112],[211,212],[311,312]]),
           i([0,1,4,6]), 6)

def observe(result, args):
    return args[0], result
