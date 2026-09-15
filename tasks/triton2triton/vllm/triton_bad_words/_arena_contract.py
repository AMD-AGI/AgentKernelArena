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

FUNCTION = 'apply_bad_words'
MUTABLE = (0,)
ATOL = 0.01
RTOL = 0.01
CONTROL_INDEX = 5

def reference(harness, args):
    logits, mapping, words, offsets, counts, history, prompt_len, total_len, inputs, local, max_words = args
    out = logits.clone()
    for row, req in enumerate(mapping.tolist()):
        pos = int(local[row]); start = row-pos
        seen = history[req,int(prompt_len[req]):int(total_len[req])].tolist() + inputs[start:start+pos].tolist()
        for word in range(int(counts[req])):
            begin, end = int(offsets[req,word]), int(offsets[req,word+1])
            tokens = words[req,begin:end].tolist()
            prefix = tokens[:-1]
            if not prefix or (len(prefix)<=len(seen) and seen[-len(prefix):]==prefix):
                out[row,tokens[-1]] = -torch.inf
    return out

def fresh(args):
    args[2].add_(1).remainder_(args[0].shape[1])

def control_inputs(harness):
    yield (torch.zeros((4,16)), i([1,1,0,0]), i([[3,4,9,2,8],[5,6,10,7,11]]),
           i([[0,3,5],[0,3,5]]), i([2,2]), i([[0,0,3,4,0,0],[0,0,5,0,0,0]]),
           i([2,2]), i([4,3]), i([6,0,2,0]), i([0,1,0,1]), 2)

def observe(result, args):
    return args[MUTABLE[0]]


SCORED_CASE_IDS = ('perf_prefix_routing',)
SCORED_TARGET_MS = 1.0

def scored_inputs(harness):
    args = list(next(control_inputs(harness)))
    args[0] = torch.linspace(-2,2,32*1024).reshape(32,1024)
    args[1] = args[1].repeat(8)
    args[8] = args[8].repeat(8)
    args[9] = args[9].repeat(8)
    yield SCORED_CASE_IDS[0], tuple(args)
