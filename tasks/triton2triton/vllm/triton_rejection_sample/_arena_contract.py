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


RAGGED_CONTROLS = {6: {'case_index': 6,
     'ragged_case': 'ragged_bonus_full_early',
     'spans': [1, 3, 5, 2],
     'cu_num_logits': [0, 1, 4, 9, 11],
     'num_speculative_steps': 4,
     'reject_positions': [None, None, 0, None],
     'expected_counts': [1, 3, 1, 2]},
 7: {'case_index': 7,
     'ragged_case': 'ragged_late_full_bonus',
     'spans': [4, 2, 6, 1],
     'cu_num_logits': [0, 4, 6, 12, 13],
     'num_speculative_steps': 5,
     'reject_positions': [2, None, None, None],
     'expected_counts': [3, 2, 6, 1]}}


def ragged_inputs(case_index):
    config = RAGGED_CONTROLS[case_index]
    target, draft = [], []
    for request, (span, rejection) in enumerate(zip(config['spans'], config['reject_positions'])):
        tokens = [10 + request * 20 + offset for offset in range(span)]
        proposed = [0] + tokens[:-1]
        if rejection is not None:
            proposed[rejection + 1] = (tokens[rejection] + 1) % 100
        target.extend(tokens)
        draft.extend(proposed)
    return (l(target), l(draft), i(config['cu_num_logits']), config['num_speculative_steps'])


def ragged_scalar_answer(args):
    # Independent prefix selection rather than mirroring the kernel's writes
    # and rejected flag. Every active request has at least one target/bonus logit.
    target, draft, offsets, steps = args
    output = torch.empty(len(offsets)-1, steps+1, dtype=target.dtype)
    counts = []
    for request, (begin, end) in enumerate(zip(offsets.tolist(), offsets.tolist()[1:])):
        choices = target[begin:end].tolist()
        proposals = draft[begin+1:end].tolist()
        if not 1 <= len(choices) <= steps+1:
            raise ValueError('Ragged control requires a positive active-request span within output capacity')
        mismatch = next((index for index, pair in enumerate(zip(choices[:-1], proposals))
                         if pair[0] != pair[1]), None)
        count = len(choices) if mismatch is None else mismatch+1
        output[request, :count] = target[begin:begin+count]
        counts.append(count)
    return output, i(counts)


def run_ragged_control(harness, function, device, case_index):
    from _arena_replay import clone, restore, to_device, unchanged
    cpu_args = ragged_inputs(case_index)
    cpu_answer = ragged_scalar_answer(cpu_args)
    config = RAGGED_CONTROLS[case_index]
    if cpu_answer[1].tolist() != config['expected_counts']:
        raise AssertionError('Ragged scalar answer disagrees with protected case identity')
    args = to_device(cpu_args, device)
    saved = clone(args)
    try:
        actual = function(*args)
        unchanged(args, saved, MUTABLE)
        check(harness, actual, to_device(cpu_answer, device), args)
        return {'unscored_control': True, 'ragged_case': config['ragged_case'],
                'spans': config['spans'], 'cu_num_logits': config['cu_num_logits'],
                'num_sampled': actual[1].tolist(), 'independent_scalar_oracle_checked': True,
                'readonly_inputs_checked': True}
    finally:
        restore(args, saved)
