"""Untimed noncontiguous-input controls with a high-precision log-partition oracle."""
import torch

CASES = [
    {'name': 'strided_logits', 'shape': [2, 1033, 7], 'strided_logits': True, 'strided_ids': False},
    {'name': 'strided_int64_ids', 'shape': [2, 1033, 7], 'strided_logits': False, 'strided_ids': True},
    {'name': 'strided_logits_and_ids', 'shape': [2, 1033, 7], 'strided_logits': True, 'strided_ids': True},
]


def make_case(index, device):
    if type(index) is not int or not 0 <= index < len(CASES):
        raise ValueError('Unknown layout correctness control')
    case = CASES[index]
    batch, vocab, count = case['shape']
    values = ((torch.arange(batch * vocab, device=device).reshape(batch, vocab) % 29) - 14).float() / 3
    values[:, 0], values[:, -1] = 7., -7.
    ids = torch.tensor([[0, vocab - 1, 1, 1024, vocab // 2, 7, vocab - 2],
                        [vocab - 2, 7, vocab // 2, 1024, 1, vocab - 1, 0]],
                       device=device, dtype=torch.int64)
    if case['strided_logits']:
        storage = torch.full((batch, vocab * 2), 900., device=device)
        logits = storage[:, ::2]
        logits.copy_(values)
    else:
        logits = values
    if case['strided_ids']:
        storage = torch.zeros((batch, count * 2), device=device, dtype=torch.int64)
        token_ids = storage[:, ::2]
        token_ids.copy_(ids)
    else:
        token_ids = ids
    return logits, token_ids


def reference(logits, token_ids):
    values = logits.double()
    return (values - torch.logsumexp(values, dim=-1, keepdim=True)).gather(1, token_ids.long()).float()


def validate_reference(harness):
    for index in range(len(CASES)):
        logits, token_ids = make_case(index, 'cpu')
        expected = reference(logits, token_ids)
        original = torch.log_softmax(logits.float(), dim=-1).gather(1, token_ids)
        torch.testing.assert_close(expected, original, atol=1e-2, rtol=1e-2)
        if torch.allclose(torch.zeros_like(expected), expected, atol=1e-2, rtol=1e-2):
            raise AssertionError('Layout control accepts zero output')


def run_control(index, load_module):
    try:
        logits, token_ids = make_case(index, 'cuda')
        pristine = logits.clone(), token_ids.clone()
        expected = reference(*pristine)
        output = load_module().compute_token_logprobs(logits, token_ids)
        if not torch.equal(logits, pristine[0]) or not torch.equal(token_ids, pristine[1]):
            raise AssertionError('Layout control inputs are read-only')
        if (not isinstance(output, torch.Tensor) or output.shape != token_ids.shape
                or output.dtype != torch.float32 or output.device != logits.device):
            raise AssertionError('Layout control output metadata differs from contract')
        if not torch.isfinite(output).all():
            raise AssertionError('Layout control output contains nonfinite values')
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)
        return True, None
    except Exception as exc:
        return False, f'Layout control {index}: {type(exc).__name__}: {exc}'
