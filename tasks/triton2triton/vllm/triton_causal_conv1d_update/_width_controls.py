"""Untimed variable-width controls with an independent dense convolution oracle."""
import torch

CASES = [
    {'name': 'width2', 'shape': [2, 17, 1, 2], 'bias': False, 'activation': None},
    {'name': 'width5_silu', 'shape': [2, 33, 1, 5], 'bias': True, 'activation': 'silu'},
    {'name': 'width6', 'shape': [3, 65, 1, 6], 'bias': True, 'activation': None},
]


def make_case(index, device):
    if type(index) is not int or not 0 <= index < len(CASES):
        raise ValueError('Unknown width correctness control')
    case = CASES[index]
    batch, dim, seqlen, width = case['shape']
    feature = torch.arange(dim, device=device, dtype=torch.float32)
    x = (1 + torch.arange(batch, device=device, dtype=torch.float32)[:, None] / 4
         + feature[None, :].remainder(7) / 8).unsqueeze(-1)
    storage = (1 + torch.arange(batch + 1, device=device, dtype=torch.float32)[:, None, None] / 4
               + torch.arange(width - 1, device=device, dtype=torch.float32)[None, :, None] / 8
               + feature[None, None, :].remainder(7) / 8)
    state = storage.contiguous().transpose(1, 2)
    weight = (0.5 + feature[:, None].remainder(3) / 8).expand(dim, width).contiguous()
    bias = (0.25 + feature.remainder(5) / 8) if case['bias'] else None
    indices = torch.tensor([2, 0] if batch == 2 else [3, 0, 2], device=device, dtype=torch.int32)
    return x, state, weight, bias, case['activation'], indices


def reference(x, state, weight, bias, activation, indices):
    # Dense time windows and a tensor contraction are independent of the
    # device kernel's hand-written width-specific registers and branches.
    # This small untimed oracle runs on CPU. Snapshot before the candidate,
    # avoiding a device BLAS dependency for the independent FP64 contraction.
    cpu_x, cpu_state, cpu_weight, cpu_indices = (
        value.detach().cpu() for value in (x, state, weight, indices))
    cpu_bias = None if bias is None else bias.detach().cpu()
    history = torch.cat((cpu_state.index_select(0, cpu_indices.long()), cpu_x), dim=-1)
    windows = history.double().unfold(-1, cpu_weight.shape[-1], 1)
    output = torch.einsum('bdtw,dw->bdt', windows, cpu_weight.double())
    if cpu_bias is not None:
        output += cpu_bias.double()[None, :, None]
    if activation == 'silu':
        output *= output.sigmoid()
    expected_state = cpu_state.clone()
    expected_state.index_copy_(0, cpu_indices.long(), history[..., -cpu_state.shape[-1]:])
    return (output.to(device=x.device, dtype=x.dtype),
            expected_state.to(device=state.device, dtype=state.dtype))


def validate_reference(harness):
    for index in range(len(CASES)):
        args = make_case(index, 'cpu')
        expected, _ = reference(*args)
        x, state, weight, bias, activation, indices = args
        original = harness.reference_conv1d_update(x, state.index_select(0, indices.long()), weight, bias, activation)
        torch.testing.assert_close(original, expected, atol=1e-1, rtol=1e-1)
        if torch.allclose(torch.zeros_like(expected), expected, atol=1e-1, rtol=1e-1):
            raise AssertionError('Width control accepts zero output')


def run_control(index, load_module):
    try:
        x, state, weight, bias, activation, indices = make_case(index, 'cuda')
        expected = reference(x, state, weight, bias, activation, indices)
        readonly = [(value, value.clone()) for value in (weight, bias, indices) if value is not None]
        output = load_module().causal_conv1d_update(x, state, weight, bias=bias,
            activation=activation, conv_state_indices=indices)
        if any(not torch.equal(value, saved) for value, saved in readonly):
            raise AssertionError('Width control modified a read-only input')
        if (not isinstance(output, torch.Tensor) or output.shape != x.shape or output.dtype != x.dtype
                or output.device != x.device or output.data_ptr() != x.data_ptr()):
            raise AssertionError('Width control violates in-place output metadata')
        if not torch.isfinite(output).all() or not torch.isfinite(state).all():
            raise AssertionError('Width control contains nonfinite output/state')
        torch.testing.assert_close(output, expected[0], atol=1e-1, rtol=1e-1)
        if not torch.equal(state, expected[1]):
            raise AssertionError('Width control cached state differs from pristine history')
        return True, None
    except Exception as exc:
        return False, f'Width control {index}: {type(exc).__name__}: {exc}'
