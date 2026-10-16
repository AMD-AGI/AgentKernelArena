"""Protected independent default/explicit non-normalized decode controls."""
import torch


def manifest_cases():
    return [
        {'test_case_id': 'control_qk_false_explicit', 'dtype': 'bfloat16',
         'params': {'B': 2, 'H': 1, 'HV': 2, 'K': 128, 'V': 128, 'pool': 4,
                    'qk_l2norm': False, 'argument': 'explicit_false', 'scale': .25,
                    'input_distribution': 'independent_dyadic_closed_form'},
         'checks': ['correctness']},
        {'test_case_id': 'control_qk_false_default', 'dtype': 'bfloat16',
         'params': {'B': 2, 'H': 1, 'HV': 2, 'K': 128, 'V': 128, 'pool': 4,
                    'qk_l2norm': False, 'argument': 'omitted_default', 'scale': .25,
                    'input_distribution': 'independent_dyadic_closed_form'},
         'checks': ['correctness']},
    ]


def make_case(declaration, device):
    p = declaration['params']
    B, H, HV, K, V, pool = (p[key] for key in ('B', 'H', 'HV', 'K', 'V', 'pool'))
    mixed = torch.zeros(B, 2 * H * K + HV * V, dtype=torch.bfloat16)
    a = torch.zeros(B, HV, dtype=torch.bfloat16)
    b = torch.zeros_like(a)
    alog = torch.zeros(HV, dtype=torch.bfloat16)
    bias = torch.zeros_like(alog)
    indices = torch.tensor([2, 0], dtype=torch.int32)
    slot = torch.arange(pool)[:, None, None, None]
    head = torch.arange(HV)[None, :, None, None]
    value = torch.arange(V)[None, None, :, None] % 4
    feature = torch.arange(K)[None, None, None, :] % 4
    initial = ((1 + slot + head + value + feature) / 16.).to(torch.bfloat16)
    expected_state = initial.float().clone()
    expected_out = torch.empty(B, 1, HV, V, dtype=torch.bfloat16)
    for token in range(B):
        q0, qlast = 2. + token, 1.
        k0, klast = .5, -1. / (token + 1)
        mixed[token, 0] = q0
        mixed[token, K - 1] = qlast
        mixed[token, H * K] = k0
        mixed[token, 2 * H * K - 1] = klast
        state_slot = int(indices[token])
        for hv in range(HV):
            values = 2. + (torch.arange(V) % 4) * .5 + hv * .25 + token * .5
            mixed[token, 2 * H * K + hv * V:2 * H * K + (hv + 1) * V] = values
            pristine = initial[state_slot, hv].float()
            # exp(-softplus(0))=1/2 and sigmoid(0)=1/2 exactly. Only
            # k[0],k[K-1] contribute; all other features simply decay by 1/2.
            residual = values - .5 * (pristine[:, 0] * k0 + pristine[:, -1] * klast)
            updated = pristine * .5
            updated[:, 0] += .5 * residual * k0
            updated[:, -1] += .5 * residual * klast
            expected_state[state_slot, hv] = updated
            expected_out[token, 0, hv] = ((updated[:, 0] * q0 + updated[:, -1] * qlast) * p['scale']).to(torch.bfloat16)
    inputs = {'B': B, 'H': H, 'HV': HV, 'K': K, 'V': V, 'pool_size': pool,
              'mixed_qkv': mixed, 'a': a, 'b': b, 'A_log': alog, 'dt_bias': bias,
              'ssm_states': initial, 'cache_indices': indices, 'scale': p['scale']}
    inputs = {name: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
              for name, tensor in inputs.items()}
    return inputs, (expected_out.to(device), expected_state.to(torch.bfloat16).to(device))


def reference_self_test(reference, compare):
    rows = []
    for declaration in manifest_cases():
        inp, expected = make_case(declaration, 'cpu')
        actual = reference(inp, use_qk_l2norm_in_kernel=False)
        compare(actual, expected, inp)
        try:
            compare(reference(inp, use_qk_l2norm_in_kernel=True), expected, inp)
        except AssertionError:
            pass
        else:
            raise AssertionError('Non-unit q/k failed to reject forced normalization')
        rows.append({'control': declaration['test_case_id'], 'known_answer': 'PASS',
                     'negative_output': 'rejected'})
    return rows


def run_cases(mod, compare, require_unchanged, device='cuda'):
    records = []
    for index, declaration in enumerate(manifest_cases(), 11):
        record = {'shape_id': index, 'shape': [2, 1, 2, 128, 128, 4],
                  'control': declaration['test_case_id'], 'qk_l2norm': False,
                  'argument': declaration['params']['argument'], 'passed': False}
        tensors, originals = (), ()
        try:
            inp, expected = make_case(declaration, device)
            tensors = tuple(value for value in inp.values() if isinstance(value, torch.Tensor))
            originals = tuple(value.clone() for value in tensors)
            state = inp['ssm_states'].clone()
            out = inp['mixed_qkv'].new_full((2, 1, 2, 128), float('nan'))
            kwargs = dict(mixed_qkv=inp['mixed_qkv'], a=inp['a'], b=inp['b'],
                          A_log=inp['A_log'], dt_bias=inp['dt_bias'], scale=inp['scale'],
                          initial_state=state, out=out, ssm_state_indices=inp['cache_indices'])
            if declaration['params']['argument'] == 'explicit_false':
                kwargs['use_qk_l2norm_in_kernel'] = False
            returned = mod.fused_recurrent_gated_delta_rule_packed_decode(**kwargs)
            require_unchanged(tensors, originals)
            if (not isinstance(returned, tuple) or len(returned) != 2
                    or returned[0] is not out or returned[1] is not state):
                raise AssertionError('Decode must return the supplied output and working state')
            if any(out.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() for value in (*tensors, state)):
                raise AssertionError('Decode output aliases readonly input or state')
            record['out_max_diff'] = (out.float() - expected[0].float()).abs().max().item()
            record['state_max_diff'] = (state.float() - expected[1].float()).abs().max().item()
            compare((out, state), expected, inp)
            record['passed'] = True
        except Exception as error:
            record['error'] = type(error).__name__ + ': ' + str(error)
        finally:
            for tensor, original in zip(tensors, originals):
                tensor.copy_(original)
        records.append(record)
    return records
