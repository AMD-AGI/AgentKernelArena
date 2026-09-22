"""Unscored GPT-J RoPE branch controls from PR105; original decode rows unchanged."""
import torch
from _timed_contract import checked_call

CASES = [
    {'test_case_id': 'control-mla-rope-single',
     'params': {'ctx_len': 21, 'batch_size': 1, 'nhead': 16, 'rotary_dim': 64, 'seed': 42}},
    {'test_case_id': 'control-mla-rope-multiple',
     'params': {'ctx_len': 64, 'batch_size': 3, 'nhead': 16, 'rotary_dim': 64, 'seed': 42}},
]


def rotate(x, cos_sin_cache, positions):
    """Independent adjacent-pair rotation, rounded to the input's BF16 storage."""
    cosine, sine = cos_sin_cache[positions].float().chunk(2, dim=-1)
    while cosine.ndim < x.ndim:
        cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    return torch.stack((even * cosine - odd * sine,
                        odd * cosine + even * sine), -1).flatten(-2).to(x.dtype)


def transformed(harness, inputs, cache, positions):
    """Transform private operands before the existing independent attention oracle."""
    query, keys = inputs['q'].clone(), inputs['k_input'].clone()
    rank = inputs['kv_lora_rank']
    query[..., rank:] = rotate(query[..., rank:], cache, positions)
    last = inputs['kv_indptr'][1:].long() - 1
    selected = inputs['kv_indices'][last].long()
    rotated_key = rotate(keys[selected, 0, rank:], cache, positions)
    keys[selected, 0, rank:] = rotated_key
    return {**inputs, 'q': query, 'k_input': keys}, rotated_key


def run(harness, case):
    shape = case['params']
    inputs = harness.setup_inputs(shape['ctx_len'], shape['batch_size'], shape['nhead'])
    dimension = shape['rotary_dim']
    frequencies = torch.outer(torch.arange(shape['ctx_len'] + 1, device=inputs['q'].device).float(),
                              10000.0 ** (-torch.arange(0, dimension, 2, device=inputs['q'].device).float() / dimension))
    cache = torch.cat((frequencies.cos(), frequencies.sin()), -1).to(inputs['q'].dtype)
    positions = torch.full((shape['batch_size'],), shape['ctx_len'], dtype=torch.int64, device=inputs['q'].device)
    key_output = torch.full((shape['batch_size'], dimension), torch.nan,
                            dtype=inputs['q'].dtype, device=inputs['q'].device)
    readonly = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)
                and k not in ('output', 'attn_logits')}
    readonly.update(rope_cache=cache, positions=positions)
    numeric = {}

    def reference(saved):
        private, expected_key = transformed(harness, {**inputs, **saved}, saved['rope_cache'], saved['positions'])
        protected, oracle, check = harness._mla_contract(private)
        expected = oracle(protected)
        numeric['check'] = check
        return expected, expected_key

    def invoke():
        harness.decode_attention_fwd_grouped_rope(
            inputs['q'], inputs['k_input'], inputs['v_input'], inputs['output'],
            inputs['kv_indptr'], inputs['kv_indices'], key_output, inputs['kv_lora_rank'],
            dimension, cache, positions, inputs['attn_logits'], inputs['num_kv_splits'],
            sm_scale=inputs['sm_scale'], logit_cap=0.0, use_rope=True)
        return inputs['output'], key_output

    def check(actual, expected):
        numeric['check'](actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-2, atol=1e-2)

    checked_call(invoke, inputs=readonly, reference=reference, check=check)
