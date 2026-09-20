"""Single-launch qualification using the retained independent AITER torch recipe.

This is the recipe in ut/_l2_reference.py, not an inverse interpretation of the
scored pre-shuffled random bytes. It adds qualification cases; scored inputs and
launch counts are unchanged. Every individual result must pass the original tol.
"""
import importlib


def run(scope, kernel, reference, seed, contract, torch, check):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import torch_moe_stage1
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle
    retained = importlib.import_module('kimi_l2_reference')
    worker = importlib.import_module('generated_worker')
    meta = scope['META']
    geo = meta['geometry']
    e, topk, model_dim, inter_dim = (int(geo[name]) for name in
                                    ('num_experts', 'topk', 'model_dim', 'inter_dim'))
    if (e, topk, model_dim, inter_dim) != (retained.E, retained.TOPK, retained.MODEL_DIM, retained.INTER_DIM):
        raise RuntimeError('independent reference dimensions differ from the retained recipe')
    kwargs = dict(meta['live_call_kwargs'])
    generator = torch.Generator(device=scope['DEV']).manual_seed(seed)
    w1 = torch.randint(0, 256, (e, 2 * inter_dim, model_dim // 2), dtype=torch.uint8,
                       device=scope['DEV'], generator=generator).view(torch.float4_e2m1fn_x2)
    scales = torch.randint(117, 121, (e * 2 * inter_dim, model_dim // 32), dtype=torch.uint8,
                           device=scope['DEV'], generator=generator).view(torch.float8_e8m0fnu)
    shuffled_w1, shuffled_scales = shuffle_weight(w1, (16, 16)), e8m0_shuffle(scales)
    w2_shape = torch.empty((e, model_dim, inter_dim // 2), dtype=torch.uint8, device='meta')
    rows = []
    for spec in meta['case_specs']:
        if spec['kwargs'] != kwargs:
            raise RuntimeError('single-launch case differs from retained independent launch kwargs')
        m = int(spec['token_num'])
        routing_cpu = scope['ROUTING'][spec['sig']]
        ids, weights = retained.invert_routing(routing_cpu, m, topk, int(geo['sort_block_m']))
        if bool((ids < 0).any()):
            raise RuntimeError('independent routing inversion left unassigned slots')
        ids, weights = ids.to(scope['DEV']), weights.to(scope['DEV'])
        routing = {name: value.to(scope['DEV']) for name, value in routing_cpu.items()}
        inputs = torch.randn((m, model_dim), dtype=torch.bfloat16, device=scope['DEV'],
                             generator=torch.Generator(device=scope['DEV']).manual_seed(int(spec['seed'])))
        if reference:
            answer = torch_moe_stage1(inputs, w1, w2_shape, weights, ids, dtype=torch.bfloat16,
                                     activation=ActivationType.Situv2, quant_type=QuantType.per_1x32,
                                     a1_scale=None, w1_scale=scales, doweight=False,
                                     situ_beta=float(kwargs['situ_beta']),
                                     situ_linear_beta=float(kwargs['situ_linear_beta']))
        previous = []
        for trial in range(int(meta['random_draws'])):
            check()
            if not reference:
                output = torch.empty((m, topk, inter_dim), dtype=torch.bfloat16, device=scope['DEV'])
                readonly = {'a': inputs, 'w1': shuffled_w1, 'scale': shuffled_scales, 'routing': routing}
                def launch(_values):
                    result = kernel(inputs, shuffled_w1, routing['sorted_token_ids'], routing['sorted_expert_ids'],
                                    routing['num_valid_ids'], output, topk, w1_scale=shuffled_scales,
                                    a1_scale=None, sorted_weights=None, **kwargs)
                    return (result[0] if isinstance(result, (tuple, list)) else result) if result is not None else output
                encoded = worker.checked_call(launch, readonly, contract, torch, check, previous)
            else:
                encoded = contract.encode_output(answer, torch)
            torch.cuda.synchronize()
            check()
            rows.append({'id': f"{spec['sig']}:single:{trial}", 'output': encoded})
    return rows
