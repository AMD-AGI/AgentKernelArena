"""CPU-owned fresh inputs and the packed recurrent-step mathematical oracle."""
from __future__ import annotations

CHALLENGES = ('zero_query_key', 'zero_state', 'inactive_last_row', 'softplus_branch')
IMMUTABLE = ('mixed_qkv', 'a', 'b', 'A_log', 'dt_bias', 'ssm_state_indices')


def cpu_reference(torch, args):
    """FP32 state update with the native BF16 beta rounding and output cast."""
    tensors = [v for v in args.values() if isinstance(v, torch.Tensor)]
    if not tensors or any(v.device.type != 'cpu' for v in tensors):
        raise ValueError('The recurrent mathematical oracle accepts CPU inputs only')
    state = args['initial_state'].clone()
    batch = args['mixed_qkv'].shape[0]
    hv, width_v, width_k = state.shape[1:]
    packed = args['mixed_qkv'].float()
    heads = (packed.shape[1] - hv * width_v) // (2 * width_k)
    q = packed[:, :heads * width_k].reshape(batch, heads, width_k)
    k = packed[:, heads * width_k:2 * heads * width_k].reshape(batch, heads, width_k)
    v = packed[:, 2 * heads * width_k:].reshape(batch, hv, width_v)
    if args['use_qk_l2norm_in_kernel']:
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q.repeat_interleave(hv // heads, dim=1) * args['scale']
    k = k.repeat_interleave(hv // heads, dim=1)
    gate_x = args['a'].float() + args['dt_bias'].float()
    softplus = torch.where(gate_x <= 20.0, torch.log1p(torch.exp(gate_x)), gate_x)
    decay = torch.exp(-torch.exp(args['A_log'].float()) * softplus)
    # The native kernel rounds sigmoid(b) through the input b dtype.
    beta = torch.sigmoid(args['b'].float()).to(args['b'].dtype).float()
    indices = args['ssm_state_indices'].to(torch.int64)
    valid = indices >= 0
    slots = indices[valid]
    if slots.numel() and (int(slots.max()) >= state.shape[0] or slots.unique().numel() != slots.numel()):
        raise ValueError('Active recurrent slots must be unique and in range')
    output = torch.zeros((batch, 1, hv, width_v), dtype=args['out'].dtype)
    if slots.numel():
        before = state.index_select(0, slots).float() * decay[valid, :, None, None]
        delta = (v[valid] - (before * k[valid, :, None, :]).sum(-1)) * beta[valid, :, None]
        after = before + delta[:, :, :, None] * k[valid, :, None, :]
        output[valid, 0] = (after * q[valid, :, None, :]).sum(-1).to(output.dtype)
        state.index_copy_(0, slots, after.to(state.dtype))
    return output, state


def tensor_contract(value):
    return {'shape':list(value.shape), 'stride':list(value.stride()), 'dtype':str(value.dtype)}


def case_params(args, spec, geometry):
    return {'regime':spec['regime'], 'm':int(spec['m']), 'B':int(spec['B']),
            'H':int(geometry['H']), 'HV':int(geometry['HV']), 'K':int(geometry['K']),
            'V':int(geometry['V']), 'qkv_dim':int(geometry['qkv_dim']),
            'state_pool_slots':int(geometry['state_pool_slots']), 'scale':float(args['scale']),
            'use_qk_l2norm_in_kernel':bool(args['use_qk_l2norm_in_kernel']),
            'state_indices':list(spec['state_indices']),
            'tensors':{name:tensor_contract(args[name]) for name in (*IMMUTABLE, 'initial_state', 'out')}}


class CPUInputs:
    def __init__(self, torch, spec, geometry, seed):
        self.torch, self.spec, self.geometry = torch, spec, geometry
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.generated = 0

    def normal(self, shape, dtype, scale):
        return (self.torch.randn(shape, generator=self.rng, dtype=self.torch.float32) * scale).to(dtype)

    def next(self, kind='ordinary'):
        if kind not in ('ordinary', *CHALLENGES):
            raise ValueError('Unknown recurrent challenge')
        t, geo, spec = self.torch, self.geometry, self.spec
        batch, hv, k, v = int(spec['B']), int(geo['HV']), int(geo['K']), int(geo['V'])
        A = 1.0 + 15.0 * t.rand((hv,), generator=self.rng)
        dt = 1e-3 + (1e-1 - 1e-3) * t.rand((hv,), generator=self.rng)
        args = {
            'mixed_qkv':self.normal((batch, int(geo['qkv_dim'])), t.bfloat16, 0.5),
            'a':self.normal((batch, hv), t.bfloat16, 0.5),
            'b':self.normal((batch, hv), t.bfloat16, 0.5),
            'A_log':t.log(A), 'dt_bias':t.log(t.expm1(dt)).to(t.bfloat16),
            'scale':float(geo['scale']),
            'initial_state':self.normal((int(geo['state_pool_slots']), hv, v, k), t.float32, 0.05),
            'out':t.zeros((batch, 1, hv, v), dtype=t.bfloat16),
            'ssm_state_indices':t.tensor(spec['state_indices'], dtype=t.int32),
            'use_qk_l2norm_in_kernel':bool(geo['use_qk_l2norm_in_kernel']),
        }
        if kind == 'zero_query_key':
            args['mixed_qkv'][:, :2 * int(geo['H']) * k].zero_()
        elif kind == 'zero_state':
            args['initial_state'].zero_()
        elif kind == 'inactive_last_row':
            args['ssm_state_indices'][-1] = -1
        elif kind == 'softplus_branch':
            args['a'][:, ::2] = 21.0
            args['a'][:, 1::2] = -21.0
            args['dt_bias'].zero_()
        self.generated += 1
        return args, cpu_reference(t, args)
