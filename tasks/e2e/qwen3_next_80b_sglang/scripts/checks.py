"""Independent FP32 recurrent-state oracle and protected serving integration."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def source_policy():
    """Check the backend-neutral compute bundle; this is not a security sandbox."""
    import json
    from _aka_serving_bundle import Bundle
    lock = json.loads((ROOT / 'runtime.lock.json').read_text())
    return Bundle(ROOT, lock).check_sources()


def launch_command(config, port):
    # Magpie server defaults, with a locked seed instead of a new random seed
    # per launch. This reproducibility control is identical for every role.
    import json
    import shlex
    env = config['envs']
    seed = json.loads((ROOT/'runtime.lock.json').read_text())['server_seed']
    return [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(ROOT/'model'),
            '--host', '127.0.0.1', '--port', str(port), '--trust-remote-code',
            '--random-seed', str(seed),
            '--tensor-parallel-size', str(env['TP']), '--mem-fraction-static=0.8',
            '--disable-radix-cache', *shlex.split(env['EXTRA_SGLANG_ARGS'])], 'get_model_info'


def load_candidate():
    import json
    from types import SimpleNamespace
    from _aka_serving_bundle import effective_function
    lock = json.loads((ROOT/'runtime.lock.json').read_text())
    return SimpleNamespace(fused_recurrent_gated_delta_rule_packed_decode=effective_function(
        ROOT, lock, 'sglang.srt.layers.attention.fla.fused_recurrent',
        'fused_recurrent_gated_delta_rule_packed_decode'))


def audit_kernel_dtypes(input_dtype, state_dtype, output_dtype):
    if (input_dtype, state_dtype, output_dtype) != ('torch.bfloat16', 'torch.float32', 'torch.bfloat16'):
        raise ValueError('Serving kernel requires BF16 activations/output and FP32 recurrent state')


def case_specs():
    """Small, production-sized and padded decode shapes, plus API branches."""
    return ([dict(batch=batch, strided=strided) for batch in (1, 2, 7, 16, 32, 64)
             for strided in (False, True)] + [
        dict(batch=128, pool_size=1012),
        dict(batch=256, pool_size=1016, strided=True),
        dict(batch=512, pool_size=1012),
        dict(batch=512, pool_size=1016, active=32, index_dtype='int32'),
        dict(batch=32, normalize=False),
        dict(batch=7, normalize=False, scale=0.125),
        dict(batch=7, scale=0.25),
    ])


def make_case(batch, strided=False, dtype=None, *, pool_size=None, active=None,
              normalize=True, scale=None, index_dtype='int64', device='cuda'):
    import torch
    dtype = dtype or torch.bfloat16
    H, HV, K, V = 4, 8, 128, 128
    stride = 2 if strided else 1
    qkv = torch.randn(batch*stride, 2*H*K+HV*V, device=device, dtype=dtype)[::stride]
    a = torch.randn(batch*stride, HV, device=device, dtype=dtype)[::stride]
    b = torch.randn(batch*stride, HV, device=device, dtype=dtype)[::stride]
    log = torch.randn(HV, device=device, dtype=torch.float32) * 0.2
    bias = torch.randn(HV, device=device, dtype=torch.float32)
    pool_size = pool_size or batch+3
    state = torch.randn(pool_size, HV, V, K, device=device, dtype=torch.float32) * 0.1
    slots = torch.randperm(pool_size, device=device)[:batch].to(getattr(torch, index_dtype))
    if active is not None:
        slots[active:] = -1
    if batch > 1:
        slots[-1] = -1
    out = torch.empty(batch, 1, HV, V, device=device, dtype=dtype)
    return qkv, a, b, log, bias, K**-0.5 if scale is None else scale, state, out, slots, normalize


def oracle(args):
    import torch
    qkv, a, b, log, bias, scale, state, out, slots, normalize = args
    HV, V, K = state.shape[-3:]
    H = (qkv.shape[-1]-HV*V)//(2*K)
    q,k,v = qkv.float().split([H*K,H*K,HV*V], dim=-1)
    q,k = q.reshape(-1,H,K), k.reshape(-1,H,K)
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)
        k = k * torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
    q = q.repeat_interleave(HV//H,dim=1)*scale
    k = k.repeat_interleave(HV//H,dim=1)
    v = v.reshape(-1,HV,V)
    # The production kernel rounds beta to the activation dtype before use.
    beta = torch.sigmoid(b.float()).to(b.dtype).float()
    decay = torch.exp(-torch.exp(log)*torch.nn.functional.softplus(a.float()+bias,threshold=20))
    expected_state = state.clone()
    expected_out = torch.zeros_like(out)
    for i in range(qkv.shape[0]):
        index = int(slots[i])
        if index < 0:
            continue
        h = state[index].float()*decay[i,:,None,None]
        residual = (v[i]-(h*k[i,:,None,:]).sum(-1))*beta[i,:,None]
        h = h+residual[:,:,None]*k[i,:,None,:]
        expected_out[i,0] = (h*q[i,:,None,:]).sum(-1).to(out.dtype)
        expected_state[index] = h.to(state.dtype)
    return expected_out, expected_state


def compare_tensor(actual, expected, rule, *, state=False):
    """Enforce elementwise limits and a scale-sensitive error bound."""
    import torch
    prefix = 'state_' if state else ''
    torch.testing.assert_close(actual, expected, rtol=rule[prefix+'rtol'], atol=rule[prefix+'atol'])
    delta = actual.float()-expected.float()
    relative_rms = float(torch.linalg.vector_norm(delta)) / max(float(torch.linalg.vector_norm(expected.float())), 1e-30)
    if relative_rms > rule[prefix+'relative_rms']:
        raise ValueError(f'{prefix}relative RMS error {relative_rms} exceeds the locked bound')
    return dict(max_abs_error=float(delta.abs().max()), relative_rms=relative_rms)


def operator_checks(lock):
    source_policy()
    import torch
    torch.manual_seed(lock['correctness']['seed'])
    module = load_candidate()
    function = module.fused_recurrent_gated_delta_rule_packed_decode
    evidence = []
    rule = lock['correctness']
    for spec in case_specs():
        args = make_case(**spec)
        for step in range(4):
            expected, state = oracle(args)
            immutable = [args[i].clone() for i in (0,1,2,3,4,8)]
            result = function(*args)
            torch.cuda.synchronize()
            output_error = compare_tensor(args[7], expected, rule)
            state_error = compare_tensor(args[6], state, rule, state=True)
            if result[0].data_ptr()!=args[7].data_ptr() or result[1].data_ptr()!=args[6].data_ptr():
                raise ValueError('Candidate changed in-place output/state interface')
            for i, before in zip((0,1,2,3,4,8), immutable):
                torch.testing.assert_close(args[i], before, rtol=0,atol=0)
            evidence.append(dict(**spec,step=step,state_checked=True,
                                 output_error=output_error,state_error=state_error))
    # CUDA graph replay exercises the path used by serving, with evolving state.
    for batch in (32, 128, 512):
        spec = dict(batch=batch, pool_size=1016, active=min(batch, 32))
        args=make_case(**spec)
        function(*args)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            function(*args)
        for step in range(4):
            # Keep addresses/state while changing inputs and active slots.
            fresh = make_case(**spec)
            for i in (0,1,2,3,4,8):
                args[i].copy_(fresh[i])
            immutable = [args[i].clone() for i in (0,1,2,3,4,8)]
            expected,state=oracle(args)
            graph.replay()
            torch.cuda.synchronize()
            output_error = compare_tensor(args[7],expected,rule)
            state_error = compare_tensor(args[6],state,rule,state=True)
            for i, before in zip((0,1,2,3,4,8), immutable):
                torch.testing.assert_close(args[i], before, rtol=0,atol=0)
            evidence.append(dict(**spec,replay=step,state_checked=True,inputs_changed=True,
                                 output_error=output_error,state_error=state_error))
        del graph
    return evidence
