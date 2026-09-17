"""Additional unscored public-branch controls from PR105.

Source revision: 0acf65b3a967ef1025dbfc5fd4b415b259e3bd43
Source path: tasks/triton2triton/vllm/triton_ssd_chunk_scan/scripts/task_runner.py
Original scored inputs, gates and timing remain in scripts/task_runner.py.
"""
import sys, os, json, argparse, importlib.util
TEST_SHAPES = [(128, 4, 32, 2, 16, 64), (256, 8, 64, 4, 32, 64), (512, 4, 32, 2, 16, 128), (256, 4, 64, 2, 16, 64), (384, 8, 32, 4, 32, 128)]
CORRECTNESS_CASES = [{'name': f'regular_{i}', 'shape': shape} for i, shape in enumerate(TEST_SHAPES)] + [{'name': 'irregular_boundary_large_dstate', 'shape': (94, 4, 40, 2, 129, 48), 'chunk_lengths': (29, 48, 17), 'seq_idx': (0, 0, 1), 'D_shape': 'head'}, {'name': 'irregular_boundary_initial_state', 'shape': (53, 4, 35, 2, 31, 24), 'chunk_lengths': (24, 11, 18), 'seq_idx': (0, 1, 1), 'initial_states': True, 'disable_rocm_buffer_ops': True}, {'name': 'bf16_irregular_boundary_D_hdim_z', 'shape': (76, 6, 37, 3, 21, 33), 'chunk_lengths': (33, 16, 27), 'seq_idx': (0, 1, 1), 'dtype': 'bfloat16', 'D_shape': 'head_hdim', 'z': True}]

def reference_chunk_scan(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, seq_idx, D=None, z=None, initial_states=None):
    import torch
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = C.shape
    ratio = nheads // ngroups
    nchunks = cb.shape[0]
    cb_c = cb.float().cpu()
    x_c = x.float().cpu()
    dt_c = dt.float().cpu()
    dA_c = dA_cumsum.float().cpu()
    C_c = C.float().cpu()
    states_c = states.float().cpu()
    cu_c = cu_chunk_seqlens.cpu()
    seq_c = seq_idx.cpu()
    D_c = D.float().cpu() if D is not None else None
    z_c = z.float().cpu() if z is not None else None
    initial_states_c = initial_states.float().cpu() if initial_states is not None else None
    out = torch.zeros(seqlen, nheads, headdim, dtype=torch.float32)
    for c in range(nchunks):
        chunk_start = cu_c[c].item()
        chunk_end = cu_c[c + 1].item()
        for h in range(nheads):
            g = h // ratio
            if c == 0 or seq_c[c].item() != seq_c[c - 1].item():
                if initial_states_c is None:
                    prev_state = torch.zeros(headdim, dstate, dtype=torch.float32)
                else:
                    prev_state = initial_states_c[seq_c[c].item(), h]
            else:
                prev_state = states_c[c - 1, h]
            for t in range(chunk_end - chunk_start):
                tok = chunk_start + t
                dA_t = dA_c[h, c, t]
                acc = torch.matmul(prev_state, C_c[tok, g]) * torch.exp(dA_t)
                for k in range(t + 1):
                    tok_k = chunk_start + k
                    coeff = cb_c[c, g, t, k] * torch.exp(dA_t - dA_c[h, c, k]) * dt_c[h, c, k]
                    acc = acc + coeff * x_c[tok_k, h]
                if D_c is not None:
                    acc = acc + x_c[tok, h] * D_c[h]
                if z_c is not None:
                    z_tok = z_c[tok, h]
                    acc = acc * z_tok * torch.sigmoid(z_tok)
                out[tok, h] = acc
    return out
EXTRA_CASES = [{'name': 'irregular_boundary_large_dstate', 'shape': (94, 4, 40, 2, 129, 48), 'chunk_lengths': (29, 48, 17), 'seq_idx': (0, 0, 1), 'D_shape': 'head'}, {'name': 'irregular_boundary_initial_state', 'shape': (53, 4, 35, 2, 31, 24), 'chunk_lengths': (24, 11, 18), 'seq_idx': (0, 1, 1), 'initial_states': True, 'disable_rocm_buffer_ops': True}, {'name': 'bf16_irregular_boundary_D_hdim_z', 'shape': (76, 6, 37, 3, 21, 33), 'chunk_lengths': (33, 16, 27), 'seq_idx': (0, 1, 1), 'dtype': 'bfloat16', 'D_shape': 'head_hdim', 'z': True}]

def run_control(index, load_module):
    if type(index) is not int or not 0 <= index < len(EXTRA_CASES):
        raise ValueError('Unknown upstream correctness control')
    import torch
    try:
        mod = load_module()
        mod = _guarded_module(mod, 'chunk_scan_fwd', ('out',))
    except Exception as e:
        return (False, f'Load failed: {e}')
    device = 'cuda'
    for i, case in enumerate(EXTRA_CASES, start=5):
        if i != index + 5:
            continue
        seqlen, nheads, headdim, ngroups, dstate, chunk_size = case['shape']
        case_name = case['name']
        try:
            torch.manual_seed(42 + i)
            chunk_lengths = case.get('chunk_lengths', (chunk_size,) * (seqlen // chunk_size))
            assert sum(chunk_lengths) == seqlen
            assert all((0 < length <= chunk_size for length in chunk_lengths))
            nchunks = len(chunk_lengths)
            input_dtype = getattr(torch, case.get('dtype', 'float16'))
            cb = torch.randn(nchunks, ngroups, chunk_size, chunk_size, device=device, dtype=input_dtype) * 0.01
            x = torch.randn(seqlen, nheads, headdim, device=device, dtype=input_dtype)
            C = torch.randn(seqlen, ngroups, dstate, device=device, dtype=input_dtype)
            dt = torch.rand(nheads, nchunks, chunk_size, device=device, dtype=torch.float32) * 0.1
            dA_cumsum = torch.cumsum(dt * -0.1, dim=-1)
            states = torch.randn(nchunks, nheads, headdim, dstate, device=device, dtype=torch.float32) * 0.01
            seq_idx = torch.tensor(case.get('seq_idx', (0,) * nchunks), device=device, dtype=torch.int32)
            cu = torch.tensor((0, *chunk_lengths), device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
            initial_states = None
            if case.get('initial_states'):
                nseqs = max(case['seq_idx']) + 1
                initial_states = torch.randn(nseqs, nheads, headdim, dstate, device=device, dtype=torch.float32) * 0.01
            D = None
            if case.get('D_shape') == 'head':
                D = torch.linspace(-0.75, 0.75, nheads, device=device, dtype=torch.float32)
            elif case.get('D_shape') == 'head_hdim':
                D = torch.linspace(-0.5, 0.5, nheads * headdim, device=device, dtype=torch.float32).reshape(nheads, headdim)
            z = torch.randn_like(x) if case.get('z') else None
            out = torch.zeros(seqlen, nheads, headdim, device=device, dtype=torch.float32)
            restore_buffer_ops = None
            if case.get('disable_rocm_buffer_ops') and torch.version.hip is not None:
                import triton
                restore_buffer_ops = triton.knobs.amd.use_buffer_ops
                triton.knobs.amd.use_buffer_ops = False
            try:
                mod.chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, cu, out, seq_idx, D=D, z=z, initial_states=initial_states)
            finally:
                if restore_buffer_ops is not None:
                    triton.knobs.amd.use_buffer_ops = restore_buffer_ops
            torch.cuda.synchronize()
            ref = reference_chunk_scan(cb, x, dt, dA_cumsum, C, states, cu, seq_idx, D=D, z=z, initial_states=initial_states).to(device)
            if not torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.05):
                diff = (out.float() - ref.float()).abs().max().item()
                return (False, f'Case {case_name}: max diff = {diff:.6f}')
        except Exception as e:
            return (False, f'Case {case_name}: {e}')
    return (True, None)

def _guarded_module(module, symbol, mutable_names):
    """Freeze all read-only control inputs around this public candidate call."""
    import inspect
    import torch
    original = getattr(module, symbol)
    signature = inspect.signature(original)

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)

    def invoke(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        readonly = [value for name, arg in bound.arguments.items()
                    if name not in mutable_names for value in tensors(arg)]
        saved = [value.detach().clone() for value in readonly]
        try:
            result = original(*args, **kwargs)
            for value, initial in zip(readonly, saved):
                if (value.shape != initial.shape or value.dtype != initial.dtype
                        or value.device != initial.device or not torch.equal(
                            value.contiguous().reshape(-1).view(torch.uint8),
                            initial.contiguous().reshape(-1).view(torch.uint8))):
                    raise AssertionError('Candidate modified a read-only control input')
            return result
        finally:
            with torch.no_grad():
                for value, initial in zip(readonly, saved):
                    value.copy_(initial)

    class ControlModule:
        def __getattr__(self, name):
            return invoke if name == symbol else getattr(module, name)
    return ControlModule()
