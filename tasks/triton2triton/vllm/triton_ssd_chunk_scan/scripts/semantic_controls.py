"""Unscored independent known answers for the public SSD operator contract.

The historical performance/correctness draws remain as unscored cases. Structured
inputs make omitted state, boundary, grouping or optional-output work observable.
"""
import math
import torch
from scripts.ssd_checks import InputSnapshot, check_outputs, comparator_control, ContractFailure, NumericalMismatch


def _check(got, expected, atol, rtol):
    check_outputs(got, expected, atol=atol, rtol=rtol)
    comparator_control(expected, atol=atol, rtol=rtol)


def run_controls(mod, device="cuda"):
    rows = []
    for case in control_cases(device):
        kwargs = case['kwargs']
        readonly = InputSnapshot({k:v for k,v in kwargs.items()
                                  if isinstance(v, torch.Tensor) and k not in case.get('output_args', ())})
        actual = getattr(mod, case['entrypoint'])(**kwargs)
        if case.get('output_args'):
            buffer = kwargs[case['output_args'][0]]
            if case.get('returns_buffer') and actual is not buffer:
                raise ContractFailure('Public wrapper must return the supplied output buffer')
            actual = buffer
        readonly.check()
        check_outputs(actual, case['expected'], atol=case['atol'], rtol=case['rtol'],
                      inputs=[e[1] for e in readonly.entries])
        rows.append({'control':case['name'], 'status':'PASS', 'scored':False,
                     'full_output_contract':True, 'readonly_inputs':True})
    return rows


def control_cases(device, *, regular=False):
    boundaries=[0,4,8,12] if regular else [0,3,5,9]
    length=boundaries[-1];chunk_size=4
    scale=torch.arange(1,5,device=device).float()
    for mode in (('plain',) if regular else ('plain','initial','D_head','D_dim_z_initial')):
        x=(torch.arange(1,length+1,device=device)[:,None,None]*scale[None,:,None]).expand(length,4,16).half().contiguous()
        kwargs=dict(cb=torch.ones(3,2,4,4,device=device,dtype=torch.float16),x=x,
                    dt=torch.ones(4,3,4,device=device),dA_cumsum=torch.zeros(4,3,4,device=device),
                    C=torch.full((length,2,16),1/16,device=device,dtype=torch.float16),
                    states=(torch.tensor([2.,3.,5.],device=device)[:,None,None,None]*scale[None,:,None,None]).expand(3,4,16,16).contiguous(),
                    cu_chunk_seqlens=torch.tensor(boundaries,dtype=torch.int32,device=device),
                    out=torch.full((length,4,16),float('nan'),device=device),
                    seq_idx=torch.tensor([0,0,1],device=device,dtype=torch.int32))
        has_initial=mode in ('initial','D_dim_z_initial')
        if has_initial:
            kwargs['initial_states']=(torch.tensor([7.,11.],device=device)[:,None,None,None]*scale[None,:,None,None]).expand(2,4,16,16).contiguous()
        if mode=='D_head':kwargs['D']=scale.clone()
        if mode=='D_dim_z_initial':
            kwargs['D']=torch.arange(1,17,device=device).float()[None,:].expand(4,16).contiguous()/4
            kwargs['z']=torch.ones_like(x)
        # Closed-form prefix sums and scalar state contribution, independently
        # of the generic token-loop reference and the candidate implementation.
        expected=torch.empty(length,4,16,device=device)
        for c,(start,end) in enumerate(zip(boundaries,boundaries[1:])):
            prior=[7.,2.,11.][c] if has_initial else [0.,2.,0.][c]
            for t in range(start,end):
                prefix_sum=((t+1)*(t+2)-start*(start+1))/2
                for h in range(4):
                    for dim in range(16):
                        value=(prefix_sum+prior)*(h+1)
                        if mode=='D_head':value+=(t+1)*(h+1)*(h+1)
                        if mode=='D_dim_z_initial':value=(value+(t+1)*(h+1)*(dim+1)/4)/(1+math.exp(-1))
                        expected[t,h,dim]=value
        yield dict(name='causal_grouped_sequence_boundary_'+mode,entrypoint='chunk_scan_fwd',kwargs=kwargs,
                   expected=expected,atol=5e-2,rtol=5e-2,output_args=('out',))


def reference_controls(h):
    c=next(control_cases('cpu',regular=True));k=c['kwargs']
    got=h.reference_chunk_scan(k['cb'],k['x'],k['dt'],k['dA_cumsum'],k['C'],k['states'],k['seq_idx'],4)
    _check(got,c['expected'],5e-2,5e-2)
    return [{'control':'closed_form_causal_prefix_and_boundary_state','status':'PASS','negative_control':'rejected'}]


def inter_chunk_reference(x, dA_cumsum, C, states, seq_idx, chunk_size):
    """Independent inter-chunk term for the five regular scored shapes."""
    length, heads, width = x.shape
    groups = C.shape[1]
    C_cpu, state_cpu, decay = (value.detach().float().cpu()
                                for value in (C, states, dA_cumsum))
    sequence = seq_idx.detach().cpu()
    result = torch.zeros(length, heads, width)
    for chunk in range(length // chunk_size):
        if chunk == 0 or sequence[chunk] != sequence[chunk - 1]:
            continue
        start = chunk * chunk_size
        end = start + chunk_size
        for head in range(heads):
            group = head // (heads // groups)
            result[start:end, head] = (
                C_cpu[start:end, group] @ state_cpu[chunk - 1, head].T
            ) * decay[head, chunk].exp()[:, None]
    return result.to(x.device)


def require_intra_signal(expected, without_intra):
    try:
        check_outputs(without_intra, expected, atol=5e-2, rtol=5e-2)
    except NumericalMismatch:
        return
    raise ContractFailure('SSD control accepts omission of the intra-chunk term')


def scored_shape_inputs(shape, seed, device, *, historical=False):
    length, heads, width, groups, dstate, chunk_size = shape
    chunks = length // chunk_size
    torch.manual_seed(seed)
    # Preserve the original rounded FP16 draw before power-of-two scaling.
    # cb enters only the intra-chunk term, linearly.
    cb = torch.randn(chunks, groups, chunk_size, chunk_size,
                     device=device, dtype=torch.float16) * .01
    if not historical:
        cb.mul_(32)
    x = torch.randn(length, heads, width, device=device, dtype=torch.float16)
    C = torch.randn(length, groups, dstate, device=device, dtype=torch.float16)
    dt = torch.rand(heads, chunks, chunk_size, device=device, dtype=torch.float32) * .1
    decay = torch.cumsum(dt * -.1, dim=-1)
    states = torch.randn(chunks, heads, width, dstate, device=device, dtype=torch.float32) * .01
    sequence = torch.zeros(chunks, device=device, dtype=torch.int32)
    boundaries = torch.arange(chunks + 1, device=device, dtype=torch.int32) * chunk_size
    return dict(cb=cb, x=x, dt=dt, dA_cumsum=decay, C=C, states=states,
                cu_chunk_seqlens=boundaries, seq_idx=sequence,
                out=torch.zeros(length, heads, width, device=device, dtype=torch.float32))


def run_scored_shape_control(mod, reference, shape, seed, device='cuda'):
    kwargs = scored_shape_inputs(shape, seed, device)
    kwargs['out'].fill_(float('nan'))
    readonly = InputSnapshot({name: value for name, value in kwargs.items() if name != 'out'})
    chunk_size = shape[-1]
    args = [kwargs[name] for name in ('cb', 'x', 'dt', 'dA_cumsum', 'C', 'states', 'seq_idx')]
    expected = reference(*args, chunk_size).to(device)
    without_intra = inter_chunk_reference(
        kwargs['x'], kwargs['dA_cumsum'], kwargs['C'], kwargs['states'],
        kwargs['seq_idx'], chunk_size)
    require_intra_signal(expected, without_intra)
    mod.chunk_scan_fwd(**kwargs)
    readonly.check()
    check_outputs(kwargs['out'], expected, atol=5e-2, rtol=5e-2,
                  inputs=[entry[1] for entry in readonly.entries])
    return {'control': 'scored_shape_nonzero_intra_chunk_term', 'status': 'PASS',
            'scored': False, 'omitted_intra_chunk_rejected': True}


def run_signal_controls(harness, *, case_index, device='cuda'):
    if case_index >= 20000:
        return {'control': 'historical_weak_draw', 'status': 'PASS', 'scored': False}
    if case_index >= 10000:
        return {'control': 'upstream_public_branch', 'status': 'PASS', 'scored': False}
    return run_scored_shape_control(
        harness.load_module(), harness.reference_chunk_scan,
        harness.TEST_SHAPES[case_index], 42 + case_index, device)
