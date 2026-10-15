"""Unscored independent known answers for the public operator contract.

The five historical draws remain as correctness-only cases. Small, structured
inputs make omitted state, boundary, grouping or optional-output work observable.
"""
import math
import torch
from scripts.contract_checks import InputSnapshot, check_outputs, comparator_control, ContractFailure, NumericalMismatch


GEOMETRY_CONTROLS = [
    {'test_case_id': 'control_tail_no_initial_state', 'case_index': 30000,
     'shape': [2, 3, 3, 17, 35], 'initial_state': False, 'scale': 0.5},
    {'test_case_id': 'control_tail_initial_state', 'case_index': 30001,
     'shape': [2, 3, 3, 17, 35], 'initial_state': True, 'scale': 0.5},
    {'test_case_id': 'control_decode_multitile', 'case_index': 30002,
     'shape': [2, 1, 4, 64, 64], 'initial_state': True, 'scale': 0.5},
]


def geometry_case(index, device):
    """A closed-form recurrence with active last K column and every V tile.

    q=k=e_(K-1), decay=beta=1/2, v_t=t*f give
    a_t=4**(-t)*a_0 + f/2 * (t/(1-r)-r*(1-r**t)/(1-r)**2), r=1/4.
    An orthogonal state column simply decays by 2**(-t). No token recurrence
    or candidate execution is used to construct these expected answers.
    """
    if type(index) is not int or not 0 <= index < len(GEOMETRY_CONTROLS):
        raise ValueError('Invalid geometry control index')
    case = GEOMETRY_CONTROLS[index]
    batch, time, heads, keys, values = case['shape']
    q = torch.zeros(batch, time, heads, keys, device=device, dtype=torch.float32)
    q[..., -1] = 1.0
    k = q.clone()
    factor = (1 + 4 * torch.arange(batch, device=device, dtype=torch.float32)[:, None, None]
              + 2 * torch.arange(heads, device=device, dtype=torch.float32)[None, :, None]
              + (torch.arange(values, device=device, dtype=torch.float32)[None, None, :] + 1) / (2 * values))
    v = factor[:, None] * torch.arange(1, time + 1, device=device, dtype=torch.float32)[None, :, None, None]
    h0 = torch.zeros(batch, heads, values, keys, device=device, dtype=torch.float32) if case['initial_state'] else None
    if h0 is not None:
        h0[..., -1] = 4 * factor
        h0[..., -2] = 8 * factor
    output = torch.empty(batch, time, heads, values, device=device, dtype=torch.float32)
    history = torch.zeros(batch, time, heads, values, keys, device=device, dtype=torch.float32)
    r = 0.25
    for t in range(1, time + 1):
        coefficient = 0.5 * (t / (1 - r) - r * (1 - r ** t) / (1 - r) ** 2)
        if h0 is not None:
            coefficient += 4 * r ** t
            history[:, t - 1, ..., -2] = factor * (8 * 0.5 ** t)
        history[:, t - 1, ..., -1] = factor * coefficient
        output[:, t - 1] = factor * (coefficient * case['scale'])
    return dict(name=case['test_case_id'], entrypoint='fused_recurrent_gated_delta_rule_fwd',
                kwargs=dict(q=q, k=k, v=v,
                            g=torch.full((batch, time, heads), -math.log(2.), device=device, dtype=torch.float32),
                            beta=torch.full((batch, time, heads), 0.5, device=device, dtype=torch.float32),
                            scale=case['scale'], initial_state=h0),
                expected=(output, history.reshape(batch * time, heads, values, keys)),
                atol=5e-2, rtol=5e-2)


def run_geometry_control(index, load_module, device='cuda'):
    try:
        case = geometry_case(index, device)
        require_signal(case['expected'])
        readonly = InputSnapshot({name: value for name, value in case['kwargs'].items()
                                  if isinstance(value, torch.Tensor)})
        actual = getattr(load_module(), case['entrypoint'])(**case['kwargs'])
        readonly.check()
        check_outputs(actual, case['expected'], atol=case['atol'], rtol=case['rtol'],
                      inputs=[entry[1] for entry in readonly.entries])
        return True, None
    except Exception as exc:
        return False, f'Geometry control {index}: {type(exc).__name__}: {exc}'


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


def reference_outputs(q,k,v,g,beta,scale,initial_state=None):
    """Independent state recurrence, including every returned intermediate state.

    Runs outside timing; the original task reference still checks all scored
    output values separately. The public 'final_state' actually contains T
    states per batch, so comparing only the last state would omit its contract.
    """
    _device=q.device
    batch,time,heads,keys=q.shape;values=v.shape[-1]
    state=(initial_state.detach().cpu().double().clone() if initial_state is not None
           else torch.zeros(batch,heads,values,keys,dtype=torch.float64))
    q,k,v,g,beta=(x.detach().cpu().double() for x in (q,k,v,g,beta))
    outputs=[];states=[]
    for t in range(time):
        decayed=state*g[:,t].exp()[...,None,None]
        residual=v[:,t]-(decayed*k[:,t,:,None,:]).sum(-1)
        state=decayed+(residual*beta[:,t,:,None])[...,None]*k[:,t,:,None,:]
        outputs.append((state*q[:,t,:,None,:]*scale).sum(-1))
        states.append(state.clone())
    return (torch.stack(outputs,dim=1).to(dtype=torch.float32,device=initial_state.device if initial_state is not None else _device),
            torch.stack(states,dim=1).reshape(batch*time,heads,values,keys).to(dtype=torch.float32,device=initial_state.device if initial_state is not None else _device))


def control_cases(device):
    # q=k=e0, beta=1/2 and decay=1/2 imply h0' = h0/4 + v/2.
    # An independent second state column only decays; q does not observe it,
    # so requiring the entire returned history catches state-only corruption.
    for initialized in (False,True):
        q=torch.zeros(2,3,2,16,device=device);q[...,0]=1.;k=q.clone()
        v=torch.empty(2,3,2,16,device=device)
        h0=torch.zeros(2,2,16,16,device=device) if initialized else None
        o=torch.empty_like(v);history=torch.zeros(2,3,2,16,16,device=device)
        for batch in range(2):
            for head in range(2):
                for d in range(16):
                    factor=(batch+1)*(head+1)*(d+1)/16
                    first=4*factor if initialized else 0.
                    second=8*factor if initialized else 0.
                    if initialized:h0[batch,head,d,0]=first;h0[batch,head,d,1]=second
                    for t in range(3):
                        value=(t+1)*factor;v[batch,t,head,d]=value
                        first=first/4+value/2;second/=2
                        o[batch,t,head,d]=first*.5
                        history[batch,t,head,d,0]=first;history[batch,t,head,d,1]=second
        yield dict(name='nonzero_initial_and_full_state_history' if initialized else 'zero_initial_full_state_history',
                   entrypoint='fused_recurrent_gated_delta_rule_fwd',
                   kwargs=dict(q=q,k=k,v=v,g=torch.full((2,3,2),-math.log(2.),device=device),
                               beta=torch.full((2,3,2),.5,device=device),scale=.5,initial_state=h0),
                   expected=(o,history.reshape(6,2,16,16)),atol=5e-2,rtol=5e-2)


def reference_controls(h):
    rows=[]
    for c in control_cases('cpu'):
        _check(h.reference(**c['kwargs']),c['expected'][0],c['atol'],c['rtol'])
        _check(reference_outputs(**c['kwargs']),c['expected'],c['atol'],c['rtol'])
        rows.append(dict(control=c['name'],status='PASS',negative_control='rejected'))
    for index in range(len(GEOMETRY_CONTROLS)):
        c = geometry_case(index, 'cpu')
        _check(h.reference(**c['kwargs']), c['expected'][0], c['atol'], c['rtol'])
        _check(reference_outputs(**c['kwargs']), c['expected'], c['atol'], c['rtol'])
        require_signal(c['expected'])
        rows.append(dict(control=c['name'], status='PASS', negative_control='rejected',
                         independent_closed_form=True))
    return rows


def require_signal(expected):
    """Every returned tensor must reject a zero fill at the public tolerance."""
    for index, value in enumerate(expected):
        try:
            check_outputs(torch.zeros_like(value), value, atol=5e-2, rtol=5e-2)
        except NumericalMismatch:
            continue
        raise ContractFailure(f'Recurrent control output {index} accepts a zero fill')


def amplified_inputs(args):
    # With fixed k/g/beta, the recurrence is linear in the pair (v, h0).
    # Scaling that pair by 8 scales every state by 8; scaling q by 8 as well
    # scales every output by 64. Keep the scored draw and its storage intact.
    return tuple(value * 8 if index in (0, 2, 6) and value is not None else value
                 for index, value in enumerate(args))


def run_scored_shape_control(mod, args, kwargs):
    strong = amplified_inputs(args)
    readonly = InputSnapshot({str(i): value for i, value in enumerate(strong)
                              if isinstance(value, torch.Tensor)})
    expected = reference_outputs(*strong, **kwargs)
    require_signal(expected)
    actual = mod.fused_recurrent_gated_delta_rule_fwd(*strong, **kwargs)
    readonly.check()
    check_outputs(actual, expected, atol=5e-2, rtol=5e-2,
                  inputs=[entry[1] for entry in readonly.entries])
    return {'control': 'scored_shape_nonzero_output_and_full_history',
            'status': 'PASS', 'scored': False, 'zero_output_rejected': True,
            'zero_state_history_rejected': True}


def run_signal_controls(harness, *, case_index, device='cuda'):
    if case_index >= 30000:
        case = geometry_case(case_index - 30000, device)
        require_signal(case['expected'])
        return {'control': case['name'], 'status': 'PASS', 'scored': False,
                'zero_output_rejected': True, 'zero_state_history_rejected': True,
                'independent_closed_form': True}
    if case_index >= 20000:
        return {'control': 'historical_weak_draw', 'status': 'PASS', 'scored': False}
    args, kwargs = harness.gen_inputs(harness.SEEDS[case_index], device, historical=True)
    return run_scored_shape_control(harness.load_module(), args, kwargs)
