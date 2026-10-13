"""Unscored independent known answers for the public operator contract.

The five historical performance cases remain unchanged. These small, structured
inputs make omitted state, boundary, grouping or optional-output work observable.
"""
import math
import torch
from scripts.contract_checks import InputSnapshot, check_outputs, comparator_control, ContractFailure


def _check(got, expected, atol, rtol):
    check_outputs(got, expected, atol=atol, rtol=rtol)
    comparator_control(expected, atol=atol, rtol=rtol)


def run_controls(mod, device="cuda"):
    rows = []
    for case in control_cases(device):
        kwargs = case['kwargs']
        # A strided view can leave storage locations outside its logical values.
        # Check its complete backing tensor as well as the public view.
        inputs = {k:v for k,v in kwargs.items()
                  if isinstance(v, torch.Tensor) and k not in case.get('output_args', ())}
        inputs.update(case.get('storage_owners', {}))
        readonly = InputSnapshot(inputs)
        # torch.equal treats +0.0 and -0.0 as equal. For stepped views, retain
        # the exact bytes of both the logical operands and their full owners.
        byte_snapshot = ({name: value.detach().contiguous().view(torch.uint8).clone()
                          for name, value in inputs.items()}
                         if case.get('storage_owners') else None)
        actual = getattr(mod, case['entrypoint'])(**kwargs)
        if case.get('output_args'):
            buffer = kwargs[case['output_args'][0]]
            if case.get('returns_buffer') and actual is not buffer:
                raise ContractFailure('Public wrapper must return the supplied output buffer')
            actual = buffer
        readonly.check()
        if byte_snapshot is not None:
            for name, value in inputs.items():
                if not torch.equal(value.detach().contiguous().view(torch.uint8),
                                   byte_snapshot[name]):
                    raise ContractFailure(f'Read-only input bytes changed: {name}')
        check_outputs(actual, case['expected'], atol=case['atol'], rtol=case['rtol'],
                      inputs=[e[1] for e in readonly.entries])
        rows.append({'control':case['name'], 'status':'PASS', 'scored':False,
                     'full_output_contract':True, 'readonly_inputs':True})
    return rows


def control_cases(device):
    # Scalar arithmetic is independent of the vectorized F.softplus reference.
    # Cover head-dependent scale, bias, nondefault beta/threshold, and tail tiles.
    for biased in (False, True):
        vals = [-2., -.25, .75, 3., 0.] * 6
        bias = [1., -.5, .25, -1., .5] * 2 if biased else None
        beta, threshold = (2., 1.) if biased else (1., 20.)
        expected = []
        for i,v in enumerate(vals):
            h = (i % 10) // 5
            z = v + (bias[i % 10] if bias else 0.)
            sp = z if beta*z > threshold else math.log1p(math.exp(beta*z))/beta
            expected.append(-(h+1)*sp)
        yield dict(name='bias_beta_threshold_and_tail' if biased else 'scalar_softplus_per_head_scale',
                   entrypoint='fused_kda_gate',
                   kwargs=dict(g=torch.tensor(vals,device=device).reshape(1,3,10),
                               A=torch.tensor([0.,math.log(2.)],device=device),head_k_dim=5,
                               g_bias=torch.tensor(bias,device=device) if bias else None,
                               beta_val=beta,threshold=threshold),
                   expected=torch.tensor(expected,device=device).reshape(1,3,2,5),atol=1e-3,rtol=1e-3)

    # H=1 and H=3 expose specialization to the scored H=4 or legacy H=2;
    # D=33/65 cross BD=32/64 with non-power-of-two tails. H=4,D=32 also
    # exercises a heterogeneous, nonzero bias at the scored head geometry.
    for H,D in ((1,33),(3,65),(4,32)):
        B,T=2,3
        values=[];bias=[];expected=[]
        scales=[math.log(head+1) for head in range(H)]
        for head in range(H):
            for d in range(D):
                bias.append((head+1)*.125 + (d%3)*.05)
        for b in range(B):
            for t in range(T):
                for head in range(H):
                    for d in range(D):
                        value=(b+1)*.25+t*.125+(head+1)*.5+(d%7)*.1-1.25
                        values.append(value)
                        z=value+bias[head*D+d]
                        sp=z if 1.5*z>1. else math.log1p(math.exp(1.5*z))/1.5
                        expected.append(-(head+1)*sp)
        name = (f'head_dimension_bias_h{H}_d{D}' if H == 4
                else f'head_dimension_tail_h{H}_d{D}')
        yield dict(name=name,
                   entrypoint='fused_kda_gate',
                   kwargs=dict(g=torch.tensor(values,device=device).reshape(B,T,H*D),
                               A=torch.tensor(scales,device=device),head_k_dim=D,
                               g_bias=torch.tensor(bias,device=device),
                               beta_val=1.5,threshold=1.),
                   expected=torch.tensor(expected,device=device).reshape(B,T,H,D),
                   atol=1e-3,rtol=1e-3)

    # The public wrapper accepts views, while its JIT uses flat unit-stride
    # pointer arithmetic. Exercise each operand independently with a stepped
    # view and preserve the complete backing storage for the read-only guard.
    B,T,H,D = 1,2,4,32
    values = []
    bias = []
    expected = []
    scales = [math.log(head + 1) for head in range(H)]
    for head in range(H):
        for d in range(D):
            bias.append((head - 1.5) * .2 + (d % 5 - 2) * .035)
    for t in range(T):
        for head in range(H):
            for d in range(D):
                value = (t + 1) * .2 + (head - 1.5) * .25 + (d % 7 - 3) * .09
                values.append(value)
                z = value + bias[head * D + d]
                sp = z if 1.5 * z > 1. else math.log1p(math.exp(1.5 * z)) / 1.5
                expected.append(-(head + 1) * sp)
    plain_g = torch.tensor(values, device=device).reshape(B,T,H*D)
    plain_A = torch.tensor(scales, device=device)
    plain_bias = torch.tensor(bias, device=device)
    expected = torch.tensor(expected, device=device).reshape(B,T,H,D)
    for operand in ('g', 'A', 'g_bias'):
        plain = {'g': plain_g, 'A': plain_A, 'g_bias': plain_bias}[operand]
        backing = torch.empty(plain.numel() * 2, device=device)
        backing[::2].copy_(plain.reshape(-1))
        backing[1::2].fill_(2.5)
        view = backing[::2].reshape(plain.shape)
        kwargs = dict(g=plain_g.clone(), A=plain_A.clone(), head_k_dim=D,
                      g_bias=plain_bias.clone(), beta_val=1.5, threshold=1.)
        kwargs[operand] = view
        yield dict(name=f'strided_{operand}_h4_d32', entrypoint='fused_kda_gate',
                   kwargs=kwargs, storage_owners={f'{operand}_storage': backing},
                   expected=expected, atol=1e-3, rtol=1e-3)


def reference_controls(h):
    rows=[]
    for c in control_cases('cpu'):
        _check(h.reference(**c['kwargs']),c['expected'],c['atol'],c['rtol'])
        rows.append(dict(control=c['name'],status='PASS',negative_control='rejected'))
    return rows
