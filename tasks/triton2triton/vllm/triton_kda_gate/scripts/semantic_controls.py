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


def reference_controls(h):
    rows=[]
    for c in control_cases('cpu'):
        _check(h.reference(**c['kwargs']),c['expected'],c['atol'],c['rtol'])
        rows.append(dict(control=c['name'],status='PASS',negative_control='rejected'))
    return rows
