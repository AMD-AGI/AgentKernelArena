"""Unscored independent known answers for the public SSD operator contract.

The five historical performance cases remain unchanged. These small, structured
inputs make omitted state, boundary, grouping or optional-output work observable.
"""
import math
import torch
from scripts.ssd_checks import InputSnapshot, check_outputs, comparator_control, ContractFailure


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


def reference_controls(h):
    dt = torch.tensor([[-2.,1.],[0.,2.],[2.,3.],[4.,4.]])
    A = torch.tensor([-1.,-2.]); bias = torch.tensor([1.,-1.])
    cu = torch.tensor([0,2,4], dtype=torch.int32)
    expected_dt = torch.tensor([[[0.,1.],[3.,5.]], [[0.,1.],[2.,3.]]])
    expected_da = torch.tensor([[[0.,-1.],[-3.,-8.]], [[0.,-2.],[-4.,-10.]]])
    _check(h.reference(dt,A,2,cu,bias,False), (expected_da,expected_dt), 1e-3,1e-3)
    # Independent scalar softplus formula, including its linear branch >20.
    dt = torch.tensor([[-30.],[0.],[1.],[25.]])
    transformed = torch.tensor([math.log1p(math.exp(-30.)), math.log(2.),
                                math.log1p(math.e),25.]).reshape(1,1,4)
    expected = (-transformed.cumsum(-1), transformed)
    _check(h.reference(dt,torch.tensor([-1.]),4,torch.tensor([0,4]),None,True), expected,1e-3,1e-3)
    return [{'control':'signed_dt_bias_chunk_reset_known_answer','status':'PASS','negative_control':'rejected'},
            {'control':'scalar_softplus_threshold_known_answer','status':'PASS','negative_control':'rejected'}]


def control_cases(device):
    dt = torch.tensor([[-2.,1.],[0.,2.],[2.,3.],[4.,4.],[6.,5.]],device=device)
    transformed = torch.tensor([[[.5,1.,3.,0.],[3.,3.,0.,0.]],
                                [[.5,1.,2.,0.],[3.,3.,0.,0.]]],device=device)
    decay = torch.tensor([[[-.5,-1.5,-4.5,-4.5],[-3.,-6.,-6.,-6.]],
                          [[-1.,-3.,-7.,-7.],[-6.,-12.,-12.,-12.]]],device=device)
    yield dict(name='partial_chunks_bias_clamp_and_padded_cumsum', entrypoint='chunk_cumsum_fwd',
               kwargs=dict(dt=dt,A=torch.tensor([-1.,-2.],device=device),chunk_size=4,
                           cu_chunk_seqlens=torch.tensor([0,3,5],dtype=torch.int32,device=device),
                           dt_bias=torch.tensor([1.,-1.],device=device),dt_limit=(.5,3.)),
               expected=(decay,transformed),atol=1e-3,rtol=1e-3)
    vals=[-30.,0.,1.,25.]
    transformed=torch.tensor([math.log1p(math.exp(v)) if v<=20 else v for v in vals],device=device).reshape(1,1,4)
    yield dict(name='softplus_nonlinear_and_linear_threshold',entrypoint='chunk_cumsum_fwd',
               kwargs=dict(dt=torch.tensor(vals,device=device).reshape(4,1),A=torch.tensor([-1.],device=device),
                           chunk_size=4,cu_chunk_seqlens=torch.tensor([0,4],dtype=torch.int32,device=device),dt_softplus=True),
               expected=(-transformed.cumsum(-1),transformed),atol=1e-3,rtol=1e-3)
