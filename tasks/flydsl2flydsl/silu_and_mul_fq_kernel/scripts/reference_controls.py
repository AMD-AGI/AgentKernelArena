"""Independent CPU known answers and deliberately wrong-output controls.
These controls supplement, never replace, the full GPU case suite.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_support import references, control, close

def run():
    import torch
    import math
    h=references(["_torch_ref_silu_mul","reference_mxfp4","_decode_e2m1"])
    x=torch.tensor([[0.,1.,-1.,2., 3.,2.,4.,5.]])
    expected=torch.tensor([[0.,2./(1+math.exp(-1)),-4./(1+math.exp(1)),10./(1+math.exp(-2))]])
    out=[control(h._torch_ref_silu_mul(x,4),expected,close(1e-6,1e-6),"scalar sigmoid times gate and up")]
    codes=torch.arange(16,dtype=torch.uint8)
    expected_grid=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
    out.append(control(h._decode_e2m1(codes),expected_grid,close(0,0),"all signed E2M1 codes"))
    exact=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,-4.]*4).reshape(1,32)
    deq,scales=h.reference_mxfp4(exact,1)
    out.append(control(deq,exact,close(0,0),"representable MXFP4 unit-scale block"))
    if scales.tolist()!=[[127]]: raise AssertionError("Wrong E8M0 unit-scale exponent")
    from scripts.routing_checks import routing_reference
    routing_inputs = dict(rows=4, token_num=2, num_sorted_rows=4, inter_dim=32, scale_cols=1,
                          x=torch.zeros(4, 64),
                          sorted_ids=torch.tensor([0x1000001, 0, 0x1000000, 1], dtype=torch.int32),
                          num_valid_ids=torch.tensor([4], dtype=torch.int32))
    # Four independently known unit-grid blocks scaled by 1,2,4,8. The packed
    # IDs select input rows 3,0,1,2; scales must follow those sorted positions.
    known_real = exact * torch.tensor([1., 2., 4., 8.])[:, None]
    _, _, routed_deq, routed_scales = routing_reference(
        routing_inputs, lambda x, width: known_real, h.reference_mxfp4)
    out.append(control(torch.tensor(routed_scales), torch.tensor([[130], [127], [128], [129]]),
                       close(0, 0), "packed token/slot IDs select sorted E8M0 coordinates"))
    out.append(control(routed_deq, torch.cat([exact*8, exact, exact*2, exact*4]), close(0, 0),
                       "input-row MXFP4 payload paired with sorted scales"))
    return out
