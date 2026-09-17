"""Known answers supplement the original complete reference/AITER GPU checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model_control_support import load_model, verify

def run():
    import torch
    import math
    m=load_model()
    x=torch.tensor([[0.,1.,2.,3.]])
    model=m.Model(4,1,2,1,True,2.5);model.correction_bias.data.zero_();expected_w=torch.tensor([[2.5]])
    return [verify(model(x),(expected_w,torch.tensor([[3]],dtype=torch.int32)),"unique top expert with independently computed routing weight"), grouped_scalar_control()]


def grouped_scalar_control():
    """Two-stage group selection and unbiased weights from Python scalars."""
    import math
    import torch
    from reference_support import references
    m = load_model()
    x = torch.tensor([[-2., 1., -1., 2., 0., 3., -3., 4.],
                      [4., -1., 3., -2., 2., -3., 1., 0.]])
    bias = torch.tensor([-.1, .2, .05, .1, .15, -.2, .05, .25])
    known_w, known_ids = [], []
    for row in x.tolist():
        sigmoid = [1 / (1 + math.exp(-value)) for value in row]
        selection = [value + correction for value, correction in zip(sigmoid, bias.tolist())]
        # Four groups of two: their top-two sum is the sum of both members.
        groups = sorted(range(4), key=lambda g: sum(selection[2*g:2*g+2]), reverse=True)[:2]
        selected = sorted([i for g in groups for i in (2*g, 2*g+1)],
                          key=selection.__getitem__, reverse=True)[:2]
        total = sum(sigmoid[i] for i in selected)
        known_ids.append(selected)
        known_w.append([2.5 * sigmoid[i] / total for i in selected])
    expected_w = torch.tensor(known_w)
    expected_ids = torch.tensor(known_ids, dtype=torch.int32)
    actual_w, actual_ids, masked = m.grouped_route(x, bias, 2, True, 4, 2, 2.5)
    h = references(['_compare_routing'])
    mismatch, error = h._compare_routing(expected_w, expected_ids, actual_w, actual_ids, masked, 2)
    if mismatch or error > 1e-6:
        raise AssertionError('Grouped routing differs from independent scalar known answer')
    wrong_ids = actual_ids.clone()
    wrong_ids[:, 0] = 0
    if h._compare_routing(expected_w, expected_ids, actual_w, wrong_ids, masked, 2)[0] == 0:
        raise AssertionError('Routing comparator accepted wrong expert IDs')
    if h._compare_routing(expected_w, expected_ids, actual_w + 1, actual_ids, masked, 2)[1] <= h.REL_TOL:
        raise AssertionError('Routing comparator accepted wrong expert weights')
    return {'control': 'scalar group selection and unbiased normalized expert weights',
            'known_answer': 'PASS', 'negative_output': 'rejected',
            'comparator': 'task _compare_routing: wrong IDs and wrong weights'}
