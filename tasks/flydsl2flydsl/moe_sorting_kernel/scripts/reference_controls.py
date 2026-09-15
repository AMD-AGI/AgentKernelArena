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
    h=references(["moe_sorting_reference","check_sorted_ids","check_expert_ids"])
    ids=torch.tensor([[1],[0]],dtype=torch.int32);weights=torch.tensor([[.25],[.75]])
    actual=h.moe_sorting_reference(ids,weights,2,unit_size=2)
    sentinel=(1<<24)|2
    expected_ids=torch.tensor([1,sentinel,0,sentinel],dtype=torch.int32)
    if not torch.equal(actual[0][:4],expected_ids) or not torch.equal(actual[3],torch.tensor([4,2],dtype=torch.int32)):
        raise AssertionError("Hand-computed packing/padding known answer failed")
    if actual[1][0]!=.75 or actual[1][2]!=.25: raise AssertionError("Routing weight known answer failed")
    if not h.check_sorted_ids(expected_ids,actual[0],4,1,2): raise AssertionError("Valid packed IDs rejected")
    bad=expected_ids.clone();bad[0]=99
    if h.check_sorted_ids(expected_ids,bad,4,1,2): raise AssertionError("Invalid packed ID accepted")
    if not h.check_expert_ids(torch.tensor([0,1]),actual[2],num_valid_blocks=2): raise AssertionError("Valid expert IDs rejected")
    if h.check_expert_ids(torch.tensor([0,1]),torch.tensor([1,0]),num_valid_blocks=2): raise AssertionError("Wrong expert IDs accepted")
    return [{"control":"hand-computed expert-major IDs, weights and padding","known_answer":"PASS","negative_output":"rejected"}]
