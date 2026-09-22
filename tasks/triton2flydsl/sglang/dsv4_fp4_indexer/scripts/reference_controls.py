"""Independent CPU known answers for protected reference functions.

These additional small controls do not replace or resize any GPU workload or
change its numerical gate. Full reference/candidate evaluation remains in the
original harness. A deliberately wrong output must fail each control comparator.
"""
import math
import torch
import torch.nn.functional as F
from reference_support import references, control, close


def run():
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    rows = []
    def check(actual, expected, label):
        rows.append(control(actual, expected, close(1e-5, 1e-5), label))
    r = references(['reference', '_store_reference'], {"F": F, "SQRT2": math.sqrt(2)})
    # max=6 => scale 1; E2M1 +6/-6 codes are 7/15, low nibble first.
    x=t([[6.,-6.]*64]);q,s=r.reference(x)
    check(q.float(),torch.full((1,64),-9.),"packed E2M1 nibbles 0xf7")
    rows.append(control(s.to(torch.float64),torch.tensor([0x7f7f7f7f],dtype=torch.float64),lambda a,b: torch.equal(a,b),"four unit E8M0 scale bytes (exact integer comparison)"))
    cache=r._store_reference(x,q,s,torch.tensor([1]),2,1)
    e=torch.zeros(1,136,dtype=torch.uint8);e[0,64:128]=247;e[0,132:136]=127
    check(cache.float(),e.float(),"FP4 cache payload and separate scale plane")
    return rows
