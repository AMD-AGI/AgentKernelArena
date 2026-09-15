"""Independent CPU known answers for protected reference functions.

These additional small controls do not replace or resize any GPU workload or
change its numerical gate. Full reference/candidate evaluation remains in the
original harness. A deliberately wrong output must fail each control comparator.
"""
import math
import torch
import torch.nn.functional as F
from reference_support import references, control, close
from scripts.replay_checks import require_unchanged


def run():
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    rows = []
    def check(actual, expected, label):
        rows.append(control(actual, expected, close(1e-5, 1e-5), label))
    r = references(['_compare_recurrent_output', 'reference_decode'], {"F": F, "SQRT2": math.sqrt(2), "require_unchanged": require_unchanged})
    # Normalized q=k=1/sqrt(1+1e-6); g=-log(2), beta=1/2.
    u=1/math.sqrt(1.000001)
    state=2.+.5*(6.-2.*u)*u
    inp=dict(B=1,H=1,HV=1,K=1,V=1,scale=1.,mixed_qkv=t([[1.,1.,6.]]),a=t([[0.]]),b=t([[0.]]),A_log=t([0.]),dt_bias=t([0.]),cache_indices=torch.tensor([0]),ssm_states=t([4.]).reshape(1,1,1,1))
    o,s=r.reference_decode(inp)
    check(s,t([state]).reshape_as(s),"scalar decay, delta rule, state update")
    check(o,t([state*u]).reshape_as(o),"normalized query reads updated state")
    expected=(t([state*u]).reshape_as(o),t([state]).reshape_as(s))
    for slot in range(2):
        def accept(a,b,slot=slot):
            aa=list(expected);bb=list(expected);aa[slot]=a;bb[slot]=b
            try: r._compare_recurrent_output(tuple(aa),tuple(bb),inp); return True
            except AssertionError: return False
        rows.append(control((o,s)[slot],expected[slot],accept,
                            f"actual decode recurrence comparator output{slot}"))
    return rows
