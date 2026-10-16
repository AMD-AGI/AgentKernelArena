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
    r = references(['reference_cumsum'], {"F": F, "SQRT2": math.sqrt(2)})
    x=t([1.,2.,3.,4.,5.]).reshape(1,5,1)
    for reverse,expected in [(False,[2.,6.,6.,14.,10.]),(True,[6.,4.,14.,8.,10.])]:
        check(r.reference_cumsum(x,{"reverse":reverse,"scale":2.},BT=2),t(expected).reshape_as(x),"chunk reset and direction "+str(reverse))
    # Head-first/varlen control oracle: hand-computed answers with chunk size 2.
    # Sequences [1], [2,3,4], [5,6] restart chunks at every sequence start.
    from semantic_controls import chunk_index_pairs, reference, run_controls
    from replay_checks import require_tensor_contract, require_unchanged
    v=t([1.,2.,3.,4.,5.,6.]).reshape(1,6,1)
    check(reference(v,chunk_size=2,offsets=[0,1,4,6]).float(),t([1.,2.,5.,4.,5.,11.]).reshape_as(v),"varlen chunk reset at sequence starts")
    check(reference(v,chunk_size=2,reverse=True,scale=2.,offsets=[0,1,4,6]).float(),t([2.,10.,6.,8.,22.,12.]).reshape_as(v),"varlen suffix and scale")
    check(t(chunk_index_pairs([0,1,4,6],2)),t([[0,0],[1,0],[1,1],[2,0]]),"varlen chunk index pairs")
    hf=t([[1.,2.,3.],[10.,20.,30.]]).reshape(1,2,3)
    check(reference(hf,chunk_size=2,head_first=True).float(),t([[1.,3.,3.],[10.,30.,30.]]).reshape_as(hf),"head-first time axis")
    # Cross-check every actual control draw against the protected FP32 harness
    # oracle applied per sequence after an explicit [B,H,T]->[B,T,H] transpose.
    h = references(['reference_cumsum','_compare_sglang_output'], {"require_tensor_contract": require_tensor_contract})
    def operator(g, chunk_size, reverse, scale, head_first=False, cu_seqlens=None, chunk_indices=None):
        y = g.transpose(1, 2) if head_first else g
        bounds = [0, y.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
        cfg = {"reverse": reverse, "scale": scale}
        out = torch.cat([h.reference_cumsum(y[:, a:b], cfg, BT=chunk_size) for a, b in zip(bounds[:-1], bounds[1:])], 1)
        return (out.transpose(1, 2) if head_first else out).contiguous()
    rows.extend(run_controls(operator, h._compare_sglang_output, require_unchanged, device="cpu"))
    return rows
