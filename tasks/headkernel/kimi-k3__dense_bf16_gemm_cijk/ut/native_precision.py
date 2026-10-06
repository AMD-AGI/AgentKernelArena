"""Exact source-derived BF16 atomic proof for native-only calibration exceptions.

Every element is first checked against the unchanged FP32-matmul tolerance.
Only a failing native pair may use this path, and both BF16 lanes must exactly
match one common legal split arrival order. Candidate checks remain strict.
"""
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
PATTERN=re.compile(r'^flydsl_gemm\d+_abf16_wbf16_bf16_t\d+x\d+x(?P<tile_k>\d+)_split_k(?P<split_k>\d+)_block_m_warp\d+_block_n_warp\d+_block_k_warp(?P<k_warps>\d+)_async_copyTrue_b_to_ldsTrue_b_preshuffleFalse_c_to_ldsFalse_gfx950$')


def strict_mask(actual,expected):
    return (actual.float()-expected.float()).abs() > (0.02+0.01*expected.float().abs())


def spec_for(case):
    dispatch=case['capture_controls'].get('capture_native_dispatch',{})
    if dispatch.get('libtype')!='flydsl':raise AssertionError('Native result exceeds strict mathematical tolerance without a declared precision model')
    match=PATTERN.fullmatch(dispatch['config']['kernelName'])
    if match is None:raise AssertionError('Unknown native precision implementation')
    spec={k:int(v) for k,v in match.groupdict().items()}
    if not 1<=spec['split_k']<=8 or not 1<=spec['k_warps']<=4:raise AssertionError('Native precision model exceeds bounded source contract')
    return spec


@lru_cache(maxsize=1)
def verify_native_sources():
    from native_dispatch import load_native
    base=Path(load_native().__file__).resolve().parent
    pins=json.loads((ROOT/'provenance/NATIVE-PRECISION.json').read_text())['source_sha256']
    for name,digest in pins.items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=digest:raise ValueError('Native arithmetic source changed: '+name)
    return True


def bf16_partials(inputs,spec):
    import torch
    A=inputs['A'].float();W=inputs['B'].t().float()
    m,k=A.shape;split=spec['split_k'];warps=spec['k_warps'];tile=spec['tile_k']
    if k%split or (k//split)%tile or tile%warps:raise AssertionError('Invalid native K partition')
    extent=k//split;piece=tile//warps;parts=[]
    for block in range(split):
        slices=[]
        for warp in range(warps):
            if warps==1:
                lo=block*extent;hi=lo+extent
                partial=A[:,lo:hi]@W[:,lo:hi].t()
            else:
                indices=torch.tensor([block*extent+t*tile+warp*piece+j for t in range(extent//tile) for j in range(piece)],dtype=torch.int64)
                partial=A.index_select(1,indices)@W.index_select(1,indices).t()
            slices.append(partial.to(torch.bfloat16))
        merged=slices[0]
        for value in slices[1:]:merged=(merged.float()+value.float()).to(torch.bfloat16)
        parts.append(merged)
    return torch.stack(parts)


def exact_pair_orders(parts,observed,positions):
    import torch
    if observed.dtype!=torch.bfloat16 or observed.shape[1]%2:raise AssertionError('Packed atomic model requires BF16 output pairs')
    selected=parts.reshape(parts.shape[0],observed.shape[0],-1,2)[:,positions[:,0],positions[:,1],:].float()
    target=observed.reshape(observed.shape[0],-1,2)[positions[:,0],positions[:,1],:].contiguous().view(torch.int16)
    pending=torch.arange(len(positions));witnesses={};visited=0
    for order in itertools.permutations(range(parts.shape[0])):
        if pending.numel()==0:break
        accum=torch.zeros((pending.numel(),2),dtype=torch.bfloat16)
        for split in order:accum=(accum.float()+selected[split,pending]).to(torch.bfloat16)
        exact=(accum.contiguous().view(torch.int16)==target[pending]).all(dim=1)
        for index in pending[exact].tolist():witnesses[index]=list(order)
        pending=pending[~exact];visited+=1
    if pending.numel():
        failures=positions[pending].tolist()
        raise AssertionError('No exact legal BF16 atomic order for native output pairs: '+str(failures[:16]))
    return {'orders_visited':visited,'pairs_proved_exactly':len(positions),
            'witnesses':[{'row':int(positions[i,0]),'pair_column':int(positions[i,1]),'arrival_order':witnesses[i]} for i in range(len(positions))]}


def calibrate(case,inputs,actual,expected,*,check_sources=True):
    import torch
    if not bool(torch.isfinite(actual.float()).all()) or not bool(torch.isfinite(expected.float()).all()):raise AssertionError('Nonfinite native output or reference')
    bad=strict_mask(actual,expected)
    record={'case_id':case['case_id'],'elements_checked':actual.numel(),'strict_failure_elements':int(bad.sum()),'rtol':0.01,'atol':0.02,'exact_exception_pairs':0}
    if not bool(bad.any()):return record
    if check_sources:verify_native_sources()
    spec=spec_for(case)
    positions=bad.reshape(bad.shape[0],-1,2).any(dim=2).nonzero()
    proof=exact_pair_orders(bf16_partials(inputs,spec),actual,positions)
    record.update(exact_exception_pairs=len(positions),native_precision=spec,proof=proof)
    return record
