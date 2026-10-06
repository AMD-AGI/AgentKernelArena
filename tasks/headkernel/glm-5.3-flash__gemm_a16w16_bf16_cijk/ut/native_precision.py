"""Source-bound, exact native ASM calibration; never a candidate tolerance.

The pinned ASM rounds each256-term FP32 partial with bits+0x8000, then
atomically adds packed BF16 pairs. A failing pair must equal one explicit
legal order bit for bit. Exhausting the bounded search rejects the result.
"""
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import random
ROOT=Path(__file__).resolve().parents[1]
KERNEL='_ZN5aiter39bf16gemm_fp32bf16_tn_64x64_splitk_cleanE'


def candidate_error_metrics(actual,expected):
    """CPU FP64 Frobenius error, without an absolute floor tied to output units."""
    import math
    import torch
    # BF16 has7 fraction bits and unit roundoff2^-8. Two independently
    # rounded copies of the same FP32 value differ by at most2u/(1-u)
    # relative to the rounded reference. Use that output-precision scale
    # as the BF16 global accuracy requirement; the probe also measures
    # the submitted FP32 reduction error against this requirement.
    if expected.dtype==torch.bfloat16:
        unit_roundoff=2.0**-8
        limit=2*unit_roundoff/(1-unit_roundoff)
    elif expected.dtype==torch.float32:
        limit=1.3e-6  # Existing PyTorch float32 relative-accuracy requirement.
    else:raise AssertionError('Unexpected dense output dtype')
    reference_norm=float(torch.linalg.vector_norm(expected.double()))
    error_norm=float(torch.linalg.vector_norm(actual.double()-expected.double()))
    if not math.isfinite(reference_norm) or not math.isfinite(error_norm):raise AssertionError('Nonfinite candidate/reference norm')
    ratio=error_norm/reference_norm if reference_norm else (0.0 if error_norm==0.0 else None)
    return {'reference_l2':reference_norm,'error_l2':error_norm,'normalized_l2':ratio,
        'normalized_l2_limit':limit,'scale_relative_pass':error_norm<=limit*reference_norm}


def candidate_close(actual,expected):
    import torch
    if actual.dtype==torch.float32:
        torch.testing.assert_close(actual,expected,rtol=1.3e-6,atol=1e-5)
    else:torch.testing.assert_close(actual,expected,rtol=0.01,atol=0.02)
    metrics=candidate_error_metrics(actual,expected)
    if not metrics['scale_relative_pass']:
        raise AssertionError('Scale-relative candidate error exceeds dtype accuracy requirement: '+str(metrics))
    return metrics


def strict_mask(actual,expected):
    return (actual.float()-expected.float()).abs()>(0.02+0.01*expected.float().abs())


@lru_cache(maxsize=1)
def verify_native_sources():
    native=importlib.import_module('aiter.tuned_gemm');base=Path(native.__file__).resolve().parents[1]
    pins=json.loads((ROOT/'provenance/NATIVE-PRECISION.json').read_text())['source_sha256']
    for name,digest in pins.items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=digest:raise ValueError('Native arithmetic source changed: '+name)
    return True


def dispatch_for(inputs):
    native=importlib.import_module('aiter.tuned_gemm');A=inputs['A'];B=inputs['B']
    cfg=native.get_GEMM_A16W16_config(M=A.shape[0],N=B.shape[1],K=A.shape[1],bias=False,dtype=str(A.dtype),otype=str(A.dtype),scaleAB=False,bpreshuffle=False)
    return {name:cfg.get(name) for name in ('libtype','kernelName','splitK')}


def round_asm_partial(value):
    import torch
    return ((value.contiguous().view(torch.int32)+0x8000)>>16).to(torch.int16).view(torch.bfloat16)


def asm_partials(inputs):
    import torch
    A=inputs['A'].float();B=inputs['B'].float()
    if A.shape[1]!=4096 or B.shape[0]!=4096:raise AssertionError('Unknown ASM K partition')
    return torch.stack([round_asm_partial(A[:,start:start+256]@B[start:start+256]) for start in range(0,4096,256)])


def exact_pair_orders(parts,observed,positions,*,max_orders=65536):
    import torch
    if parts.shape[0]!=16 or observed.dtype!=torch.bfloat16 or observed.shape[1]%2:raise AssertionError('Expected16 native packed BF16 partials')
    selected=parts.reshape(16,observed.shape[0],-1,2)[:,positions[:,0],positions[:,1]].float()
    target=observed.reshape(observed.shape[0],-1,2)[positions[:,0],positions[:,1]].contiguous().view(torch.int16)
    pending=torch.arange(len(positions));witnesses={};rng=random.Random(0);visited=0
    for _ in range(max_orders):
        if not pending.numel():break
        order=rng.sample(range(16),16);acc=torch.zeros((len(pending),2),dtype=torch.bfloat16)
        for split in order:acc=(acc.float()+selected[split,pending]).to(torch.bfloat16)
        exact=(acc.contiguous().view(torch.int16)==target[pending]).all(dim=1)
        for index in pending[exact].tolist():witnesses[index]=order
        pending=pending[~exact];visited+=1
    if pending.numel():raise AssertionError('No exact legal ASM atomic order proved for native pairs: '+str(positions[pending].tolist()[:16]))
    return {'orders_examined':visited,'pairs_proved_exactly':len(positions),
        'witnesses':[{'row':int(positions[i,0]),'pair_column':int(positions[i,1]),'arrival_order':witnesses[i]} for i in range(len(positions))]}


def calibrate(case,inputs,actual,expected,*,dispatch=None,check_sources=True):
    import torch
    if actual.device.type!='cpu' or expected.device.type!='cpu' or any(x.device.type!='cpu' for x in inputs.values()):raise AssertionError('Native proof and inputs must stay on CPU')
    if not bool(torch.isfinite(actual.float()).all()) or not bool(torch.isfinite(expected.float()).all()):raise AssertionError('Nonfinite native output/reference')
    if actual.dtype!=expected.dtype or actual.shape!=expected.shape:raise AssertionError('Native output ABI differs')
    bad=strict_mask(actual,expected)
    record={'case_id':case['case_id'],'elements_checked':actual.numel(),'strict_failure_elements':int(bad.sum()),'rtol':0.01,'atol':0.02,'exact_exception_pairs':0}
    if not bool(bad.any()):return record
    family=case['live_fixture']['capture_family'];shape=(inputs['A'].shape[0],inputs['B'].shape[1],inputs['A'].shape[1])
    if family!='bf16_gemm' or actual.dtype!=torch.bfloat16 or shape!=(64,128,4096):raise AssertionError('Native result exceeds mathematical tolerance without a declared arithmetic model')
    if check_sources:verify_native_sources()
    dispatch=dispatch_for(inputs) if dispatch is None else dispatch
    if dispatch.get('libtype')!='asm' or dispatch.get('kernelName')!=KERNEL or dispatch.get('splitK')!=16:raise AssertionError('Unsupported native precision dispatch')
    positions=bad.reshape(bad.shape[0],-1,2).any(dim=2).nonzero()
    proof=exact_pair_orders(asm_partials(inputs),actual,positions)
    record.update(exact_exception_pairs=len(positions),native_dispatch=dispatch,proof=proof)
    return record
