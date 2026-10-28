# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Protected provided FP8/BF16 MoE; independent PyTorch oracle is model.py.

All quantization, routing preparation and allocations are performed on the
current raw operands inside every invocation. No operand or answer is cached.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def quantize_rows(X, Q, S, ROWS: tl.constexpr, COLS: tl.constexpr,
                  FP8_MAX: tl.constexpr, B: tl.constexpr, RM: tl.constexpr):
    row = tl.program_id(0) * RM + tl.arange(0, RM)
    col = tl.arange(0, B)
    x = tl.load(X + row[:, None] * COLS + col[None, :],
                (row[:, None] < ROWS) & (col[None, :] < COLS), other=0.).to(tl.float32)
    maximum = tl.max(tl.abs(x), 1)
    scale = tl.where(maximum > 0, maximum * (1.0 / FP8_MAX), 1.)
    q = tl.div_rn(x, scale[:, None]).to(Q.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(Q + row[:, None] * COLS + col[None, :], q,
             (row[:, None] < ROWS) & (col[None, :] < COLS))
    tl.store(S + row, scale, row < ROWS)


@triton.jit
def group_routes(IDS, ROUTES, COUNTS, R: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    route = tl.arange(0, BLOCK)
    assigned = tl.load(IDS + route, route < R, other=-1)
    belongs = (route < R) & (assigned == expert)
    ordered = tl.sort(tl.where(belongs, route, R), descending=False)
    tl.store(ROUTES + expert * R + route, ordered, route < R)
    tl.store(COUNTS + expert, tl.sum(belongs.to(tl.int32), 0))


@triton.jit
def stage1(A, SA, W, SW, ROUTES, COUNTS, WEIGHTS, INTER,
           M: tl.constexpr, D: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GELU: tl.constexpr):
    tile = tl.program_id(0)
    expert = tl.program_id(1)
    count = tl.load(COUNTS + expert)
    columns = tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    for start in range(0, count, BM):
        position = start + tl.arange(0, BM)
        route = tl.load(ROUTES + expert * (M * TOPK) + position, position < count, other=M*TOPK)
        token = route // TOPK
        valid = (position < count) & (route < M * TOPK)
        g = tl.full((BM, BN), 0., tl.float32)
        u = tl.full((BM, BN), 0., tl.float32)
        for base in range(0, D, BK):
            kk = base + inner
            a = tl.load(A + token[:, None] * D + kk[None, :], valid[:, None] & (kk[None, :] < D), other=0.)
            gate = tl.load(W + (expert * (2*I) + columns[None, :]) * D + kk[:, None],
                           (columns[None, :] < I) & (kk[:, None] < D), other=0.)
            up = tl.load(W + (expert * (2*I) + I + columns[None, :]) * D + kk[:, None],
                         (columns[None, :] < I) & (kk[:, None] < D), other=0.)
            g = tl.dot(a, gate, g)
            u = tl.dot(a, up, u)
        s = tl.load(SA + token, valid, other=0.)
        sg = tl.load(SW + expert*(2*I) + columns, columns < I, other=0.)
        su = tl.load(SW + expert*(2*I) + I + columns, columns < I, other=0.)
        routing = tl.load(WEIGHTS + route, valid, other=0.)
        g = g * s[:, None] * sg[None, :] * routing[:, None]
        u = u * s[:, None] * su[None, :] * routing[:, None]
        if GELU:
            activated = .5 * g * (1. + libdevice.erf(g * 0.7071067811865476))
        else:
            activated = g / (1. + tl.exp(-g))
        tl.store(INTER + route[:, None] * I + columns[None, :], (activated * u).to(tl.bfloat16),
                 valid[:, None] & (columns[None, :] < I))


@triton.jit
def stage2(INTER, W, SW, ROUTES, COUNTS, PARTIAL,
           M: tl.constexpr, D: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    expert = tl.program_id(1)
    count = tl.load(COUNTS + expert)
    columns = tile * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    for start in range(0, count, BM):
        position = start + tl.arange(0, BM)
        route = tl.load(ROUTES + expert*(M*TOPK) + position, position < count, other=M*TOPK)
        valid = (position < count) & (route < M*TOPK)
        acc = tl.full((BM, BN), 0., tl.float32)
        for base in range(0, I, BK):
            kk = base + inner
            a = tl.load(INTER + route[:, None] * I + kk[None, :], valid[:, None] & (kk[None, :] < I), other=0.)
            b = tl.load(W + (expert * D + columns[None, :]) * I + kk[:, None],
                        (columns[None, :] < D) & (kk[:, None] < I), other=0.).to(tl.bfloat16)
            acc = tl.dot(a, b, acc)
        scale = tl.load(SW + expert*D + columns, columns < D, other=0.)
        tl.store(PARTIAL + route[:, None]*D + columns[None, :], acc * scale[None, :],
                 valid[:, None] & (columns[None, :] < D))


@triton.jit
def combine(PARTIAL, OUT, M: tl.constexpr, D: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    token, column = offset//D, offset%D
    total = tl.full((BLOCK,),0.,tl.float32)
    for route in range(TOPK):
        total += tl.load(PARTIAL + (token*TOPK + route)*D + column, offset < M*D, other=0.)
    tl.store(OUT + offset,total,offset < M*D)


def quantize(value, dtype):
    value = value.contiguous()
    cols=value.shape[-1];rows=value.numel()//cols
    codes=torch.empty_like(value,dtype=dtype)
    scales=torch.empty((rows,),dtype=torch.float32,device=value.device)
    quantize_rows[(triton.cdiv(rows,4),)](value,codes,scales,rows,cols,float(torch.finfo(dtype).max),
                                        triton.next_power_of_2(cols),4,num_warps=8)
    return codes,scales


def fp8_moe(hidden,w1,w2,weights,ids,fp8_dtype,activation='silu'):
    m,d=hidden.shape;experts,twice_i,_=w1.shape;i=twice_i//2;topk=ids.shape[1];r=m*topk
    weights=weights.contiguous();ids=ids.contiguous()
    a,sa=quantize(hidden,fp8_dtype);q1,s1=quantize(w1,fp8_dtype);q2,s2=quantize(w2,fp8_dtype)
    routes=torch.empty((experts,r),dtype=torch.int32,device=hidden.device)
    counts=torch.empty((experts,),dtype=torch.int32,device=hidden.device)
    intermediate=torch.empty((r,i),dtype=torch.bfloat16,device=hidden.device)
    partial=torch.empty((r,d),dtype=torch.float32,device=hidden.device)
    output=torch.empty((m,d),dtype=hidden.dtype,device=hidden.device)
    group_routes[(experts,)](ids,routes,counts,r,triton.next_power_of_2(r))
    stage1[(triton.cdiv(i,64),experts)](a,sa,q1,s1,routes,counts,weights,intermediate,
                                      m,d,i,topk,16,64,64,activation=='gelu',num_warps=4)
    stage2[(triton.cdiv(d,64),experts)](intermediate,q2,s2,routes,counts,partial,
                                      m,d,i,topk,16,64,64,num_warps=4)
    combine[(triton.cdiv(m*d,256),)](partial,output,m,d,topk,256)
    return output


def fp8_dtype(hidden):
    arch = torch.cuda.get_device_properties(hidden.device).gcnArchName.split(":")[0]
    return torch.float8_e4m3fnuz if arch == "gfx942" else torch.float8_e4m3fn
