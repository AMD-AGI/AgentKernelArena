# Newly authored whole-MoE replacement port, not the profiled AITER code object.
import triton
import triton.language as tl


@triton.jit
def quantize_input(H, Q, S, M: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * 128 + tl.arange(0, 128)
    values = tl.load(H + row * K + cols).to(tl.float32)
    maximum = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(maximum, 1.0e-10) * 0.0022321429569274187
    quantized = tl.minimum(tl.maximum(values * tl.div_rn(1.0, scale), -448.0), 448.0)
    tl.store(Q + row * K + cols, quantized)
    tl.store(S + row * (K // 128) + block, scale)


@triton.jit
def stage1(Q, S, W, WS, IDS, EXPERTS, VALID, G,
           M: tl.constexpr, H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    block = tl.program_id(0)
    column_block = tl.program_id(1)
    count = tl.load(VALID)
    if block * BM < count:
        expert = tl.load(EXPERTS + block)
        positions = block * BM + tl.arange(0, BM)
        packed = tl.load(IDS + positions, positions < count, 0)
        rows = packed & 0xFFFFFF
        slots = (packed >> 24) & 255
        route = rows * TOPK + slots
        cols = column_block * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for start in range(0, H // BK):
            kk = start * BK + ks
            a = tl.load(Q + rows[:, None] * H + kk[None, :], (rows[:, None] < M) & (positions[:, None] < count), 0.0)
            offset = (((cols[None, :] // 16 * (H // 32) + kk[:, None] // 32) * 2 + (kk[:, None] % 32) // 16) * 16 + cols[None, :] % 16) * 16 + kk[:, None] % 16
            b = tl.load(W + expert * (2 * I * H) + offset, cols[None, :] < 2 * I, 0.0)
            delta = tl.dot(a, b)
            sa = tl.load(S + rows * (H // 128) + start, rows < M, 0)
            sb = tl.load(WS + expert * (2 * I // 128) * (H // 128) + (cols // 128) * (H // 128) + start, cols < 2 * I, 0)
            acc += delta * sa[:, None] * sb[None, :]
        tl.store(G + route[:, None] * (2 * I) + cols[None, :], acc,
                 (rows[:, None] < M) & (slots[:, None] < TOPK) & (positions[:, None] < count) & (cols[None, :] < 2 * I))


@triton.jit
def activate_quantize(G, Q, S, M: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr):
    route = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * 128 + tl.arange(0, 128)
    gate = tl.load(G + route * (2 * I) + cols)
    up = tl.load(G + route * (2 * I) + I + cols)
    activated = gate / (1.0 + tl.exp(-gate)) * up
    maximum = tl.max(tl.abs(activated), axis=0)
    scale = tl.where(maximum == 0, 1.0, maximum / 448.0)
    quantized = tl.minimum(tl.maximum(activated / scale, -448.0), 448.0)
    tl.store(Q + route * I + cols, quantized)
    tl.store(S + route * (I // 128) + block, scale)


@triton.jit
def stage2(Q, S, W, WS, IDS, EXPERTS, VALID, P,
           M: tl.constexpr, H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    block = tl.program_id(0)
    column_block = tl.program_id(1)
    count = tl.load(VALID)
    if block * BM < count:
        expert = tl.load(EXPERTS + block)
        positions = block * BM + tl.arange(0, BM)
        packed = tl.load(IDS + positions, positions < count, 0)
        rows = packed & 0xFFFFFF
        slots = (packed >> 24) & 255
        route = rows * TOPK + slots
        cols = column_block * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for start in range(0, I // BK):
            kk = start * BK + ks
            a = tl.load(Q + route[:, None] * I + kk[None, :], (rows[:, None] < M) & (slots[:, None] < TOPK) & (positions[:, None] < count), 0.0)
            offset = (((cols[None, :] // 16 * (I // 32) + kk[:, None] // 32) * 2 + (kk[:, None] % 32) // 16) * 16 + cols[None, :] % 16) * 16 + kk[:, None] % 16
            b = tl.load(W + expert * H * I + offset, cols[None, :] < H, 0.0)
            delta = tl.dot(a, b)
            sa = tl.load(S + route * (I // 128) + start, (rows < M) & (slots < TOPK), 0)
            sb = tl.load(WS + expert * (H // 128) * (I // 128) + (cols // 128) * (I // 128) + start, cols < H, 0)
            acc += delta * sa[:, None] * sb[None, :]
        tl.store(P + route[:, None] * H + cols[None, :], acc,
                 (rows[:, None] < M) & (slots[:, None] < TOPK) & (positions[:, None] < count) & (cols[None, :] < H))


@triton.jit
def reduce_routes(P, ROUTE_WEIGHTS, O, M: tl.constexpr, H: tl.constexpr, TOPK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * 256 + tl.arange(0, 256)
    total = tl.zeros((256,), tl.float32)
    for slot in range(0, TOPK):
        value = tl.load(P + (row * TOPK + slot) * H + cols, cols < H, 0)
        weight = tl.load(ROUTE_WEIGHTS + row * TOPK + slot)
        total += value * weight
    tl.store(O + row * H + cols, total, cols < H)
