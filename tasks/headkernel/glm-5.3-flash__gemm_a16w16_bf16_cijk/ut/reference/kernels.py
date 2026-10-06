import triton
import triton.language as tl


@triton.jit
def gemm_kernel(A, B, SA, SB, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                FP8: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    if A.dtype.element_ty == tl.float32:
        # Keep each128-term IEEE FP32 dot separate; sum its partials in FP64.
        acc = tl.zeros((BM, BN), tl.float64)
    else:
        acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, tl.cdiv(K, BK)):
        kk = start * BK + ks
        aa = tl.load(A + rows[:, None] * K + kk[None, :], (rows[:, None] < M) & (kk[None, :] < K), 0)
        if FP8:
            # Invert the actual AITER (16,16) FP8 physical weight permutation.
            off = (((cols[None, :] // 16 * (K // 32) + kk[:, None] // 32) * 2 + (kk[:, None] % 32) // 16) * 16 + cols[None, :] % 16) * 16 + kk[:, None] % 16
            bb = tl.load(B + off, (cols[None, :] < N) & (kk[:, None] < K), 0)
            delta = tl.dot(aa, bb)
            sa = tl.load(SA + rows + start * M, rows < M, 0)
            sb = tl.load(SB + (cols // 128) * (K // 128) + start, cols < N, 0)
            acc += delta * sa[:, None] * sb[None, :]
        else:
            bb = tl.load(B + kk[:, None] + cols[None, :] * K, (cols[None, :] < N) & (kk[:, None] < K), 0)
            if A.dtype.element_ty == tl.float32:
                acc += tl.dot(aa, bb, allow_tf32=False).to(tl.float64)
            else:
                acc += tl.dot(aa, bb, allow_tf32=False)
    tl.store(C + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))
