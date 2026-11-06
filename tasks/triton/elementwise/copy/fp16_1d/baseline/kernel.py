"""Identity copy of a 1D float16 tensor, a Triton kernel extracted from the Inductor output of
`output.copy_(input)`."""
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"XBLOCK": 128}, num_warps=2),
        triton.Config({"XBLOCK": 256}, num_warps=4),
        triton.Config({"XBLOCK": 512}, num_warps=4),
        triton.Config({"XBLOCK": 1024}, num_warps=8),
    ],
    key=["xnumel"],
)
@triton.jit
def _identity_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask).to(tl.float32)
    tl.store(out_ptr0 + xindex, tmp0, xmask)


def run(*, x, out):
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["XBLOCK"]), )
    _identity_kernel[grid](x, out, n)
