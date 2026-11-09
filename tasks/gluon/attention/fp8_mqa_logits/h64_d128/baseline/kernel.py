"""FP8 MQA logits for the DeepSeek-V3.2 indexer: AITER's gfx950 Gluon kernel behind the run() interface.

Source: aiter/ops/triton/attention/fp8_mqa_logits.py and
aiter/ops/triton/_gluon_kernels/gfx950/attention/fp8_mqa_logits.py (ROCm/aiter, MIT, commit
652f2d4b768239aefd515c929e84817681ff598d; the upstream files carry no per-file header, so the
repository LICENSE in LICENSE applies). The gfx950 branch of the upstream wrapper is kept (block 32,
one wave, three waves per EU, two buffers, folded head reduction when available). The Gluon kernel
writes only the in-window columns of each row; production pre-fills the output with
torch.full(-inf) before the launch, which run() does in place with logits.fill_(-inf) on the
preallocated output.
"""

from .gluon_kernel import _gluon_fp8_mqa_logits_kernel

try:
    from triton.language.core import _unwrap_iterable, constexpr
except ImportError:
    _unwrap_iterable = None
    constexpr = None


# Upstream probes two newer Triton features. The first, _async_copy_accepts_distributed_layout(), reads
# the source of gl.amd.cdna4.async_copy.global_load_to_shared with inspect and returns True when it
# mentions "DistributedLayout"; inspect is not available to implementations, so the result is fixed here.
# In triton-lang/triton at the release/3.8.x branch,
# https://raw.githubusercontent.com/triton-lang/triton/release/3.8.x/python/triton/experimental/gluon/language/amd/cdna4/async_copy.py
# global_load_to_shared checks isinstance(ptr.type.layout, DistributedLayout), so the probe returns True.
# (Read from the branch; the exact 3.8.0+git4cff872c commit of the image was not compared.)
ASYNC_COPY_SUPPORTS_DISTRIBUTED = True


def _permute_accepts_constexpr_tuple() -> bool:
    """
    True iff Triton's _unwrap_iterable unwraps an inner constexpr.

    On versions before PR #9751 (commit 0688e7736a), passing a constexpr-wrapped
    tuple as the sole arg to permute/trans/reshape leaves the constexpr wrapped,
    causing `len(constexpr)` to fail in semantic.permute. After #9751, it gets
    unwrapped to a raw tuple of ints.
    """
    if _unwrap_iterable is None:
        return False
    probe = constexpr((0, 1, 2))
    result = _unwrap_iterable((probe,))
    return not isinstance(result, constexpr)


FOLDED_REDUCTED_SUPPORT = _permute_accepts_constexpr_tuple()

def _launch(q, kv, kv_scales, weights, cu_seqlen_ks, cu_seqlen_ke, logits, use_padded_layout):
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = kv.shape[0]
    stride_q_s, stride_q_h, stride_q_d = q.stride()
    stride_kv_s, stride_kv_d = kv.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    # gfx950 branch of the upstream wrapper.
    use_folded_reduction = FOLDED_REDUCTED_SUPPORT and num_heads > 16
    num_buffers = 2
    waves_per_eu = 3
    num_chains = 4 if use_folded_reduction else 0
    num_warps = 1
    block_kv = 32

    # Buffer ops use a 32-bit byte offset (2 GiB resource descriptor cap).
    # Fall back to plain global load/store when a tensor exceeds that.
    buffer_limit_bytes = 2 * 1024 * 1024 * 1024
    use_buffer_load = kv.numel() * kv.element_size() < buffer_limit_bytes
    use_buffer_store = logits.numel() * logits.element_size() < buffer_limit_bytes
    _gluon_fp8_mqa_logits_kernel[(seq_len,)](
        Q_ptr=q,
        KV_ptr=kv,
        kv_scales_ptr=kv_scales,
        weights_ptr=weights,
        cu_start_ptr=cu_seqlen_ks,
        cu_end_ptr=cu_seqlen_ke,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=stride_q_s,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s,
        stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s,
        stride_logits_k=stride_logits_k,
        BLOCK_KV=block_kv,
        NUM_WARPS=num_warps,
        NUM_BUFFERS=num_buffers,
        NUM_CHAINS=num_chains,
        USE_BUFFER_LOAD=use_buffer_load,
        USE_BUFFER_STORE=use_buffer_store,
        USE_PADDED_SHARED_LAYOUT=use_padded_layout,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )


def run(*, q, kv, kv_scales, weights, cu_seqlen_ks, cu_seqlen_ke, logits):
    seq_len, num_heads, head_size = q.shape
    # TODO upstream: assumes num_heads and head_size are powers of 2.
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."
    # Production allocates the logits with torch.full(-inf) because of the window masking.
    logits.fill_(float("-inf"))
    _launch(q, kv, kv_scales, weights, cu_seqlen_ks, cu_seqlen_ke, logits, ASYNC_COPY_SUPPORTS_DISTRIBUTED)
