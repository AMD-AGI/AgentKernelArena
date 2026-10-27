# SPDX-License-Identifier: MIT
"""Initial implementation: the pinned upstream AITER public GEMM dispatch."""


def mlp_down(x, weight):
    """Return the BF16 rank-local partial product x @ weight.T, without bias."""
    from aiter.tuned_gemm import gemm_a16w16

    return gemm_a16w16(x, weight)
