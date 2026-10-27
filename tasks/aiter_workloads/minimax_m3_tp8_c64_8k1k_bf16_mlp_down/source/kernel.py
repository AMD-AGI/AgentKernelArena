# SPDX-License-Identifier: MIT
"""Initial implementation: the pinned upstream AITER public GEMM dispatch."""


def mlp_down(x, weight):
    """Return the BF16 rank-local partial product x @ weight.T, without bias."""
    from aiter.tuned_gemm import gemm_a16w16

    # Contiguous views need not start at an aligned address. The selected
    # upstream FlyDSL kernel requires 16-byte alignment; copy only when needed,
    # inside this call so graph capture and timing include the preparation.
    if x.data_ptr() % 16:
        x = x.clone()
    if weight.data_ptr() % 16:
        weight = weight.clone()
    return gemm_a16w16(x, weight)
