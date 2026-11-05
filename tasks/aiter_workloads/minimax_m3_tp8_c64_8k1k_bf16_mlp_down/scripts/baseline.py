# SPDX-License-Identifier: MIT
"""Public upstream dispatch with call-local alignment preparation."""


def mlp_down(x, weight):
    from aiter.tuned_gemm import gemm_a16w16

    # Keep the upstream kernel selection while accepting contiguous views.
    # These operand copies belong to the measured/captured invocation.
    if x.data_ptr() % 16:
        x = x.clone()
    if weight.data_ptr() % 16:
        weight = weight.clone()
    return gemm_a16w16(x, weight)
