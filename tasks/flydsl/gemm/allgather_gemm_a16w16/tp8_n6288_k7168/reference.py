# SPDX-License-Identifier: Apache-2.0
"""Float64 reference of the fused all-gather + in_proj GEMM at TP8.

The logical output is out[m, 8 * 6288] = x[m, 7168] @ w[8 * 6288, 7168]^T over the gathered activations
x and every rank's in_proj rows; rank r's expected output is its columns 6288 r .. (inputs.rank_expected).
Computed one rank's rows at a time, so the float64 copy of the weight stays at one rank's 360 MB.
"""

import torch

TP = 8


def reference(*, x, w, epoch):
    """out = x @ w^T in float64 from the bfloat16 inputs (``epoch`` holds launch state, not data)."""
    del epoch
    n = w.shape[0] // TP
    x64 = x.double()
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float64, device=x.device)
    for r in range(TP):
        out[:, r * n:(r + 1) * n] = x64 @ w[r * n:(r + 1) * n].double().T
    return {"out": out}
