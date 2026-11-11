# SPDX-License-Identifier: Apache-2.0
# Vendored for AgentKernelArena from sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d,
# sgl_amd_mono/kimi_k3/gfx950/stages/dims.py. AKA changes: relative imports only.
"""Shapes the Kimi-K3 mono launches share.

The AttnRes seam, o_proj and the KDA front all read these, so they live
apart from any one launch.
"""

import math

HIDDEN = 7168
NA = 56  # attn CTAs
COLS = HIDDEN // NA  # 128 columns a slice
QC = COLS // 4  # threads a (token, source) row: 4 columns each
GSL = 7  # slices a gate / mix-norm thread sums (NA = 8 x 7)
HD = 128
NH = 12  # local heads at TP8
PROJ = NH * HD  # 1536
ROWS = 16
LOG2E = 1.0 / math.log(2.0)
LDS_PAD = 8  # bf16 a padded x row
