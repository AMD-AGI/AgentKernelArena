# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Upstream reads the AITER_TRITON_CONV_AUTOTUNE environment variable here; it is off by default (the
# production and CI path uses the tuned configuration tables). The environment lookup is removed and the
# default kept, so the kernel files below launch with the tuned table and never wrap in triton.autotune.
CONV_AUTOTUNE_ENABLED = False
