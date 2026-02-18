# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

roipoint_pool3d_ext = load(name="roipoint_pool3d",
                           sources=["src/roipoint_pool3d_kernel.hip", "src/roipoint_pool3d.cpp"],
                           verbose=True)


