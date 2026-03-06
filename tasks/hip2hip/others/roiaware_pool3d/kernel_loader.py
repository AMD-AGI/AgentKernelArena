# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

roiaware_pool3d_ext = load(name="roiaware_pool3d",
                           sources=["src/roiaware_pool3d_kernel.cu", "src/roiaware_pool3d.cpp"],
                           verbose=True)


