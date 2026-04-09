# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

gather_points_ext = load(name="gather_points",
                         sources=["src/gather_points_cuda.hip", "src/gather_points.cpp"],
                         verbose=True)


