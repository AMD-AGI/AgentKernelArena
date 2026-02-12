# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

gather_points_ext = load(name="gather_points",
                         extra_include_paths=["src/include"],
                         sources=["src/gather_points_cuda.cu", "src/gather_points.cpp"],
                         verbose=True)


