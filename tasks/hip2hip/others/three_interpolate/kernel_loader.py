# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

interpolate_ext = load(name="three_interpolate",
                       sources=["src/three_interpolate_cuda.hip", "src/three_interpolate.cpp"],
                       verbose=True)


