# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

interpolate_ext = load(name="three_nn",
                       sources=["src/three_nn_cuda.hip", "src/three_nn.cpp"],
                       verbose=True)


