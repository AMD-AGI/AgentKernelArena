# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

points_in_boxes_ext = load(name="points_in_boxes",
                           sources=["src/points_in_boxes_cuda.hip", "src/points_in_boxes.cpp"],
                           verbose=True)


