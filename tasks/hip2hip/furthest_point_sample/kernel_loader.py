# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

furthest_point_sample_ext = load(name="furthest_point_sample",
               extra_include_paths=["src/include"],
               sources=["src/furthest_point_sample_cuda.hip", "src/furthest_point_sample.cpp"],
               verbose=True)


