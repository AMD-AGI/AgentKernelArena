# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

assign_score_withk_ext = load(name="assign_score_withk",
                              extra_include_paths=["src/include"],
                              sources=["src/assign_score_withk_cuda.hip", "src/assign_score_withk.cpp"],
                              verbose=True)


