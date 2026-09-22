# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
kernel_loader_template = """

from torch.utils.cpp_extension import load

hip_{kernel_name}_ext = load(name="{kernel_name}",
                             sources=["{code_dir}/{code_file}"],
                             verbose=True)
hip_fn = hip_{kernel_name}_ext.forward

"""
