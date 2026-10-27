# SPDX-License-Identifier: MIT
"""New normal BF16 operands, following upstream test_gemm_a16w16.py.

These are generated test values, not checkpoint weights or serving captures.
The same draws, layouts and tensor lifetime apply to both evaluation roles.
"""


def make_inputs(case, seed, *, device="cuda"):
    import torch

    m, n, k = case["shape"]
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, k), generator=generator, device=device, dtype=torch.bfloat16)
    # Keep the prepared site's weight draw equal across the two M cases.
    weight_generator = torch.Generator(device=device).manual_seed(seed + 1_000_000)
    weight = torch.randn((n, k), generator=weight_generator, device=device, dtype=torch.bfloat16)
    return x, weight


def snapshot_inputs(values):
    return [(value, value.clone(), value.shape, value.stride(), value.dtype,
             value.device, value.data_ptr(), value.storage_offset(),
             value.untyped_storage().nbytes()) for value in values]


def check_inputs_unchanged(snapshots):
    import torch

    for value, before, shape, stride, dtype, device, pointer, offset, nbytes in snapshots:
        if (value.shape != shape or value.stride() != stride or value.dtype != dtype
                or value.device != device or value.data_ptr() != pointer
                or value.storage_offset() != offset or value.untyped_storage().nbytes() != nbytes
                or not torch.equal(value, before)):
            raise AssertionError("Public GEMM inputs must remain unchanged")
