# aiter/ops/triton/utils/types.py has no license header; the repository's MIT license applies.
import torch


def _is_fp8(x):
    # Upstream additionally probes the architecture (arch_info.is_fp8_avail()) before
    # returning True for a float8 dtype; this task never passes float8 tensors.
    return x.dtype in {
        torch.float8_e4m3fnuz,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }
