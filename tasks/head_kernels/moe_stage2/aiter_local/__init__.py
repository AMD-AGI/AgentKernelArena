import logging
import torch

logger = logging.getLogger("aiter_local")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[aiter_local] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

from .utility import dtypes  # noqa: E402,F401
from .ops.enum import *  # noqa: F403,E402
from .ops.quant import *  # noqa: F403,E402
from .ops.shuffle import *  # noqa: F403,E402
from .ops.moe_op import *  # noqa: F403,E402
from .ops.moe_sorting import *  # noqa: F403,E402
from .ops.moe_sorting_opus import *  # noqa: F403,E402

def get_torch_act(activation):
    from .ops.enum import ActivationType
    if activation == ActivationType.Silu:
        return torch.nn.functional.silu
    if activation == ActivationType.Gelu:
        return torch.nn.functional.gelu
    if activation == ActivationType.Relu:
        return torch.nn.functional.relu
    return torch.nn.functional.silu

def fused_dynamic_mxfp4_quant_moe_sort(*_args, **_kwargs):
    raise NotImplementedError("mxfp4 MoE path is not vendored for this task")

def mxfp4_moe_sort_fwd(*_args, **_kwargs):
    raise NotImplementedError("mxfp4 MoE path is not vendored for this task")
