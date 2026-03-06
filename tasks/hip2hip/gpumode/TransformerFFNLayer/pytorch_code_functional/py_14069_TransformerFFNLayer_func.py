# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import collections

# Helper for unique instance id tracking for incremental state
INCREMENTAL_STATE_INSTANCE_ID = collections.defaultdict(int)

def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__
    if not hasattr(module_instance, '_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]
    return '{}.{}.{}'.format(module_name, module_instance._instance_id, key)

def get_incremental_state(module, incremental_state, key):
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]

def set_incremental_state(module, incremental_state, key, value):
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value

# Functional kernel for TransformerFFNLayer
def transformer_ffn_layer_fn(
    x: torch.Tensor,
    ffn1_weight: torch.Tensor,
    ffn1_bias: torch.Tensor,
    ffn2_weight: torch.Tensor,
    ffn2_bias: torch.Tensor,
    kernel_size: int,
    dropout: float,
    act: str,
    padding: str = 'SAME',
    training: bool = False,
    incremental_state: dict = None,
    module_instance=None,
):
    """
    Functional implementation of TransformerFFNLayer forward.
    """
    # Handle incremental state
    if incremental_state is not None and module_instance is not None:
        # Get buffer
        def _get_input_buffer(module, incremental_state):
            result = get_incremental_state(module, incremental_state, 'f')
            return result if isinstance(result, dict) else {}

        def _set_input_buffer(module, incremental_state, buffer):
            set_incremental_state(module, incremental_state, 'f', buffer)

        saved_state = _get_input_buffer(module_instance, incremental_state)
        if isinstance(saved_state, dict) and 'prev_input' in saved_state:
            prev_input = saved_state['prev_input']
            x = torch.cat((prev_input, x), dim=0)
        x = x[-kernel_size:]
        saved_state['prev_input'] = x
        _set_input_buffer(module_instance, incremental_state, saved_state)

    # x: (T, B, C) -> (B, C, T)
    x = x.permute(1, 2, 0)
    # Conv1d
    if padding == 'SAME':
        x = F.conv1d(x, ffn1_weight, ffn1_bias, padding=kernel_size // 2)
    elif padding == 'LEFT':
        x = F.pad(x, (kernel_size - 1, 0), value=0.0)
        x = F.conv1d(x, ffn1_weight, ffn1_bias)
    # (B, F, T) -> (T, B, F)
    x = x.permute(2, 0, 1)
    x = x * (kernel_size ** -0.5)
    if incremental_state is not None:
        x = x[-1:]
    # Activation
    if act == 'gelu':
        x = F.gelu(x)
    elif act == 'relu':
        x = F.relu(x)
    # Dropout
    x = F.dropout(x, dropout, training=training)
    # Linear
    # x: (T, B, F)
    x = F.linear(x, ffn2_weight, ffn2_bias)
    return x

# Module wrapper for functional implementation
class TransformerFFNLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, padding='SAME', kernel_size=1, dropout=0.0, act='gelu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        self.padding = padding
        if padding == 'SAME':
            conv = nn.Conv1d(hidden_size, filter_size, kernel_size, padding=kernel_size // 2)
        elif padding == 'LEFT':
            conv = nn.Conv1d(hidden_size, filter_size, kernel_size)
        self.ffn1_weight = nn.Parameter(conv.weight)
        self.ffn1_bias = nn.Parameter(conv.bias)
        linear = Linear(filter_size, hidden_size)
        self.ffn2_weight = nn.Parameter(linear.weight)
        self.ffn2_bias = nn.Parameter(linear.bias)

    def forward(self, x, incremental_state=None, fn=transformer_ffn_layer_fn):
        return fn(
            x,
            self.ffn1_weight,
            self.ffn1_bias,
            self.ffn2_weight,
            self.ffn2_bias,
            self.kernel_size,
            self.dropout,
            self.act,
            self.padding,
            self.training,
            incremental_state,
            self,
        )

def get_inputs():
    """
    Generate multiple test cases with varying sizes
    TransformerFFNLayer forward() expects x: (T, B, C) where C=hidden_size=4
    """
    configs = [
        # (T, B, C) where C must be 4 to match hidden_size=4 initialization
        ([4, 4, 4],),  # T=4, B=4, C=4
        ([8, 4, 4],),  # T=8, B=4, C=4
        ([16, 4, 4],),  # T=16, B=4, C=4
        ([32, 4, 4],),  # T=32, B=4, C=4
        ([64, 4, 4],),  # T=64, B=4, C=4
    ]
    
    for shape in configs:
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        x = torch.randn(shape_list, dtype=torch.float32)
        yield [x]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'filter_size': 4}]
