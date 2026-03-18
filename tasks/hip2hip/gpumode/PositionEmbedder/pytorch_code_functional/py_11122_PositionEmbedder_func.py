# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

def position_embedding_kernel(
    input_embeddings: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Adds positional embeddings to input embeddings.

    Args:
        input_embeddings (torch.Tensor): Input tensor of shape [batch_size, seq_length, embedding_size]
        embedding_weight (torch.Tensor): Embedding weight tensor of shape [max_sequence_length, embedding_dim]

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, seq_length, embedding_size]
    """
    # Get positions [0, 1, ..., seq_length-1]
    positions = torch.arange(input_embeddings.shape[1], device=input_embeddings.device)
    # Gather positional embeddings
    pos_emb = F.embedding(positions, embedding_weight)
    # Broadcast and add
    return input_embeddings + pos_emb

def module_fn(
    input_embeddings: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Functional implementation of PositionEmbedder.

    Args:
        input_embeddings (torch.Tensor): Input tensor of shape [batch_size, seq_length, embedding_size]
        embedding_weight (torch.Tensor): Embedding weight tensor of shape [max_sequence_length, embedding_dim]

    Returns:
        torch.Tensor: Output tensor of shape [batch_size, seq_length, embedding_size]
    """
    return position_embedding_kernel(input_embeddings, embedding_weight)

class PositionEmbedder(nn.Module):
    """
    [batch_size, seq_length, embedding_size]
    """

    def __init__(self, max_sequence_length: int, embedding_dim: int):
        super(PositionEmbedder, self).__init__()
        self.embedding = nn.Embedding(max_sequence_length, embedding_dim, padding_idx=None)

    def forward(self, input_embeddings: torch.Tensor, fn=module_fn):
        return fn(input_embeddings, self.embedding.weight)

def get_inputs():
    """
    Generate multiple test cases with varying sizes
    Input shape: [batch_size, seq_length, embedding_size]
    - embedding_size must be 4 to match embedding_dim=4 in get_init_inputs()
    - seq_length can vary but must be <= max_sequence_length=4
    """
    configs = [
        ([4, 1, 4],),  # batch=4, seq=1, embed=4
        ([8, 2, 4],),  # batch=8, seq=2, embed=4
        ([16, 3, 4],),  # batch=16, seq=3, embed=4
        ([32, 4, 4],),  # batch=32, seq=4, embed=4
        ([64, 4, 4],),  # batch=64, seq=4, embed=4
    ]
    
    for shape in configs:
        # Unpack tuple if shape is a tuple containing a list (e.g., ([1024],) -> [1024])
        shape_list = shape[0] if isinstance(shape, tuple) and len(shape) == 1 else shape
        # Only yield input_embeddings - embedding_weight is a model parameter
        input_embeddings = torch.randn(shape_list, dtype=torch.float32)
        yield [input_embeddings]


def get_init_inputs():
    return [[], {'max_sequence_length': 4, 'embedding_dim': 4}]
