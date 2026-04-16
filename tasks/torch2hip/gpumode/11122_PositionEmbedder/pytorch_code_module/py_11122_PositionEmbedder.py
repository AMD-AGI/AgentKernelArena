# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.nn


class PositionEmbedder(torch.nn.Module):
    """
    [batch_size, seq_length, embedding_size]
    """

    def __init__(self, max_sequence_length: 'int', embedding_dim: 'int'):
        super(PositionEmbedder, self).__init__()
        self.embedding = torch.nn.Embedding(max_sequence_length,
            embedding_dim, padding_idx=None)

    def forward(self, input_embeddings: 'torch.Tensor', fn=None):
        positions = torch.arange(input_embeddings.shape[1], device=
            input_embeddings.device)
        return input_embeddings + self.embedding(positions)


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
