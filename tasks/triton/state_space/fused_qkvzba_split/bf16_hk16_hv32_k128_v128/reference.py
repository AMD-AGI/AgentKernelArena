import torch

NUM_KEY_HEADS = 16
NUM_VALUE_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128


def reference(*, mixed_qkvz, mixed_ba):
    """Pure data movement; the outputs are exact copies of input elements, returned in float32
    (every bfloat16 value is exact there) so that a bitwise comparison in bfloat16 is exact."""
    tokens = mixed_qkvz.shape[0]
    ratio = NUM_VALUE_HEADS // NUM_KEY_HEADS
    group = mixed_qkvz.view(tokens, NUM_KEY_HEADS, 2 * HEAD_K_DIM + 2 * ratio * HEAD_V_DIM)
    q = group[:, :, :HEAD_K_DIM]
    k = group[:, :, HEAD_K_DIM:2 * HEAD_K_DIM]
    v = group[:, :, 2 * HEAD_K_DIM:2 * HEAD_K_DIM + ratio * HEAD_V_DIM]
    z = group[:, :, 2 * HEAD_K_DIM + ratio * HEAD_V_DIM:]
    mixed_qkv = torch.cat([q.reshape(tokens, -1), k.reshape(tokens, -1), v.reshape(tokens, -1)], dim=-1)
    pairs = mixed_ba.view(tokens, NUM_KEY_HEADS, 2 * ratio)
    b = pairs[:, :, :ratio].reshape(tokens, NUM_VALUE_HEADS)
    a = pairs[:, :, ratio:].reshape(tokens, NUM_VALUE_HEADS)
    return (mixed_qkv.float(), z.reshape(tokens, NUM_VALUE_HEADS, HEAD_V_DIM).float(), b.float(), a.float())
