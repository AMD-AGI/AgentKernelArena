"""Float64 reference for conv2d -> tanh -> scale -> bias -> max_pool2d(4), used only for correctness."""
import torch
import torch.nn.functional as F

POOL_SIZE = 4
BATCH_CHUNK = 16  # bounds the float64 convolution intermediate of the largest batches


def _float32(value):
    """The value of a declared float32 scalar, as the module's float32 arithmetic applies it."""
    return float(torch.tensor(value, dtype=torch.float32))


def reference(*, x, conv_weight, conv_bias, scaling_factor, bias):
    weight = conv_weight.double()
    conv_b = conv_bias.double()
    channel_bias = bias.double()
    scale = _float32(scaling_factor)
    pooled = []
    for part in x.split(BATCH_CHUNK):
        y = torch.tanh(F.conv2d(part.double(), weight, conv_b))
        y = y * scale + channel_bias
        pooled.append(F.max_pool2d(y, POOL_SIZE))
    return torch.cat(pooled)
