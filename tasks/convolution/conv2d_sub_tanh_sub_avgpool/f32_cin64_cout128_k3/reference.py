"""Float64 reference for conv2d -> subtract -> tanh -> subtract -> avg_pool2d(2), used only for correctness."""
import torch
import torch.nn.functional as F

POOL_SIZE = 2
BATCH_CHUNK = 8  # bounds the float64 convolution intermediate of the largest images


def _float32(value):
    """The value of a declared float32 scalar, as the module's float32 arithmetic applies it."""
    return float(torch.tensor(value, dtype=torch.float32))


def reference(*, x, conv_weight, conv_bias, subtract1_value, subtract2_value):
    weight = conv_weight.double()
    bias = conv_bias.double()
    subtract1 = _float32(subtract1_value)
    subtract2 = _float32(subtract2_value)
    pooled = []
    for part in x.split(BATCH_CHUNK):
        y = F.conv2d(part.double(), weight, bias)
        y = torch.tanh(y - subtract1) - subtract2
        pooled.append(F.avg_pool2d(y, POOL_SIZE))
    return torch.cat(pooled)
