"""Float64 reference: conv3d with bias -> softmax over channels -> two 2x2x2 max pools."""
import torch
import torch.nn.functional as F

# Convolution output elements per batch chunk; bounds the float64 temporaries.
_CHUNK_ELEMENTS = 1 << 25


def reference(*, x, weight, bias):
    w64 = weight.double()
    b64 = bias.double()
    per_clip = w64.shape[0]
    for axis in range(3):
        per_clip *= x.shape[2 + axis] - w64.shape[2 + axis] + 1
    step = max(1, _CHUNK_ELEMENTS // per_clip)
    pooled = []
    for start in range(0, x.shape[0], step):
        y = F.conv3d(x[start:start + step].double(), w64, b64)
        y = torch.softmax(y, dim=1)
        y = F.max_pool3d(y, kernel_size=2)
        y = F.max_pool3d(y, kernel_size=2)
        pooled.append(y)
    return torch.cat(pooled)
