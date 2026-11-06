"""Float64 depthwise 3x3 convolution reference used only for correctness."""
import torch

IMAGES_PER_CHUNK = 4


def reference(*, x, weight):
    batch, channels, height, width = x.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    out_h, out_w = height - kernel_h + 1, width - kernel_w + 1
    taps = weight.double().view(1, channels, kernel_h, kernel_w)
    out = torch.zeros(batch, channels, out_h, out_w, dtype=torch.float64, device=x.device)
    # Direct sum over the taps, a few images at a time to bound float64 temporaries.
    for start in range(0, batch, IMAGES_PER_CHUNK):
        chunk = x[start:start + IMAGES_PER_CHUNK].double()
        target = out[start:start + IMAGES_PER_CHUNK]
        for i in range(kernel_h):
            for j in range(kernel_w):
                target.addcmul_(chunk[:, :, i:i + out_h, j:j + out_w], taps[:, :, i:i + 1, j:j + 1])
    return out
