"""High-precision reference for the InnerProd forward product, used only for correctness."""
import torch


def reference(*, feat_img, feat_sound, scale, bias):
    batch, channels = feat_sound.shape[0], feat_sound.shape[1]
    weighted = feat_img.double().reshape(batch, 1, channels) * scale.double().reshape(1, 1, channels)
    z = torch.bmm(weighted, feat_sound.double().reshape(batch, channels, -1))
    return z.reshape(batch, 1, *feat_sound.shape[2:]) + bias.double()
