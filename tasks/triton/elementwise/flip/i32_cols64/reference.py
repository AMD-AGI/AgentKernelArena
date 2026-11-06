import torch


def reference(*, x):
    return x.flip(-1)
