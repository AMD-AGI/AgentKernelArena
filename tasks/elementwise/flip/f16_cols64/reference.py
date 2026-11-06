import torch


def reference(*, x):
    return x.double().flip(-1)
