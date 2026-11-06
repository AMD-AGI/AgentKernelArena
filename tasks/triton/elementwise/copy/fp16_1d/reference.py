import torch


def reference(*, x):
    return x.double().clone()
