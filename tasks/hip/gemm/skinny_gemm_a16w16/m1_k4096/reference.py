import torch


def reference(*, x, weight):
    return x.double() @ weight.double().T
