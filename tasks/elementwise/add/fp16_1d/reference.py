import torch


def reference(*, x, y):
    return x.double() + y.double()
