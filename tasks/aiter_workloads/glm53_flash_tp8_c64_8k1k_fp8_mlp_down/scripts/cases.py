"""Fresh generated operands for the declared workload scenario; no saved kernels."""


def make_case(case, seed):
    import torch
    from aiter.ops.shuffle import shuffle_weight

    m, n, k = case["shape"]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # Finite E4M3 inputs and positive FP32 block scales follow the upstream
    # test's operand roles. The exact distributions here are declared policy,
    # not samples of a trained checkpoint's activations or weights.
    x = torch.empty((m, k), device="cuda").uniform_(-2, 2, generator=gen).to(torch.float8_e4m3fn)
    canonical = torch.empty((n, k), device="cuda").uniform_(-2, 2, generator=gen).to(torch.float8_e4m3fn)
    xs = torch.pow(2.0, torch.randint(-4, 1, (m, k // 128), device="cuda", generator=gen).float())
    ws = torch.pow(2.0, torch.randint(-7, -1, (n // 128, k // 128), device="cuda", generator=gen).float())
    # The public tensor has a column-major scale stride, not reinterpreted
    # row-major logical values. AITER accepts this documented strided form.
    xs = xs.t().contiguous().t()
    weight = shuffle_weight(canonical, layout=(16, 16))
    return (x, weight, xs, ws), canonical


def snapshot(args):
    import torch

    return tuple((identity(value), whole_bytes(value).clone()) for value in args)


def identity(value):
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device,
            value.data_ptr(), value.storage_offset(), value.untyped_storage().nbytes(),
            getattr(value, "is_shuffled", None))


def whole_bytes(value):
    import torch

    return torch.empty(0, dtype=torch.uint8, device=value.device).set_(
        value.untyped_storage(), 0, (value.untyped_storage().nbytes(),), (1,)
    )


def check_unchanged(args, before):
    import torch

    for value, (old_identity, old) in zip(args, before):
        if identity(value) != old_identity or not torch.equal(whole_bytes(value), old):
            raise AssertionError("The implementation mutated a read-only input")
