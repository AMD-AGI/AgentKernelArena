"""Input-derived forward-error contract for FP16 weight-only MoE.

The bound permits FP16 dequantization, FP32 products/accumulation and routing,
then FP16 output rounding. It is independent of any candidate/baseline result.
All work runs on private CPU inputs outside the device timer.
"""
import torch


def reference_and_bound(inputs, options):
    data = {key: value.detach().cpu() for key, value in inputs.items()}
    a = data['A'].double()
    q = data['qweight'].to(torch.int64)
    scales = data['scales'].double()
    ids = data['ids'].to(torch.int64)
    m, k = a.shape
    experts, _, n = q.shape
    groups = torch.arange(k) // options['group_size']
    int4 = options['use_int4']
    if int4:
        shifts = (torch.arange(k) % 2 * 4).reshape(1, k, 1)
        q = (q[:, torch.arange(k) // 2, :] >> shifts) & 15
    zeros = data.get('zeros')
    if zeros is None:
        zp = torch.full(scales.shape, 8 if int4 else 128, dtype=torch.float64)
    elif int4:
        shifts = (torch.arange(n) % 2 * 4).reshape(1, 1, n)
        zp = ((zeros.to(torch.int64)[:, :, torch.arange(n) // 2] >> shifts) & 15).double()
    else:
        zp = zeros.double()
    dequant = (q.double() - zp[:, groups, :]) * scales[:, groups, :]

    u16, u32, u64 = 2.0**-11, 2.0**-24, 2.0**-53
    tiny16, eta16 = 2.0**-14, 2.0**-25
    # gamma(2K) allows a separately rounded FP32 multiply and add per term.
    gamma32 = (2*k*u32) / (1 - 2*k*u32)
    gamma64 = (2*k*u64) / (1 - 2*k*u64)
    abs_a, abs_d = a.abs(), dequant.abs()
    # Also allow hardware flush-to-zero of subnormal FP16 inputs/weights.
    delta_a = torch.where(abs_a < tiny16, abs_a, 0.)
    delta_d = u16*abs_d + eta16 + torch.where(abs_d < tiny16, abs_d, 0.)
    topk = ids.shape[1]
    expected = torch.zeros(m*topk, n, dtype=torch.float64)
    bound = torch.zeros_like(expected)
    routing = data.get('weights')
    for expert in range(experts):
        token, lane = torch.where(ids == expert)
        if not token.numel():
            continue
        aa, da = abs_a[token], delta_a[token]
        d, dd = abs_d[expert], delta_d[expert]
        dot = a[token] @ dequant[expert]
        scale = aa @ d
        representation_error = aa @ dd + da @ (d + dd)
        dot_error = representation_error + gamma32 * ((aa + da) @ (d + dd))
        # Include the independent FP64 oracle's own rounding, not just the GPU's.
        dot_error += gamma64 * scale
        rows = token*topk + lane
        route = torch.ones(token.numel(), 1, dtype=torch.float64)
        if options['mul_routed_weight'] and routing is not None:
            route = routing[rows].double().reshape(-1, 1)
        ideal = dot * route
        error = route.abs()*dot_error
        error += (u32 + u64)*route.abs()*(dot.abs() + dot_error) + 2.0**-126
        error += u16*(ideal.abs() + error) + eta16
        # A possibly subnormal result may also be flushed at the output boundary.
        error += torch.where(ideal.abs() - error < tiny16, tiny16, 0.)
        expected[rows], bound[rows] = ideal, error
    # Invalid expert rows stay exactly zero with a zero error budget.
    assert torch.isfinite(expected).all() and torch.isfinite(bound).all(), 'Nonfinite accuracy oracle'
    return expected, bound


def assert_accuracy(actual, expected, bound, mismatch_type):
    error = (actual.detach().cpu().double() - expected).abs()
    if not torch.all(error <= bound):
        raise mismatch_type(
            f'Arithmetic accuracy bound exceeded: max_abs={error.max().item()}, '
            f'max_excess={(error-bound).max().item()}')
