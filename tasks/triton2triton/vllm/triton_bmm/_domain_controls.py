"""Untimed public-domain controls for ``bmm_triton`` with an FP64 CPU oracle.

The protected wrapper accepts any 3-D ``(B, M, K) x (B, K, N)`` pair of equal
float16, bfloat16 or float32 dtype and forwards every operand stride to the
kernel. It allocates a fresh contiguous ``(B, M, N)`` output of the input dtype
on ``a.device``. These controls exercise that domain beyond the five scored
contiguous float16 shapes: both other dtypes, dimensions that are not
multiples of the wrapper's tile sizes (128/128/32 for 16-bit, 64/64/32 for
float32), and strided caller views. They never contribute a timing score.
"""
import torch

SENTINEL = 512.0
# fp16: the task's original gate. bf16: one bf16 ulp is at most 2**-7 (0.78%)
# of |x|, below rtol after FP32 accumulation and one final rounding. fp32: the
# same bound the protected _arena_checks wrapper already applies to every call;
# layout, tail and shape defects produce O(1) errors for these N(0, 1) inputs.
TOLERANCES = {'float16': (1e-2, 1e-2), 'bfloat16': (1e-2, 1e-2), 'float32': (1e-2, 1e-2)}
LAYOUTS = ('contiguous', 'transposed', 'batch_inner', 'padded', 'inner_step2',
           'batch_step2', 'broadcast_batch')

CASES = [
    {'name': 'bf16_contiguous_tails', 'dtype': 'bfloat16', 'shape_bmnk': [3, 129, 130, 33],
     'a_layout': 'contiguous', 'b_layout': 'contiguous'},
    {'name': 'fp32_contiguous_tails', 'dtype': 'float32', 'shape_bmnk': [1, 65, 67, 37],
     'a_layout': 'contiguous', 'b_layout': 'contiguous'},
    {'name': 'fp16_small_odd', 'dtype': 'float16', 'shape_bmnk': [7, 3, 5, 9],
     'a_layout': 'contiguous', 'b_layout': 'contiguous'},
    {'name': 'fp16_single_row_unit_k', 'dtype': 'float16', 'shape_bmnk': [5, 1, 129, 1],
     'a_layout': 'contiguous', 'b_layout': 'contiguous'},
    {'name': 'fp16_a_transposed', 'dtype': 'float16', 'shape_bmnk': [3, 77, 45, 50],
     'a_layout': 'transposed', 'b_layout': 'contiguous'},
    {'name': 'bf16_b_transposed', 'dtype': 'bfloat16', 'shape_bmnk': [4, 33, 70, 129],
     'a_layout': 'contiguous', 'b_layout': 'transposed'},
    {'name': 'fp32_both_transposed', 'dtype': 'float32', 'shape_bmnk': [2, 70, 33, 65],
     'a_layout': 'transposed', 'b_layout': 'transposed'},
    {'name': 'bf16_a_batch_inner', 'dtype': 'bfloat16', 'shape_bmnk': [6, 17, 72, 40],
     'a_layout': 'batch_inner', 'b_layout': 'contiguous'},
    {'name': 'fp16_padded_offset_views', 'dtype': 'float16', 'shape_bmnk': [3, 45, 67, 35],
     'a_layout': 'padded', 'b_layout': 'padded'},
    {'name': 'fp32_inner_step2_views', 'dtype': 'float32', 'shape_bmnk': [2, 31, 29, 19],
     'a_layout': 'inner_step2', 'b_layout': 'inner_step2'},
    {'name': 'fp16_batch_step2_broadcast_b', 'dtype': 'float16', 'shape_bmnk': [4, 40, 48, 24],
     'a_layout': 'batch_step2', 'b_layout': 'broadcast_batch'},
]


def _case(index):
    if type(index) is not int or not 0 <= index < len(CASES):
        raise ValueError('Unknown domain correctness control')
    return CASES[index]


def _place(values, layout, device):
    """Return a caller view holding logical (batch, rows, cols) values and its owner."""
    batch, rows, cols = values.shape
    options = {'device': device, 'dtype': values.dtype}
    if layout == 'contiguous':
        owner = torch.empty((batch, rows, cols), **options)
        view = owner
    elif layout == 'transposed':
        owner = torch.empty((batch, cols, rows), **options)
        view = owner.transpose(1, 2)
    elif layout == 'batch_inner':
        owner = torch.empty((rows, batch, cols), **options)
        view = owner.transpose(0, 1)
    elif layout == 'padded':
        owner = torch.full((batch, rows + 3, cols + 5), SENTINEL, **options)
        view = owner[:, 2:rows + 2, 3:cols + 3]
    elif layout == 'inner_step2':
        owner = torch.full((batch, rows, 2 * cols), SENTINEL, **options)
        view = owner[:, :, ::2]
    elif layout == 'batch_step2':
        owner = torch.full((2 * batch, rows, cols), SENTINEL, **options)
        view = owner[::2]
    elif layout == 'broadcast_batch':
        owner = torch.empty((1, rows, cols), **options)
        owner.copy_(values[:1])
        return owner.expand(batch, rows, cols), owner
    else:
        raise ValueError(f'Unknown operand layout {layout}')
    view.copy_(values)
    return view, owner


def make_case(index, device):
    """Return (a, b, owners); owners are the complete backing tensors of a and b."""
    case = _case(index)
    batch, m, n, k = case['shape_bmnk']
    dtype = getattr(torch, case['dtype'])
    generator = torch.Generator().manual_seed(20000 + index)
    a_values = torch.randn((batch, m, k), generator=generator).to(dtype)
    b_values = torch.randn((batch, k, n), generator=generator).to(dtype)
    if case['b_layout'] == 'broadcast_batch':
        b_values = b_values[:1].expand(batch, k, n)
    a, a_owner = _place(a_values, case['a_layout'], device)
    b, b_owner = _place(b_values, case['b_layout'], device)
    return a, b, (a_owner, b_owner)


def reference(a, b):
    """FP64 CPU batched product of the caller-visible operand values."""
    return torch.bmm(a.detach().cpu().double(), b.detach().cpu().double())


def check_output(output, expected, case, device):
    batch, m, n, _ = case['shape_bmnk']
    if not isinstance(output, torch.Tensor):
        raise AssertionError('Domain control output must be a tensor')
    if (tuple(output.shape) != (batch, m, n) or output.dtype != getattr(torch, case['dtype'])
            or output.device != device):
        raise AssertionError('Domain control output shape/dtype/device differs from contract')
    if not output.is_contiguous() or output.stride() != (m * n, n, 1):
        raise AssertionError('Domain control output is not the contiguous (B, M, N) allocation')
    observed = output.detach().cpu().double()
    if not torch.isfinite(observed).all():
        raise AssertionError('Domain control output contains nonfinite values')
    atol, rtol = TOLERANCES[case['dtype']]
    if not torch.allclose(observed, expected, atol=atol, rtol=rtol):
        diff = (observed - expected).abs().max().item()
        raise AssertionError(f'Domain control max abs diff {diff:.6f} exceeds atol={atol}, rtol={rtol}')


def expected_strides(case):
    """Strides each declared layout must give the caller views (used by validation)."""
    batch, m, n, k = case['shape_bmnk']
    def strides(layout, rows, cols):
        return {'contiguous': (rows * cols, cols, 1), 'transposed': (rows * cols, 1, rows),
                'batch_inner': (cols, batch * cols, 1),
                'padded': ((rows + 3) * (cols + 5), cols + 5, 1),
                'inner_step2': (2 * rows * cols, 2 * cols, 2),
                'batch_step2': (2 * rows * cols, cols, 1), 'broadcast_batch': (0, cols, 1)}[layout]
    return strides(case['a_layout'], m, k), strides(case['b_layout'], k, n)


def validate_reference(harness):
    """CPU-only oracle checks; they do not execute the candidate or claim GPU results."""
    for index, case in enumerate(CASES):
        a, b, owners = make_case(index, 'cpu')
        if (a.stride(), b.stride()) != expected_strides(case):
            raise AssertionError(f'Domain control {case["name"]} layout differs from declaration')
        for view, owner, layout in zip((a, b), owners, (case['a_layout'], case['b_layout'])):
            gaps = owner.numel() - view.numel() if layout in ('padded', 'inner_step2', 'batch_step2') else 0
            if int((owner == SENTINEL).sum()) != gaps:
                raise AssertionError(f'Domain control {case["name"]} storage gaps lack sentinels')
        expected = reference(a, b)
        explicit = (a.double()[:, :, :, None] * b.double()[:, None, :, :]).sum(2)
        torch.testing.assert_close(expected, explicit, atol=1e-9, rtol=1e-9)
        original = torch.bmm(a.float(), b.float()).to(a.dtype)
        check_output(original, expected, case, a.device)
        rejected = (torch.zeros_like(original),
                    torch.bmm(a[..., :-1].float(), b[:, :-1].float()).to(a.dtype))
        for wrong in rejected:
            try:
                check_output(wrong, expected, case, a.device)
            except AssertionError:
                continue
            raise AssertionError(f'Domain control {case["name"]} accepts a wrong output')


def run_control(index, load_module):
    try:
        case = _case(index)
        a, b, owners = make_case(index, 'cuda')
        pristine = a.clone(), b.clone()
        saved_owners = tuple(owner.clone() for owner in owners)
        expected = reference(*pristine)
        output = load_module().bmm_triton(a, b)
        if (not torch.equal(a, pristine[0]) or not torch.equal(b, pristine[1]) or
                any(not torch.equal(owner, saved) for owner, saved in zip(owners, saved_owners))):
            raise AssertionError('Domain control inputs and their storage are read-only')
        check_output(output, expected, case, a.device)
        return True, None
    except Exception as exc:
        return False, f'Domain control {index}: {type(exc).__name__}: {exc}'
