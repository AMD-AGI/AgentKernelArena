"""Untimed actual-operator head_first and varlen (cu_seqlens/chunk_indices) controls.

The ten scored workloads use the regular [B,T,H(,S)] layout without cu_seqlens.
These correctness-only identities call the same public chunk_local_cumsum with
head_first=True ([B,H,T(,S)]) and with packed variable-length sequences
(B=1, cu_seqlens holding N+1 offsets). Chunks restart at every sequence start;
the final chunk of a sequence may be partial, including a one-token sequence.
chunk_indices, when supplied, lists (sequence, local chunk) pairs in order.
head_first with cu_seqlens is exercised for one sequence only: the retained
kernel addresses each sequence as its own [H,T_n] block (bos*H + h*T_n), which
equals the logical [1,H,T] layout only when N=1, and upstream FLA/SGLang
callers reject variable-length head-first inputs.

Expected values are FP64 plain-torch chunk sums computed on CPU from pristine
input clones before each call. Each identity reuses one input allocation for a
base, changed and base-again draw. Nothing here is timed or scored.
"""

CHUNK_SIZE = 64


def _row(name, seed, **params):
    params = dict(params, seed=seed)
    return {"test_case_id": name, "params": params, "checks": ["correctness"]}


CONTROL_CASES = [
    _row("control_head_first_scalar", 1001, ndim=3, B=2, T=200, H=3, head_first=True,
         seq_lens=None, cu_seqlens_dtype=None, chunk_indices=None, reverse=False, scale=None),
    _row("control_head_first_scalar_reverse_scale", 1002, ndim=3, B=1, T=129, H=5,
         head_first=True, seq_lens=None, cu_seqlens_dtype=None, chunk_indices=None,
         reverse=True, scale=0.5),
    _row("control_head_first_vector", 1003, ndim=4, B=2, T=100, H=2, S=24, head_first=True,
         seq_lens=None, cu_seqlens_dtype=None, chunk_indices=None, reverse=True, scale=None),
    _row("control_varlen_scalar", 1004, ndim=3, B=1, T=301, H=4, head_first=False,
         seq_lens=[1, 64, 100, 129, 7], cu_seqlens_dtype="int32", chunk_indices="omitted",
         reverse=False, scale=None),
    _row("control_varlen_scalar_chunk_indices", 1005, ndim=3, B=1, T=269, H=3,
         head_first=False, seq_lens=[65, 1, 200, 3], cu_seqlens_dtype="int32",
         chunk_indices="supplied", reverse=True, scale=0.5),
    _row("control_varlen_vector", 1006, ndim=4, B=1, T=232, H=2, S=40, head_first=False,
         seq_lens=[70, 1, 128, 33], cu_seqlens_dtype="int64", chunk_indices="omitted",
         reverse=False, scale=0.5),
    _row("control_varlen_vector_chunk_indices", 1007, ndim=4, B=1, T=193, H=3, S=16,
         head_first=False, seq_lens=[1, 63, 64, 65], cu_seqlens_dtype="int64",
         chunk_indices="supplied", reverse=True, scale=None),
    _row("control_head_first_varlen_scalar", 1008, ndim=3, B=1, T=150, H=4, head_first=True,
         seq_lens=[150], cu_seqlens_dtype="int32", chunk_indices="omitted",
         reverse=False, scale=None),
    _row("control_head_first_varlen_vector", 1009, ndim=4, B=1, T=97, H=2, S=8,
         head_first=True, seq_lens=[97], cu_seqlens_dtype="int64", chunk_indices="supplied",
         reverse=True, scale=0.5),
]


def input_shape(params):
    """Public input shape: [B,T,H(,S)], or [B,H,T(,S)] when head_first."""
    B, T, H = params["B"], params["T"], params["H"]
    shape = [B, H, T] if params["head_first"] else [B, T, H]
    return shape + ([params["S"]] if params["ndim"] == 4 else [])


def sequence_offsets(params):
    """Validated cu_seqlens values for a varlen identity, else None."""
    lens = params["seq_lens"]
    if lens is None:
        return None
    if params["B"] != 1 or not lens or any(type(n) is not int or n < 1 for n in lens):
        raise ValueError("Varlen controls use B=1 and positive sequence lengths")
    if sum(lens) != params["T"]:
        raise ValueError("Sequence lengths must cover the packed time axis")
    if params["head_first"] and len(lens) != 1:
        raise ValueError("head_first with cu_seqlens is defined here for one sequence only")
    offsets = [0]
    for n in lens:
        offsets.append(offsets[-1] + n)
    return offsets


def chunk_index_pairs(offsets, chunk_size=CHUNK_SIZE):
    """(sequence, local chunk) pairs, in order, computed with plain Python."""
    return [[n, c] for n in range(len(offsets) - 1)
            for c in range(-(-(offsets[n + 1] - offsets[n]) // chunk_size))]


def reference(g, *, chunk_size=CHUNK_SIZE, reverse=False, scale=None,
              head_first=False, offsets=None):
    """FP64 chunk-local inclusive prefix (or suffix) sums on CPU."""
    import torch

    x = g.detach().to(device="cpu", dtype=torch.float64)
    if head_first:
        x = x.transpose(1, 2)
    out = torch.empty_like(x)
    T = x.shape[1]
    bounds = [(0, T)] if offsets is None else list(zip(offsets[:-1], offsets[1:]))
    if bounds[-1][1] != T or (offsets is not None and x.shape[0] != 1):
        raise ValueError("Sequence offsets disagree with the input")
    for b in range(x.shape[0]):
        for bos, eos in bounds:
            for t0 in range(bos, eos, chunk_size):
                t1 = min(t0 + chunk_size, eos)
                chunk = x[b, t0:t1]
                out[b, t0:t1] = (chunk.flip(0).cumsum(0).flip(0) if reverse
                                 else chunk.cumsum(0))
    if scale is not None:
        out = out * scale
    if head_first:
        out = out.transpose(1, 2)
    return out.contiguous()


def make_draw(params, changed=False, device="cuda"):
    """Seeded negative log-gates plus public keyword arguments for one draw."""
    import torch
    import torch.nn.functional as F

    generator = torch.Generator().manual_seed(params["seed"] + (500 if changed else 0))
    g = F.logsigmoid(torch.randn(input_shape(params), generator=generator,
                                 dtype=torch.float32)).to(device)
    kwargs = {"chunk_size": CHUNK_SIZE, "reverse": params["reverse"], "scale": params["scale"]}
    if params["head_first"]:
        kwargs["head_first"] = True
    offsets = sequence_offsets(params)
    if offsets is not None:
        dtype = {"int32": torch.int32, "int64": torch.int64}[params["cu_seqlens_dtype"]]
        kwargs["cu_seqlens"] = torch.tensor(offsets, dtype=dtype, device=device)
        if params["chunk_indices"] == "supplied":
            kwargs["chunk_indices"] = torch.tensor(chunk_index_pairs(offsets), dtype=dtype,
                                                   device=device)
        elif params["chunk_indices"] != "omitted":
            raise ValueError("Varlen controls state whether chunk_indices is supplied")
    return g, kwargs, offsets


def run_controls(operator, compare, unchanged, *, device="cuda", start_shape_id=11):
    """Call the actual operator for every control; raise on the first failure."""
    import torch

    records = []
    for index, case in enumerate(CONTROL_CASES):
        p = case["params"]
        g = None
        max_diff = 0.0
        for draw, changed in enumerate((False, True, False)):
            try:
                values, kwargs, offsets = make_draw(p, changed, device)
                if g is None:
                    g = values
                else:
                    g.copy_(values)
                inputs = (g,) + tuple(v for v in kwargs.values() if isinstance(v, torch.Tensor))
                originals = tuple(v.clone() for v in inputs)
                expected = reference(originals[0], reverse=p["reverse"], scale=p["scale"],
                                     head_first=p["head_first"], offsets=offsets)
                expected = expected.to(device=g.device, dtype=torch.float32)
                with torch.no_grad():
                    actual = operator(g, **kwargs)
                unchanged(inputs, originals)
                compare(actual, expected)
                max_diff = max(max_diff, (actual.double() - expected.double()).abs().max().item())
                try:
                    compare(expected + 100.0, expected)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("Comparator accepted deliberately incorrect output")
            except Exception as error:
                raise AssertionError(f"{case['test_case_id']} draw {draw}: "
                                     f"{type(error).__name__}: {error}") from error
        records.append({"shape_id": start_shape_id + index, "shape": input_shape(p),
                        "control": case["test_case_id"], "head_first": p["head_first"],
                        "seq_lens": p["seq_lens"], "chunk_indices": p["chunk_indices"],
                        "reverse": p["reverse"], "scale": p["scale"], "max_diff": max_diff,
                        "passed": True, "known_answer": "PASS", "actual_operator_draws": 3,
                        "changed_input": "PASS", "read_only": "PASS",
                        "negative_output": "rejected", "timed": False})
    return records
