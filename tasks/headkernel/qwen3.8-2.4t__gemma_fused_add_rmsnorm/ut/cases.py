"""Live-shape callable cases for fused Gemma add + RMSNorm."""

import importlib
import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "meta.json")) as _fh:
    META = json.load(_fh)

_RESOLVED = None


def _resolve(target):
    module_name, _, attr_path = target.partition(":")
    obj = importlib.import_module(module_name)
    for part in attr_path.split("."):
        if part:
            obj = getattr(obj, part)
    return obj


def _current_fn():
    global _RESOLVED
    if _RESOLVED is None:
        _RESOLVED = _resolve(META["target_callable"])
    return _RESOLVED


def _storage_ptr(tensor):
    return tensor.untyped_storage().data_ptr()


def _validate_inputs(args):
    required = {"x", "residual", "weight", "eps"}
    missing = required.difference(args)
    if missing:
        raise KeyError(f"add+rmsnorm case is missing {sorted(missing)}")
    x, residual, weight = args["x"], args["residual"], args["weight"]
    if tuple(x.shape) != tuple(residual.shape):
        raise ValueError("x and residual must have the same shape")
    if x.dtype != residual.dtype or weight.dtype != x.dtype:
        raise TypeError("x, residual, and weight must have the same dtype")
    if tuple(weight.shape) != (int(x.shape[-1]),):
        raise ValueError("weight must have one element per hidden column")


def _validate_outputs(result, x, residual, weight):
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("live callable must return (normed, pre_norm_sum)")
    normed, pre_norm_sum = result
    for name, value in (("normed", normed), ("pre_norm_sum", pre_norm_sum)):
        if tuple(value.shape) != tuple(x.shape):
            raise ValueError(f"{name} shape {tuple(value.shape)} != input shape {tuple(x.shape)}")
        if value.dtype != x.dtype:
            raise TypeError(f"{name} dtype {value.dtype} != input dtype {x.dtype}")
        if tuple(value.stride()) != tuple(x.stride()):
            raise ValueError(f"{name} stride {tuple(value.stride())} != {tuple(x.stride())}")
    input_storage = {_storage_ptr(x), _storage_ptr(residual), _storage_ptr(weight)}
    output_storage = {_storage_ptr(normed), _storage_ptr(pre_norm_sum)}
    if len(output_storage) != 2:
        raise RuntimeError("normed and pre_norm_sum must not alias each other")
    if input_storage.intersection(output_storage):
        raise RuntimeError("both outputs must be fresh and must not alias any input")
    return normed, pre_norm_sum


def call(args):
    """Call the selected live seam and enforce its full functional contract."""
    _validate_inputs(args)
    x, residual, weight = args["x"], args["residual"], args["weight"]
    verify_inputs = bool(args.get("verify_inputs", False))
    before = None
    if verify_inputs:
        before = (x.clone(), residual.clone(), weight.clone())
    result = _current_fn()(x, residual, weight, float(args["eps"]))
    result = _validate_outputs(result, x, residual, weight)
    if before is not None:
        if not x.equal(before[0]):
            raise RuntimeError("callable mutated x")
        if not residual.equal(before[1]):
            raise RuntimeError("callable mutated residual")
        if not weight.equal(before[2]):
            raise RuntimeError("callable mutated weight")
    return result


def _make_tensor(torch, shape, stride, rng, low, high):
    tensor = torch.empty_strided(
        tuple(int(v) for v in shape),
        tuple(int(v) for v in stride),
        dtype=torch.bfloat16,
        device="cuda",
    )
    tensor.uniform_(float(low), float(high), generator=rng)
    return tensor


def _make_inputs(h, spec, rng, verify_inputs):
    torch = h._torch()
    x = _make_tensor(torch, spec["x_shape"], spec["x_stride"], rng, -0.75, 0.875)
    residual = _make_tensor(
        torch, spec["residual_shape"], spec["residual_stride"], rng, -0.625, 0.5)
    weight = _make_tensor(
        torch, spec["weight_shape"], spec["weight_stride"], rng, -0.125, 0.125)
    x.reshape(-1)[0] = 0.75
    residual.reshape(-1)[0] = -0.25
    weight.reshape(-1)[0] = 0.0625
    actual = {
        "x": tuple(x.stride()),
        "residual": tuple(residual.stride()),
        "weight": tuple(weight.stride()),
    }
    expected = {
        "x": tuple(spec["x_stride"]),
        "residual": tuple(spec["residual_stride"]),
        "weight": tuple(spec["weight_stride"]),
    }
    if actual != expected:
        raise RuntimeError(f"failed to construct frozen strides: {actual} != {expected}")
    return {
        "x": x,
        "residual": residual,
        "weight": weight,
        "eps": float(spec["eps"]),
        "verify_inputs": bool(verify_inputs),
    }


def eager_cases(h, meta, baseline_outputs, seed=0):
    torch = h._torch()
    result = []
    for spec in meta["workload"]["cases"]:
        rng = torch.Generator(device="cuda").manual_seed(int(seed))
        args = _make_inputs(h, spec, rng, verify_inputs=True)
        key = f"{spec['sig']}|0"
        if key not in baseline_outputs:
            raise KeyError(f"baseline oracle is missing {key}")
        result.append({
            "args": args,
            "ref": h.to_device_like(baseline_outputs[key], "cuda"),
            "sig": spec["sig"],
            "regime": spec["regime"],
            "m": int(spec["m"]),
        })
    return result


def timing_cases(h, meta):
    torch = h._torch()
    result = []
    for offset, spec in enumerate(meta["workload"]["cases"]):
        rng = torch.Generator(device="cuda").manual_seed(31000 + offset)
        result.append({
            "sig": spec["sig"],
            "regime": spec["regime"],
            "m": int(spec["m"]),
            "args": _make_inputs(h, spec, rng, verify_inputs=False),
        })
    return result


def random_shapes(h, meta):
    result = []
    for spec in meta["workload"]["cases"]:
        frozen = dict(spec)
        result.append({
            "sig": frozen["sig"],
            "make_inputs": lambda rng, frozen=frozen: _make_inputs(
                h, frozen, rng, verify_inputs=True),
        })
    return result


def ordered_boundary_cases(eager):
    by_m = {int(case["m"]): case for case in eager}
    if len(by_m) < 2:
        raise RuntimeError("ordered boundary check needs two live M values")
    small = by_m[min(by_m)]
    large = by_m[max(by_m)]
    return [small, large, small]


def graph_replay_bundle(h, eager):
    """Capture at M=8192 and replay both M=8192 and an M=64 prefix."""
    torch = h._torch()
    replay_cases = sorted(eager, key=lambda case: int(case["m"]), reverse=True)
    largest = replay_cases[0]
    source = largest["args"]
    static = {
        "x": torch.empty_strided(
            source["x"].shape, source["x"].stride(), dtype=source["x"].dtype, device="cuda"),
        "residual": torch.empty_strided(
            source["residual"].shape,
            source["residual"].stride(),
            dtype=source["residual"].dtype,
            device="cuda",
        ),
        "weight": torch.empty_strided(
            source["weight"].shape,
            source["weight"].stride(),
            dtype=source["weight"].dtype,
            device="cuda",
        ),
        "eps": float(source["eps"]),
    }
    state = {"rows": int(source["x"].shape[0]), "outputs": None}
    fn = _current_fn()

    def fill(case):
        current = case["args"]
        rows = int(current["x"].shape[0])
        if rows > int(static["x"].shape[0]):
            raise ValueError(f"replay M={rows} exceeds capture M={static['x'].shape[0]}")
        if float(current["eps"]) != static["eps"]:
            raise ValueError("graph replay cases disagree on eps")
        static["x"].zero_()
        static["residual"].zero_()
        static["x"][:rows].copy_(current["x"])
        static["residual"][:rows].copy_(current["residual"])
        static["weight"].copy_(current["weight"])
        state["rows"] = rows

    def run():
        result = fn(static["x"], static["residual"], static["weight"], static["eps"])
        state["outputs"] = _validate_outputs(
            result, static["x"], static["residual"], static["weight"])

    def read_out():
        if state["outputs"] is None:
            raise RuntimeError("graph output is unavailable")
        rows = state["rows"]
        return tuple(value[:rows] for value in state["outputs"])

    return {
        "fill": fill,
        "run": run,
        "read_out": read_out,
        "cases": replay_cases,
        "capture_idx": 0,
    }
