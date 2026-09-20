"""Real-callable cases for the SGLang packed-decode gated-delta kernel."""

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


def _generated_blob(device="cuda", seed=0):
    """Generate numeric values while retaining the frozen structural contract."""
    import importlib.util
    from pathlib import Path
    import sys
    torch = importlib.import_module("torch")
    path = Path(HERE) / "generated_contract.py"
    module = sys.modules.get("generated_contract")
    if module is None:
        spec = importlib.util.spec_from_file_location("generated_contract", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_contract"] = module
        spec.loader.exec_module(module)
    elif Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError("generated contract alias resolves outside this task")
    return module.build_blob(module.load_contract(HERE), seed, torch, device)


def _current_fn():
    global _RESOLVED
    if _RESOLVED is None:
        _RESOLVED = _resolve(META["target_callable"])
    return _RESOLVED


def _payload(args):
    if not isinstance(args, dict) or "kwargs" not in args:
        raise TypeError("gated-delta case args must contain a kwargs mapping")
    return args["kwargs"], bool(args.get("fresh", True))


def call(args):
    """Invoke the live seam and expose its complete in-place return contract.

    Correctness cases clone the two mutated buffers before every call, which makes
    consecutive results independent while preserving the callable's real aliasing
    contract. Timing cases reuse fixed buffers so cloning is outside the timed op.
    """
    source, fresh = _payload(args)
    kwargs = dict(source)
    if fresh:
        kwargs["initial_state"] = source["initial_state"].clone()
        kwargs["out"] = source["out"].clone()
    result = _current_fn()(**kwargs)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("live gated-delta callable must return (out, initial_state)")
    out, state = result
    if out.data_ptr() != kwargs["out"].data_ptr():
        raise RuntimeError("callable returned an output that does not alias the supplied out buffer")
    if state.data_ptr() != kwargs["initial_state"].data_ptr():
        raise RuntimeError("callable returned a state that does not alias the supplied initial_state buffer")
    return out, state


def _restore(h, obj, shared, device):
    return h.reconstruct_captured(h.resolve_oracle_shared(obj, shared), device=device)


def eager_cases(h, meta, device="cuda", seed=0):
    blob = _generated_blob(device, seed)
    shared = blob.get("shared") or {}
    result = []
    for record in blob.get("records") or []:
        raw_kwargs = dict(record.get("kwargs") or {})
        raw_kwargs.pop("initial_state", None)
        raw_kwargs.pop("out", None)
        kwargs = _restore(h, raw_kwargs, shared, device)
        before = _restore(h, record.get("kwargs_before") or {}, shared, device)
        if set(before) != {"initial_state", "out"}:
            raise RuntimeError(
                "oracle must contain pre-call snapshots for exactly initial_state and out")
        kwargs.update(before)
        ref = _restore(h, record.get("output"), shared, device)
        batch = int(kwargs["mixed_qkv"].shape[0])
        result.append({
            "args": {"kwargs": kwargs, "fresh": True},
            "ref": ref,
            "sig": f"decode_b{batch}_live",
            "regime": "decode",
            "m": batch,
        })
    if len(result) < 2:
        raise RuntimeError("hk10 requires at least two live boundary cases")
    return sorted(result, key=lambda case: int(case["m"]), reverse=True)


def _randn(torch, shape, rng, dtype, scale=1.0):
    value = torch.randn(*shape, generator=rng, dtype=torch.float32, device="cuda")
    return (value * float(scale)).to(dtype)


def _synthetic_kwargs(h, spec, rng):
    torch = h._torch()
    geo = META["geometry"]
    batch = int(spec["B"])
    hv, k, v = int(geo["HV"]), int(geo["K"]), int(geo["V"])
    pool = int(geo["state_pool_slots"])
    qkv_dim = int(geo["qkv_dim"])
    indices = [int(value) for value in spec["state_indices"]]
    if len(indices) != batch or len(set(indices)) != batch:
        raise ValueError(f"{spec['sig']}: state indices must be unique and match B")
    if min(indices) < 0 or max(indices) >= pool:
        raise ValueError(f"{spec['sig']}: state index outside the captured pool")

    A = 1.0 + 15.0 * torch.rand(hv, generator=rng, dtype=torch.float32, device="cuda")
    dt = 1e-3 + (1e-1 - 1e-3) * torch.rand(
        hv, generator=rng, dtype=torch.float32, device="cuda")
    return {
        "mixed_qkv": _randn(torch, (batch, qkv_dim), rng, torch.bfloat16, 0.5),
        "a": _randn(torch, (batch, hv), rng, torch.bfloat16, 0.5),
        "b": _randn(torch, (batch, hv), rng, torch.bfloat16, 0.5),
        "A_log": torch.log(A),
        "dt_bias": torch.log(torch.expm1(dt)).to(torch.bfloat16),
        "scale": float(geo["scale"]),
        "initial_state": _randn(torch, (pool, hv, v, k), rng, torch.float32, 0.05),
        "out": _randn(torch, (batch, 1, hv, v), rng, torch.bfloat16, 0.1),
        "ssm_state_indices": torch.tensor(indices, dtype=torch.int32, device="cuda"),
        "use_qk_l2norm_in_kernel": bool(geo["use_qk_l2norm_in_kernel"]),
    }


def timing_cases(h, meta):
    torch = h._torch()
    result = []
    for offset, spec in enumerate(meta["workload"]["cases"]):
        rng = torch.Generator(device="cuda").manual_seed(31000 + offset)
        result.append({
            "sig": spec["sig"],
            "regime": spec["regime"],
            "m": int(spec["m"]),
            "args": {"kwargs": _synthetic_kwargs(h, spec, rng), "fresh": False},
        })
    return result


def random_shapes(h, meta):
    result = []
    for spec in meta["workload"]["cases"]:
        frozen = dict(spec)
        frozen["state_indices"] = list(spec["state_indices"])
        result.append({
            "sig": frozen["sig"],
            "make_inputs": lambda rng, frozen=frozen: {
                "kwargs": _synthetic_kwargs(h, frozen, rng),
                "fresh": True,
            },
        })
    return result


def ordered_boundary_cases(eager):
    by_batch = {int(case["m"]): case for case in eager}
    batches = sorted(by_batch)
    if len(batches) < 2:
        raise RuntimeError("ordered boundary check needs at least two batch sizes")
    small, large = by_batch[batches[0]], by_batch[batches[-1]]
    return [small, large, small]


def graph_replay_bundle(h, eager):
    """Capture at B=max and replay the real B=max/B=min cases in one static buffer."""
    torch = h._torch()
    replay_cases = sorted(eager, key=lambda case: int(case["m"]), reverse=True)
    largest = replay_cases[0]
    static = {
        key: value.clone() if hasattr(value, "clone") else value
        for key, value in largest["args"]["kwargs"].items()
    }
    max_batch = int(static["mixed_qkv"].shape[0])
    active = {"batch": max_batch}
    fn = _current_fn()

    def fill(case):
        source = case["args"]["kwargs"]
        batch = int(source["mixed_qkv"].shape[0])
        if batch > max_batch:
            raise ValueError(f"replay case B={batch} exceeds capture B={max_batch}")
        if source["scale"] != static["scale"]:
            raise ValueError("graph replay cases disagree on scale")
        if source["use_qk_l2norm_in_kernel"] != static["use_qk_l2norm_in_kernel"]:
            raise ValueError("graph replay cases disagree on L2-normalization mode")

        static["mixed_qkv"].zero_()
        static["a"].zero_()
        static["b"].zero_()
        static["out"].zero_()
        static["ssm_state_indices"].fill_(-1)
        static["mixed_qkv"][:batch].copy_(source["mixed_qkv"])
        static["a"][:batch].copy_(source["a"])
        static["b"][:batch].copy_(source["b"])
        static["out"][:batch].copy_(source["out"])
        static["ssm_state_indices"][:batch].copy_(source["ssm_state_indices"])
        static["A_log"].copy_(source["A_log"])
        static["dt_bias"].copy_(source["dt_bias"])
        static["initial_state"].copy_(source["initial_state"])
        active["batch"] = batch

    def run():
        fn(**static)

    def read_out():
        batch = active["batch"]
        return static["out"][:batch], static["initial_state"]

    return {
        "fill": fill,
        "run": run,
        "read_out": read_out,
        "cases": replay_cases,
        "capture_idx": 0,
    }
