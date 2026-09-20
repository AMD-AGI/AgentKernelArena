"""Standalone cases for the live AITER ragged paged-attention callable."""

import ast
import importlib
import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "meta.json")) as _fh:
    META = json.load(_fh)

_RESOLVED = None
_LIVE_CASE = None



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


def _load_frozen_capture(torch, path, **kwargs):
    import importlib.util
    from pathlib import Path
    helper = Path(__file__).resolve().with_name("task_contract.py")
    spec = importlib.util.spec_from_file_location("_frozen_capture_loader", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verified_torch_load(torch, path, **kwargs)


def _resolve(target):
    module_name, _, attr_path = target.partition(":")
    obj = importlib.import_module(module_name)
    for part in attr_path.split("."):
        if part:
            obj = getattr(obj, part)
    return obj


def current_callable():
    global _RESOLVED
    if _RESOLVED is None:
        _RESOLVED = _resolve(META["target_callable"])
    return _RESOLVED


def baseline_callable():
    # The immutable overlay captures this before loading any editable candidate code.
    registry = importlib.import_module("_aka_frozen_baseline_bindings")
    return registry.bindings[META["candidate_bind"]["target"]]


def _torch_dtype(torch, value):
    name = str(value)
    if name.startswith("torch."):
        name = name.split(".", 1)[1]
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported captured dtype {value!r}")
    return dtype


def _restore(obj, shared, device):
    torch = importlib.import_module("torch")
    if isinstance(obj, dict) and set(obj) == {"__shared__"}:
        return shared[obj["__shared__"]]
    if isinstance(obj, dict) and obj.get("__tensor__"):
        import harness_lib
        return harness_lib.capture_contract().restore_tensor(torch, obj, device)
    if isinstance(obj, dict) and obj.get("__tensor_factory__"):
        return torch.empty_strided(
            tuple(int(v) for v in obj["shape"]),
            tuple(int(v) for v in obj["stride"]),
            dtype=_torch_dtype(torch, obj["dtype"]),
            device=device,
        )
    if isinstance(obj, dict) and "__enum__" in obj:
        module_name, _, qualname = obj["__enum__"].partition(":")
        enum_type = importlib.import_module(module_name)
        for part in qualname.split("."):
            enum_type = getattr(enum_type, part)
        return enum_type[obj["name"]]
    if isinstance(obj, dict) and "__torch_dtype__" in obj:
        return _torch_dtype(torch, obj["__torch_dtype__"])
    if isinstance(obj, dict) and "__repr__" in obj:
        try:
            return ast.literal_eval(obj["__repr__"])
        except (SyntaxError, ValueError) as exc:
            raise TypeError(f"unsupported captured repr {obj['__repr__']!r}") from exc
    if isinstance(obj, dict):
        return {key: _restore(value, shared, device) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_restore(value, shared, device) for value in obj)
    if isinstance(obj, list):
        return [_restore(value, shared, device) for value in obj]
    return obj


def _empty_like_strided(torch, tensor):
    return torch.empty_strided(
        tuple(tensor.shape), tuple(tensor.stride()), dtype=tensor.dtype, device=tensor.device
    )


def load_live_case(device="cuda", seed=0):
    global _LIVE_CASE, _GENERATED_KEY
    if _LIVE_CASE is not None and _GENERATED_KEY == (str(device), int(seed)):
        return _LIVE_CASE
    torch = importlib.import_module("torch")
    blob = _generated_blob(device, seed)
    _GENERATED_KEY = (str(device), int(seed))
    records = blob.get("records") or []
    if len(records) != 1:
        raise RuntimeError(f"expected one live attention record, found {len(records)}")
    shared = {
        key: _restore(value, {}, device)
        for key, value in (blob.get("shared") or {}).items()
    }
    record = records[0]
    args = tuple(_restore(record.get("args") or (), shared, device))
    ref = _restore(record.get("output"), shared, device)
    if len(args) != 19:
        raise RuntimeError(f"expected 19 positional attention arguments, found {len(args)}")
    if tuple(args[2].shape) != (64, 8, 256):
        raise RuntimeError(f"unexpected live query shape {tuple(args[2].shape)}")
    if int(args[6][-1].item()) != int(args[7].numel()):
        raise RuntimeError("compacted kv_page_indices must be fully referenced")
    _LIVE_CASE = {
        "args": {"positional": args, "fresh": True},
        "ref": ref,
        "sig": "decode_m64_live",
        "regime": "decode",
        "m": 64,
    }
    return _LIVE_CASE


def _payload(payload, fresh_override=None):
    if not isinstance(payload, dict) or "positional" not in payload:
        raise TypeError("attention args must contain positional operands")
    source = tuple(payload["positional"])
    fresh = bool(payload.get("fresh", True)) if fresh_override is None else bool(fresh_override)
    if not fresh:
        return source
    torch = importlib.import_module("torch")
    values = list(source)
    values[0] = _empty_like_strided(torch, source[0])
    values[1] = _empty_like_strided(torch, source[1])
    return tuple(values)


def _invoke(fn, payload):
    args = _payload(payload)
    out = fn(*args)
    if out.untyped_storage().data_ptr() != args[0].untyped_storage().data_ptr():
        raise RuntimeError("paged_attention_ragged must return the supplied out buffer")
    return out


def call(payload):
    return _invoke(current_callable(), payload)


def baseline_call(payload):
    return _invoke(baseline_callable(), payload)


def _make_query(template, rng):
    torch = importlib.import_module("torch")
    query = torch.empty_like(template)
    query.uniform_(-0.75, 0.875, generator=rng)
    query.reshape(-1)[0] = 0.5
    return query


def random_shapes(case):
    template = tuple(case["args"]["positional"])

    def make_inputs(rng):
        values = list(template)
        values[2] = _make_query(template[2], rng)
        return {"positional": tuple(values), "fresh": True}

    return [{"sig": case["sig"], "make_inputs": make_inputs}]


def baseline_random_outputs(case, draws, seed=0):
    torch = importlib.import_module("torch")
    outputs = {}
    for shape in random_shapes(case):
        for draw in range(max(1, int(draws))):
            rng = torch.Generator(device="cuda").manual_seed(int(seed) + draw)
            args = shape["make_inputs"](rng)
            out = baseline_call(args)
            outputs[f"{shape['sig']}|{draw}"] = out.detach().cpu().clone()
            del out, args
    return outputs


def graph_replay_bundle(h, case):
    torch = h._torch()
    template = tuple(case["args"]["positional"])
    replay_cases = []
    for index, seed in enumerate((9101, 9102)):
        rng = torch.Generator(device="cuda").manual_seed(seed)
        values = list(template)
        values[2] = _make_query(template[2], rng)
        payload = {"positional": tuple(values), "fresh": True}
        ref = baseline_call(payload).detach().clone()
        replay_cases.append({
            "args": payload,
            "ref": ref,
            "sig": f"{case['sig']}_value{index}",
        })
    if torch.equal(
        replay_cases[0]["args"]["positional"][2],
        replay_cases[1]["args"]["positional"][2],
    ):
        raise RuntimeError("graph replay queries must differ")
    static = list(template)
    static[0] = _empty_like_strided(torch, template[0])
    static[1] = _empty_like_strided(torch, template[1])
    static[2] = _empty_like_strided(torch, template[2])
    state = {"output": None}

    def fill(item):
        source = item["args"]["positional"]
        static[0].zero_()
        static[1].zero_()
        static[2].copy_(source[2])

    def run():
        state["output"] = current_callable()(*static)

    def read_out():
        if state["output"] is None:
            raise RuntimeError("graph output is unavailable")
        return state["output"]

    return {
        "fill": fill,
        "run": run,
        "read_out": read_out,
        "cases": replay_cases,
        "capture_idx": 0,
    }


def callable_contract(h, case):
    torch = h._torch()
    args = tuple(case["args"]["positional"])
    immutable_indices = [2, 3, 4, 6, 7, 8, 15, 16]
    snapshots = {index: args[index].clone() for index in immutable_indices}
    payload = {"positional": args, "fresh": False}
    out = call(payload)
    unchanged = {
        str(index): torch.equal(args[index], before)
        for index, before in snapshots.items()
    }
    alias_ok = out.untyped_storage().data_ptr() == args[0].untyped_storage().data_ptr()
    shape_ok = tuple(out.shape) == tuple(args[2].shape)
    dtype_ok = out.dtype == args[2].dtype
    ok = all(unchanged.values()) and alias_ok and shape_ok and dtype_ok
    return bool(ok), {
        "correct": bool(ok),
        "immutable_inputs_unchanged": unchanged,
        "returns_supplied_out": bool(alias_ok),
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
    }


def negative_check(h, case, tol):
    def corrupt(args):
        out = call(args).clone()
        out.reshape(-1)[0] += 1024
        return out

    accepted, report = h.check_correct_multi(corrupt, [case], tol)
    return (not accepted), {
        "contract": "negative_control",
        "output_corruption_rejected": not accepted,
        "report": report,
    }


def profile_engagement(h, case, required_kernels):
    torch = h._torch()
    call(case["args"])
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        call(case["args"])
        torch.cuda.synchronize()
    rows = []
    for event in prof.key_averages():
        name = str(getattr(event, "key", ""))
        device_us = float(
            getattr(event, "self_device_time_total", 0.0)
            or getattr(event, "device_time_total", 0.0)
            or 0.0
        )
        if device_us > 0:
            rows.append({"name": name, "device_time_us": device_us})
    matched = {
        kernel: sorted({row["name"] for row in rows if kernel.lower() in row["name"].lower()})
        for kernel in required_kernels
    }
    ok = all(matched[kernel] for kernel in required_kernels)
    return bool(ok), {
        "correct": bool(ok),
        "required_kernels": list(required_kernels),
        "matched": matched,
        "device_event_count": len(rows),
    }


def timing_smoke(h, case):
    timing_args = dict(case["args"])
    timing_args["fresh"] = False
    base = h.time_op(
        lambda: baseline_call(timing_args),
        warmup=1,
        repeats=2,
        inner=1,
        graph=False,
        flush_cache=False,
        detail=True,
    )
    cand = h.time_op(
        lambda: call(timing_args),
        warmup=1,
        repeats=2,
        inner=1,
        graph=False,
        flush_cache=False,
        detail=True,
    )
    baseline_ms = (base or {}).get("ms")
    candidate_ms = (cand or {}).get("ms")
    ok = bool(baseline_ms and candidate_ms)
    return ok, {
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "baseline_timer": (base or {}).get("timer"),
        "candidate_timer": (cand or {}).get("timer"),
        "speedup": baseline_ms / candidate_ms if ok else None,
        "purpose": "smoke_only",
    }
