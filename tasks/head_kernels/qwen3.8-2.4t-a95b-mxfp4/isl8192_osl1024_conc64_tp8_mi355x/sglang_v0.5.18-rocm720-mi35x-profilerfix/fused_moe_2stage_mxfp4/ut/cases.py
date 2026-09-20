"""Standalone cases for the live AITER compact-MXFP4 fused-MoE callable."""

import ast
import importlib
import json
import os
import re


HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "meta.json")) as _fh:
    META = json.load(_fh)

_RESOLVED = None
_LIVE_CASES = None



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


def _restore_repr(value):
    match = re.fullmatch(r"<(ActivationType|QuantType)\.([^:>]+):\s*[^>]+>", value)
    if match:
        module = importlib.import_module("aiter.fused_moe")
        return getattr(getattr(module, match.group(1)), match.group(2))
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise TypeError(f"unsupported captured repr {value!r}") from exc


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
        return _restore_repr(obj["__repr__"])
    if isinstance(obj, dict):
        return {key: _restore(value, shared, device) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_restore(value, shared, device) for value in obj)
    if isinstance(obj, list):
        return [_restore(value, shared, device) for value in obj]
    return obj


def load_live_cases(device="cuda", seed=0):
    global _LIVE_CASES, _GENERATED_KEY
    if _LIVE_CASES is not None and _GENERATED_KEY == (str(device), int(seed)):
        return _LIVE_CASES
    torch = importlib.import_module("torch")
    blob = _generated_blob(device, seed)
    _GENERATED_KEY = (str(device), int(seed))
    shared = {
        key: _restore(value, {}, device)
        for key, value in (blob.get("shared") or {}).items()
    }
    for key in ("w1", "w2"):
        if key not in shared:
            raise RuntimeError(f"live MoE oracle is missing shared {key}")
        shared[key].is_shuffled = True
    cases = {}
    for record in blob.get("records") or []:
        kwargs = _restore(record.get("kwargs") or {}, shared, device)
        ref = _restore(record.get("output"), shared, device)
        m = int(kwargs["hidden_states"].shape[0])
        regime = record.get("regime") or ("decode" if m <= 256 else "prefill")
        if (regime, m) in cases:
            raise RuntimeError(f"duplicate live MoE case: {regime}, M={m}")
        cases[(regime, m)] = {
            "args": {"kwargs": kwargs},
            "ref": ref,
            "sig": f"{regime}_m{m}_live",
            "regime": regime,
            "m": m,
        }
    expected = {("decode", 64), ("prefill", 8192)}
    if set(cases) != expected:
        raise RuntimeError(f"unexpected live MoE cases: {sorted(cases)}")
    _LIVE_CASES = cases
    return cases


def selected_live_case(case_id, device="cuda"):
    contract = META["case_contracts"][case_id]
    return load_live_cases(device)[(contract["regime"], int(contract["m"]))]


def _payload(args):
    if not isinstance(args, dict) or "kwargs" not in args:
        raise TypeError("MoE args must contain a kwargs mapping")
    return args["kwargs"]


def call(args):
    return current_callable()(**dict(_payload(args)))


def baseline_call(args):
    return baseline_callable()(**dict(_payload(args)))


def _make_hidden(template, rng):
    torch = importlib.import_module("torch")
    hidden = torch.empty_like(template)
    hidden.uniform_(-0.75, 0.875, generator=rng)
    hidden.reshape(-1)[0] = 0.5
    return hidden


def random_shapes(case):
    template = case["args"]["kwargs"]
    sig = case["sig"]

    def make_inputs(rng):
        kwargs = dict(template)
        kwargs["hidden_states"] = _make_hidden(template["hidden_states"], rng)
        return {"kwargs": kwargs}

    return [{"sig": sig, "make_inputs": make_inputs}]


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
    template = case["args"]["kwargs"]
    replay_cases = []
    for index, seed in enumerate((9011, 9012)):
        rng = torch.Generator(device="cuda").manual_seed(seed)
        kwargs = dict(template)
        kwargs["hidden_states"] = _make_hidden(template["hidden_states"], rng)
        args = {"kwargs": kwargs}
        ref = baseline_call(args).detach().clone()
        replay_cases.append({
            "args": args,
            "ref": ref,
            "sig": f"{case['sig']}_value{index}",
        })
    if torch.equal(
        replay_cases[0]["args"]["kwargs"]["hidden_states"],
        replay_cases[1]["args"]["kwargs"]["hidden_states"],
    ):
        raise RuntimeError("graph replay inputs must differ")
    static_kwargs = dict(template)
    static_kwargs["hidden_states"] = torch.empty_like(template["hidden_states"])
    state = {"output": None}

    def fill(item):
        static_kwargs["hidden_states"].copy_(
            item["args"]["kwargs"]["hidden_states"]
        )

    def run():
        state["output"] = current_callable()(**static_kwargs)

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


def _snapshot_tensor(torch, tensor):
    if str(tensor.dtype).startswith("torch.float4_"):
        return tensor.view(torch.uint8).clone()
    return tensor.clone()


def _tensor_unchanged(torch, tensor, snapshot):
    current = tensor.view(torch.uint8) if str(tensor.dtype).startswith("torch.float4_") else tensor
    return torch.equal(current, snapshot)


def callable_contract(h, case):
    torch = h._torch()
    kwargs = case["args"]["kwargs"]
    snapshots = {
        key: _snapshot_tensor(torch, value)
        for key, value in kwargs.items()
        if torch.is_tensor(value)
    }
    first = call(case["args"])
    first_copy = first.clone()
    second = call(case["args"])
    unchanged = {
        key: _tensor_unchanged(torch, kwargs[key], value)
        for key, value in snapshots.items()
    }
    input_ptrs = {
        value.untyped_storage().data_ptr()
        for value in kwargs.values()
        if torch.is_tensor(value)
    }
    fresh = (
        first.untyped_storage().data_ptr() not in input_ptrs
        and second.untyped_storage().data_ptr() not in input_ptrs
        and first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    )
    independent = torch.equal(first, first_copy)
    shape_ok = tuple(first.shape) == tuple(kwargs["hidden_states"].shape)
    dtype_ok = first.dtype == kwargs["hidden_states"].dtype
    ok = all(unchanged.values()) and fresh and independent and shape_ok and dtype_ok
    return bool(ok), {
        "correct": bool(ok),
        "inputs_unchanged": unchanged,
        "fresh_outputs": bool(fresh),
        "prior_output_unchanged": bool(independent),
        "output_shape": list(first.shape),
        "output_dtype": str(first.dtype),
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


def profile_engagement(h, case, required_fragments):
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
        fragment: sorted({row["name"] for row in rows if fragment.lower() in row["name"].lower()})
        for fragment in required_fragments
    }
    ok = all(matched[fragment] for fragment in required_fragments)
    return bool(ok), {
        "correct": bool(ok),
        "required_fragments": list(required_fragments),
        "matched": matched,
        "device_event_count": len(rows),
    }


def timing_smoke(h, case):
    base = h.time_op(
        lambda: baseline_call(case["args"]),
        warmup=1,
        repeats=2,
        inner=1,
        graph=False,
        flush_cache=False,
        detail=True,
    )
    cand = h.time_op(
        lambda: call(case["args"]),
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
