"""Protected decoding of the original captured MoE fixture; no candidate imports."""
import ast
import importlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "meta.json")) as stream:
    META = json.load(stream)
_LIVE_CASES = None

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
        data = obj["data"].to(device)
        packed = obj.get("packed_view_dtype") or ""
        if packed:
            value = data.view(_torch_dtype(torch, packed))
        else:
            value = data
        shape = tuple(int(v) for v in obj["shape"])
        stride = tuple(int(v) for v in obj["stride"])
        if tuple(value.shape) != shape:
            value = value.reshape(shape)
        if tuple(value.stride()) != stride:
            exact = torch.empty_strided(shape, stride, dtype=value.dtype, device=device)
            exact.copy_(value)
            value = exact
        return value
    if isinstance(obj, dict) and obj.get("__tensor_factory__"):
        raise ValueError("captured MoE input cannot be an uninitialized tensor factory")
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

def load_live_cases(device="cuda"):
    global _LIVE_CASES
    if _LIVE_CASES is not None:
        return _LIVE_CASES
    torch = importlib.import_module("torch")
    blob = torch.load(
        os.path.join(HERE, META["reference_io"]),
        map_location="cpu",
        weights_only=False,
    )
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
            raise ValueError("duplicate captured MoE case")
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
