"""Local finite generators at the exact captured DeepSeek structural contracts."""

from __future__ import annotations
import ast
import base64
import hashlib
import importlib
import json
import math
from pathlib import Path
import zlib

_STATE = {"seed": 1234}
_STORAGE = {}
_KV = {}


def set_seed(seed):
    """Trusted runtime state may change; public immutable constants may not."""
    _STATE["seed"] = int(seed)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract(ut):
    ut = Path(ut)
    meta = json.loads((ut / "meta.json").read_text())
    settings = meta["generated_inputs"]
    path = ut / settings["contract_file"]
    if digest(path) != settings["contract_sha256"]:
        raise RuntimeError("generated structural contract hash mismatch")
    data = json.loads(path.read_text())
    if (
        data.get("schema_version") != 1
        or len(data["records"]) != meta["num_cases"]
        or data["source_reference_sha256"]
        != meta["archival_capture"]["reference_io_sha256"]
    ):
        raise RuntimeError("generated contract changed the captured case set")
    expected = settings["record_signatures"]
    if [row["sig"] for row in data["records"]] != expected or len(set(expected)) != len(
        expected
    ):
        raise RuntimeError("missing, reordered or duplicate generated case")
    requirement = meta.get("sequence_requirement")
    if requirement:
        evidence_path = ut / requirement["coverage_evidence"]
        if digest(evidence_path) != requirement["coverage_evidence_sha256"]:
            raise RuntimeError("sequence coverage evidence hash mismatch")
        evidence = json.loads(evidence_path.read_text())
        if (
            requirement["required_calls"] != evidence["required_sequence_calls"]
            or requirement["call_sequence_sha256"] != evidence["call_sequence_sha256"]
        ):
            raise RuntimeError("mandatory call-sequence requirement changed")
    return data


def dtype(torch, name):
    result = getattr(torch, str(name).removeprefix("torch."), None)
    if result is None or not isinstance(result, torch.dtype):
        raise ValueError("unsupported captured dtype: " + str(name))
    return result


def decode_bytes(node):
    if node["codec"] != "zlib-base64":
        raise ValueError("unknown structural codec")
    expected = int(node["bytes"])
    if not 0 <= expected <= 1024 << 20:
        raise ValueError("structural data exceeds reviewed bound")
    encoded = base64.b64decode(node["data"], validate=True)
    stream = zlib.decompressobj()
    raw = stream.decompress(encoded, expected + 1)
    if (
        len(raw) != expected
        or not stream.eof
        or stream.unused_data
        or stream.unconsumed_tail
    ):
        raise ValueError("structural payload length mismatch")
    return raw


def _generator(torch, device, key):
    seed = int.from_bytes(
        hashlib.sha256((str(_STATE["seed"]) + "|" + str(key)).encode()).digest()[:8],
        "little",
    ) % (2**63)
    return torch.Generator(device=device).manual_seed(seed)


def _fill_bytes(raw, recipe, name, torch, generator, logical_dtype):
    """Never cast arbitrary bytes to FP8/E8M0: choose finite normal encodings."""
    if recipe in ("zero_output", "captured_structure"):
        raw.zero_()
        return
    if recipe == "generated_mxfp4_weight":
        # Every E2M1 nibble is finite; packing two independent nibbles is valid.
        raw.random_(0, 256, generator=generator)
        return
    if recipe == "generated_block_scale":
        if logical_dtype in ("torch.float8_e8m0fnu", "torch.uint8"):
            if name in ("a1_scale", "a2_scale"):
                # A uniform finite activation scale remains coherent when the
                # same token is repeated in several sorted expert slots.
                raw.fill_(123)
            else:
                raw.random_(121, 125, generator=generator)
        else:
            view = raw.view(dtype(torch, logical_dtype))
            view.copy_(
                torch.randint(
                    -6, -2, view.shape, device=view.device, generator=generator
                )
                .float()
                .exp2()
            )
        return
    if logical_dtype in ("torch.float8_e4m3fn", "torch.float8_e4m3fnuz", "torch.uint8"):
        # Normal E4M3 magnitudes 0x28..0x3f; no subnormal, infinity or NaN.
        raw.random_(0x28, 0x40, generator=generator)
        signs = torch.randint(
            0, 2, raw.shape, dtype=torch.uint8, device=raw.device, generator=generator
        )
        raw.bitwise_or_(signs * 128)
        return
    view = raw.view(dtype(torch, logical_dtype))
    view.normal_(mean=0, std=0.125, generator=generator)


def build_tensor(node, name, torch, device):
    dt = dtype(torch, node["dtype"])
    key = (str(device), _STATE["seed"], node["storage_group"])
    needed = int(node["storage_bytes"])
    if key not in _STORAGE:
        raw = torch.empty(needed, dtype=torch.uint8, device=device)
        _fill_bytes(
            raw,
            node["recipe"],
            name,
            torch,
            _generator(torch, device, key),
            node["dtype"],
        )
        _STORAGE[key] = raw
    raw = _STORAGE[key]
    if raw.numel() != needed:
        raise ValueError("inconsistent captured alias extent")
    value = torch.empty(0, dtype=dt, device=device)
    value.set_(
        raw.untyped_storage(),
        int(node["storage_offset"]),
        tuple(node["shape"]),
        tuple(node["stride"]),
    )
    if node["recipe"] == "captured_structure":
        payload = decode_bytes(node["payload"])
        cpu = torch.frombuffer(bytearray(payload), dtype=dt).reshape(node["shape"])
        value.copy_(cpu.to(device))
    elif node["recipe"] == "zero_output":
        value.zero_()
    for attr, val in node.get("tensor_attrs", {}).items():
        setattr(value, attr, val)
    return value


def build_kv(node, key, torch, device):
    """Full-size paged storage; block IDs and cache stride are never compacted."""
    shape = tuple(node["shape"])
    stride0 = int(node["stride0"])
    page = shape[1]
    if shape[2:] != (1, 584) or stride0 < page * 584 or stride0 % 4:
        raise ValueError("unsupported captured DSA combined-cache layout")
    cache_key = (str(device), _STATE["seed"], key)
    if cache_key in _KV:
        return _KV[cache_key]
    raw = torch.zeros((shape[0], stride0), dtype=torch.uint8, device=device)
    rows = build_tensor(node["rows_idx"], "rows_idx", torch, device).long()
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= shape[0]):
        raise ValueError("captured DSA block ID outside full cache")
    generator = _generator(torch, device, cache_key)
    # Bound temporary generation memory while retaining every referenced page.
    for start in range(0, rows.numel(), 32):
        ids = rows[start : start + 32]
        count = ids.numel()
        block = torch.zeros((count, stride0), dtype=torch.uint8, device=device)
        payload = block[:, : page * 576].view(count, page, 576)
        nope = torch.randint(
            0x28,
            0x40,
            (count, page, 448),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        signs = torch.randint(
            0, 2, nope.shape, dtype=torch.uint8, device=device, generator=generator
        )
        payload[:, :, :448] = nope.bitwise_or(signs * 128)
        rope = (
            torch.randn(
                count, page, 64, dtype=torch.float32, device=device, generator=generator
            )
            * 0.125
        ).to(torch.bfloat16)
        payload[:, :, 448:] = rope.view(torch.uint8)
        scales = block[:, page * 576 : page * 584].view(count, page, 8)
        scales[:, :, :7] = torch.randint(
            121,
            125,
            (count, page, 7),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        scales[:, :, 7] = 127  # unused byte in each 8-byte scale record
        raw[ids] = block
    logical = raw.view(dtype(torch, node["dtype"])).as_strided(
        shape, (stride0, 584, 584, 1)
    )
    _KV[cache_key] = logical
    return logical


def scalar(rep, torch, enum_modules):
    if rep.startswith("torch."):
        return dtype(torch, rep)
    try:
        return ast.literal_eval(rep)
    except (ValueError, SyntaxError):
        pass
    import re

    match = re.match(r"^<?([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", rep)
    if match:
        for name in enum_modules:
            module = importlib.import_module(name)
            cls = getattr(module, match[1], None)
            if cls is not None and hasattr(cls, match[2]):
                return getattr(cls, match[2])
    raise ValueError("unsupported recorded scalar " + rep)


def build_record(record, meta, torch, device="cuda"):
    def build(node, name):
        if isinstance(node, dict) and node.get("tensor"):
            return build_tensor(node, name, torch, device)
        if isinstance(node, dict) and node.get("dsa_kv"):
            return build_kv(node, record["sig"] + "|" + name, torch, device)
        if isinstance(node, dict) and "repr" in node:
            return scalar(node["repr"], torch, meta.get("enum_modules", []))
        if isinstance(node, dict) and "sequence" in node:
            vals = [build(item, name) for item in node["items"]]
            return tuple(vals) if node["sequence"] == "tuple" else vals
        if isinstance(node, dict):
            return {k: build(v, k) for k, v in node.items()}
        return node

    args = [build(node, str(index)) for index, node in enumerate(record["args"])]
    kwargs = {name: build(node, name) for name, node in record["kwargs"].items()}
    for key, attrs in meta.get("tensor_attrs", {}).items():
        target = args[int(key)] if key.lstrip("-").isdigit() else kwargs[key]
        for name, value in attrs.items():
            setattr(target, name, value)
    return args, kwargs


def undefined_mask(record, torch, device, m=None):
    rows = record.get("undefined_rows", [])
    partial = record.get("undefined_partial_rows", [])
    if not rows and not partial:
        return None
    full_shape = list(record["undefined_shape"])
    shape = list(full_shape)
    shape[0] = min(shape[0], int(m or shape[0]))
    selected = [row for row in rows if row < shape[0]]
    pairs = [(index, row) for index, row in enumerate(partial) if row < shape[0]]
    if not selected and not pairs:
        return None
    value = torch.zeros(shape, dtype=torch.bool, device=device)
    value[selected] = True
    if pairs:
        raw = decode_bytes(record["undefined_partial_payload"])
        masks = torch.frombuffer(bytearray(raw), dtype=torch.bool).reshape(
            len(partial), *full_shape[1:]
        )
        if len(pairs) != len(partial):
            positions = torch.tensor([index for index, _ in pairs], dtype=torch.int64)
            masks = masks.index_select(0, positions)
        row_ids = torch.tensor(
            [row for _, row in pairs], dtype=torch.int64, device=device
        )
        value[row_ids] = masks.to(device)
    return value


def profiles(meta):
    recorded = meta["generated_inputs"]["record_signatures"]
    if "case_specs" in meta:
        eager = [row["name"] for row in meta["case_specs"]]
        random = eager + [
            "replay_shortctx:" + row["name"]
            for row in meta["case_specs"]
            if row.get("replay")
        ]
        replay = []
        for row in meta["case_specs"]:
            if row.get("replay"):
                replay = [
                    "replay_recorded_idx:" + row["name"],
                    "replay_shortctx_idx:" + row["name"],
                ]
        sequence = list(meta["call_sequence"])
    else:
        eager = list(recorded)
        random = list(recorded)
        gr = meta["graph_replay"]
        replay = [gr["capture_sig"], gr["second_sig"]]
        sequence = [
            row["sig"] if isinstance(row, dict) else row
            for row in meta.get("call_sequence", [])
        ]
    if len(replay) != 2 or any(
        not isinstance(x, str) for x in eager + random + sequence + replay
    ):
        raise ValueError("missing captured boundary or sequence identifiers")
    return {
        "eager": eager,
        "random": random,
        "sequence": sequence,
        "replay": replay + replay[:1],
    }


def profile_coverage(meta, data):
    """Audit every mandatory MoE input without loading Torch or filtering IDs."""
    if data["task_kind"] == "dsa":
        return None
    requirement = meta.get("sequence_requirement", {})
    ledger = meta.get("call_sequence", [])
    ledger_hash = hashlib.sha256(
        json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        requirement.get("policy") != "all_declared_calls_required"
        or requirement.get("required_calls") != len(ledger)
        or requirement.get("call_sequence_sha256") != ledger_hash
        or not ledger
    ):
        raise RuntimeError("mandatory call-sequence ledger count or hash changed")
    catalog = profiles(meta)
    available = {row["sig"] for row in data["records"]}
    missing = {}
    for profile, identifiers in catalog.items():
        by_sig = {}
        for position, sig in enumerate(identifiers):
            if sig not in available:
                by_sig.setdefault(sig, []).append(position)
        if by_sig:
            missing[profile] = [
                {"sig": sig, "positions_0based": positions, "count": len(positions)}
                for sig, positions in by_sig.items()
            ]
    required = len(catalog["sequence"])
    absent = sum(row["count"] for row in missing.get("sequence", []))
    return {
        "status": "incomplete" if missing else "complete",
        "scope": "input_availability_only_not_gpu_validation",
        "required_sequence_calls": required,
        "available_sequence_calls": required - absent,
        "missing_sequence_calls": absent,
        "call_sequence_sha256": ledger_hash,
        "missing_inputs": missing,
    }


def tensor_leaves(value, torch):
    if type(value) is torch.Tensor:
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensor_leaves(v, torch)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in tensor_leaves(v, torch)]
    return []


def snapshot_inputs(value, torch):
    tensors = []
    storages = {}
    for tensor in tensor_leaves(value, torch):
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        tensors.append(
            (
                tensor,
                (
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.dtype,
                    tensor.storage_offset(),
                    storage.data_ptr(),
                    dict(tensor.__dict__),
                ),
            )
        )
        if key not in storages:
            raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                storage, 0, (storage.nbytes(),), (1,)
            )
            storages[key] = (raw, raw.clone())
    return tensors, list(storages.values())


def require_unchanged(saved, torch):
    tensors, storages = saved
    for tensor, signature in tensors:
        current = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.storage_offset(),
            tensor.untyped_storage().data_ptr(),
            dict(tensor.__dict__),
        )
        if current != signature:
            raise RuntimeError("candidate changed input layout or attributes")
    for raw, copy in storages:
        if not torch.equal(raw, copy):
            raise RuntimeError("candidate mutated a read-only input")


def encode_tensor(tensor, torch):
    if type(tensor) is not torch.Tensor:
        raise TypeError("output must be an ordinary tensor")
    result = {
        "tensor": True,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "storage_offset": tensor.storage_offset(),
        "device": str(tensor.device),
    }
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    compressor = zlib.compressobj(1)
    parts = []
    for offset in range(0, raw.numel(), 1 << 20):
        parts.append(
            compressor.compress(bytes(raw[offset : offset + (1 << 20)].tolist()))
        )
    parts.append(compressor.flush())
    result["payload"] = {
        "codec": "zlib-base64",
        "bytes": raw.numel(),
        "data": base64.b64encode(b"".join(parts)).decode("ascii"),
    }
    return result


def encode_output(output, torch):
    # Every component supplied by the protected comparison adapter is numerical.
    # Defined packed scales and LSE values must not become structure-only results.
    parts = output if isinstance(output, (tuple, list)) else [output]
    encoded = [
        (encode_tensor(value, torch) if type(value) is torch.Tensor else value)
        for index, value in enumerate(parts)
    ]
    return (
        {"sequence": type(output).__name__, "items": encoded}
        if isinstance(output, (tuple, list))
        else encoded[0]
    )


def compare_output(observed, expected, tol, torch):
    if isinstance(expected, dict) and expected.get("tensor"):
        if not isinstance(observed, dict) or any(
            observed.get(k) != expected[k]
            for k in (
                "tensor",
                "shape",
                "stride",
                "dtype",
                "storage_offset",
                "device",
            )
        ):
            return False
        dt = dtype(torch, expected["dtype"])
        a = torch.frombuffer(
            bytearray(decode_bytes(observed["payload"])), dtype=dt
        ).reshape(expected["shape"])
        b = torch.frombuffer(
            bytearray(decode_bytes(expected["payload"])), dtype=dt
        ).reshape(expected["shape"])
        if not a.is_floating_point():
            return bool(torch.equal(a, b))
        a, b = a.float(), b.float()
        floor = float(tol) * b.pow(2).mean().sqrt().clamp_min(1e-6)
        return bool(((a - b).abs() <= floor + float(tol) * b.abs()).all())
    if isinstance(expected, dict) and "sequence" in expected:
        return (
            isinstance(observed, dict)
            and observed.get("sequence") == expected["sequence"]
            and len(observed.get("items", [])) == len(expected["items"])
            and all(
                compare_output(a, b, tol, torch)
                for a, b in zip(observed["items"], expected["items"])
            )
        )
    return type(observed) is type(expected) and observed == expected


def undefined_lse_mask(record, torch, device, m=None):
    node = record["undefined_lse"]
    shape = list(node["shape"])
    raw = decode_bytes(node["mask"])
    mask = torch.frombuffer(bytearray(raw), dtype=torch.bool).reshape(shape)
    if m is not None:
        mask = mask[: int(m)]
    return mask.to(device)


def scale_byte_offsets(rows, columns, padded_columns):
    """AITER d9e5ef7 mixed_moe_gemm_2stage.py:2623-2644 tiled E8M0 layout."""
    row = rows[:, None]
    column = columns[None, :]
    return (
        (row >> 5) * (32 * padded_columns)
        + (column >> 3) * 256
        + (column & 3) * 64
        + (row & 15) * 4
        + ((column >> 2) & 1) * 2
        + ((row >> 4) & 1)
    )


def moe1_comparison(output, args, torch, active_tokens=None):
    """Compare the FP8 mantissas and the consumed, jointly dequantized result.

    Padding in the allocated scale buffer is not a numerical output. Select
    exactly the producer's valid sorted rows and the real inter-dimension groups.
    """
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise RuntimeError("stage 1 must return its FP8 values and E8M0 scales")
    values, scales = output
    kw = args["kwargs"]
    activation = args["args"][0] if args["args"] else kw["a"]
    tokens = int(activation.shape[0])
    topk = int(kw["topk"])
    inter_dim = int(kw["w1"].shape[1]) // 2
    if (
        type(values) is not torch.Tensor
        or type(scales) is not torch.Tensor
        or values.dtype != torch.float8_e4m3fn
        or scales.dtype != torch.float8_e8m0fnu
        or tuple(values.shape) != (tokens, topk, inter_dim)
        or tuple(values.stride()) != (topk * inter_dim, inter_dim, 1)
        or values.storage_offset() != 0
        or values.device != activation.device
        or scales.device != activation.device
    ):
        raise RuntimeError("stage-1 output dtype or shape changed")
    sorted_ids = kw["sorted_token_ids"]
    expert_ids = kw["sorted_expert_ids"]
    tile_m = int(kw["tile_m"])
    rows_count = max(sorted_ids.numel(), expert_ids.numel() * tile_m)
    padded_rows = (rows_count + 255) // 256 * 256
    groups = inter_dim // 32
    padded_columns = (groups + 7) // 8 * 8
    if (
        tuple(scales.shape) != (padded_rows, padded_columns)
        or tuple(scales.stride()) != (padded_columns, 1)
        or scales.storage_offset() != 0
    ):
        raise RuntimeError("stage-1 scale allocation contract changed")
    limit = tokens if active_tokens is None else int(active_tokens)
    rows = torch.arange(sorted_ids.numel(), device=sorted_ids.device, dtype=torch.int64)
    packed_ids = sorted_ids.to(torch.int64)
    token_ids, slots = packed_ids & 0xFFFFFF, packed_ids >> 24
    block_ids = rows // tile_m
    safe_blocks = block_ids.clamp_max(expert_ids.numel() - 1)
    experts = expert_ids[safe_blocks]
    grid_y = min(
        min(tokens * topk * tile_m, sorted_ids.numel()) // tile_m, expert_ids.numel()
    )
    valid = (
        (rows < kw["num_valid_ids"].reshape(-1)[0])
        & (rows < grid_y * tile_m)
        & (token_ids < limit)
        & (slots >= 0)
        & (slots < topk)
        & (block_ids < expert_ids.numel())
        & (experts >= 0)
        & (experts < kw["w1"].shape[0])
    )
    selected = rows[valid]
    offsets = scale_byte_offsets(
        selected, torch.arange(groups, device=values.device), padded_columns
    )
    codes = scales.contiguous().view(torch.uint8).reshape(-1)[offsets]
    if bool((codes == 255).any()):
        raise RuntimeError("stage-1 returned undefined E8M0 scales for consumed routes")
    factors = (codes.float() - 127).exp2()
    routed = values[token_ids[valid], slots[valid]].float().reshape(-1, groups, 32)
    dequantized = (routed * factors[:, :, None]).reshape(-1, inter_dim)
    return values[:limit], dequantized


def dsa_comparison(output, args, record, torch):
    """Numerically check LSE; retain only capture-proven NaN don't-care positions.

    The combine kernel writes +inf for empty contexts. Preserve its sign/position
    as exact boolean outputs and compare finite values with the declared tolerance.
    """
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise RuntimeError("DSA must return attention values and LSE")
    values, lse = output
    q = args["q"]
    if (
        type(values) is not torch.Tensor
        or type(lse) is not torch.Tensor
        or values.dtype != torch.bfloat16
        or lse.dtype != torch.float32
        or tuple(values.shape) != tuple(q.shape[:-1]) + (int(args["head_dim_v"]),)
        or tuple(lse.shape) != tuple(q.shape[:-1])
        or tuple(values.stride())
        != (
            q.shape[1] * q.shape[2] * int(args["head_dim_v"]),
            q.shape[2] * int(args["head_dim_v"]),
            int(args["head_dim_v"]),
            1,
        )
        or tuple(lse.stride()) != (q.shape[1] * q.shape[2], q.shape[2], 1)
        or values.storage_offset() != 0
        or lse.storage_offset() != 0
        or values.device != q.device
        or lse.device != q.device
    ):
        raise RuntimeError("DSA output dtype or shape changed")
    mask = undefined_mask(record, torch, values.device, m=q.shape[0])
    if mask is not None:
        values = values.masked_fill(mask, 0)
    lse = lse.masked_fill(
        undefined_lse_mask(record, torch, lse.device, m=q.shape[0]), 0
    )
    positive, negative = torch.isposinf(lse), torch.isneginf(lse)
    finite = lse.masked_fill(positive | negative, 0)
    return values, finite, positive, negative
