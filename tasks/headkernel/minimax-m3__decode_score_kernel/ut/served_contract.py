"""Protected MiniMax served-case validation and lossless addressing codec."""
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib

from evaluation_contract import canonical, fingerprint, require, strict_json, validate_manifest

FLOAT_INPUTS = {"q", "k_cache", "v_cache", "sink"}
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float8_e4m3fn": 1,
               "float8_e4m3fnuz": 1, "int32": 4, "int64": 8, "uint8": 1, "bool": 1}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def encode_bytes(value):
    data = bytes(value)
    return {"encoding": "zlib-base64", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "data": base64.b64encode(zlib.compress(data, 9)).decode("ascii")}


def _inflate(compressed, size):
    inflater = zlib.decompressobj()
    data = inflater.decompress(compressed, size + 1)
    require(len(data) == size and inflater.eof and not inflater.unused_data and not inflater.unconsumed_tail,
            "captured geometry is truncated, oversized or contains extra streams")
    return data


def encode_geometry(value, root, *, inline_limit=65536):
    """Losslessly share repeated integer-page chunks across full-window states.

    The compressed binary chunk index keeps cases.json compact even when a
    physical request table has many unused rows. All chunks are protected task
    files and retain exact content hashes; no observed tensor value is dropped.
    """
    data = bytes(value)
    if len(data) <= inline_limit:
        return encode_bytes(data)
    directory = Path(root) / "cases/geometry"
    require(directory.resolve().is_relative_to(Path(root).resolve()), "geometry directory escapes task root")
    directory.mkdir(parents=True, exist_ok=True)
    hashes = bytearray()
    checked = set()
    chunk_bytes = 4096
    for offset in range(0, len(data), chunk_bytes):
        chunk = data[offset:offset+chunk_bytes]
        checksum = hashlib.sha256(chunk).hexdigest()
        path = directory / (checksum + ".zlib")
        require(not path.is_symlink(), "geometry chunk must not be a symlink")
        if checksum in checked:
            hashes.extend(bytes.fromhex(checksum))
            continue
        if path.exists():
            require(path.is_file() and not path.is_symlink()
                    and _inflate(path.read_bytes(), len(chunk)) == chunk, "existing geometry chunk differs")
        else:
            path.write_bytes(zlib.compress(chunk, 9))
        checked.add(checksum)
        hashes.extend(bytes.fromhex(checksum))
    return {"encoding": "zlib-chunk-sha256-v1", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "chunk_bytes": chunk_bytes,
            "chunk_index": encode_bytes(hashes)}


def decode_bytes(value, *, limit=2 << 30, root=None):
    size = value.get("bytes")
    require(type(size) is int and 0 <= size <= limit, "unbounded captured geometry")
    if value.get("encoding") == "zlib-base64":
        compressed = base64.b64decode(value["data"], validate=True)
        data = _inflate(compressed, size)
    else:
        require(value.get("encoding") == "zlib-chunk-sha256-v1" and root is not None,
                "unknown captured geometry encoding or missing task root")
        chunk_bytes = value.get("chunk_bytes")
        require(type(chunk_bytes) is int and chunk_bytes == 4096, "geometry chunk width differs")
        count = (size + chunk_bytes-1)//chunk_bytes
        index = value["chunk_index"]
        require(index.get("encoding") == "zlib-base64", "nested geometry chunk index is forbidden")
        hashes = decode_bytes(index, limit=count*32)
        require(len(hashes) == count*32, "geometry chunk index length differs")
        directory = Path(root).resolve() / "cases/geometry"
        cache, parts = {}, []
        for number in range(count):
            checksum = hashes[number*32:(number+1)*32].hex()
            length = min(chunk_bytes, size-number*chunk_bytes)
            if checksum not in cache:
                path = directory / (checksum + ".zlib")
                require(path.is_file() and not path.is_symlink()
                        and path.resolve().is_relative_to(directory.resolve())
                        and directory.resolve().is_relative_to(Path(root).resolve())
                        and path.stat().st_size <= chunk_bytes+1024, "unsafe or oversized geometry chunk")
                chunk = _inflate(path.read_bytes(), length)
                require(hashlib.sha256(chunk).hexdigest() == checksum, "geometry chunk hash mismatch")
                cache[checksum] = chunk
            require(len(cache[checksum]) == length, "geometry chunk byte count differs")
            parts.append(cache[checksum])
        data = b"".join(parts)
    require(hashlib.sha256(data).hexdigest() == value.get("sha256"), "captured geometry hash mismatch")
    return data


def span_bytes(spec):
    shape, strides = spec["shape"], spec["strides"]
    require(spec["dtype"] in DTYPE_BYTES, "unsupported captured dtype")
    if any(n == 0 for n in shape):
        return spec["storage_offset"] * DTYPE_BYTES[spec["dtype"]]
    return (spec["storage_offset"] + 1 + sum((n-1)*s for n, s in zip(shape, strides))) * DTYPE_BYTES[spec["dtype"]]


def case_identity(case):
    # Values and counts are separately protected; identity is the complete ABI,
    # not a truncated textual rendering or an assumed logical batch size.
    keys = ("kind", "production_mode", "tensors", "scalars", "input_aliases", "output_structure",
            "launch_contract", "served_contract", "original_storage_nbytes", "capture_schema_sha256")
    return "minimax-" + fingerprint({key: case[key] for key in keys if key in case})


def input_names(case):
    return {name for name, spec in case["tensors"].items() if spec["role"] == "input"}


def validate_distribution(case):
    from minimax_work import PROJECTION_ID, structural_work, work_metric
    require(case.get("work_projection_id") == PROJECTION_ID, "wrong protected work projection")
    distribution = case.get("work_distribution", {})
    require(distribution and sum(x["occurrences"] for x in distribution.values()) == case["occurrences"],
            "exact work distribution does not cover all observed calls")
    for identity, row in distribution.items():
        controls = row["tensor_controls"]
        expected_controls = {"inputs."+name for name in input_names(case)
                             if name in {"seq_lens", "cu_seqlens", "prefix_lens", "cu_seqblocks_q"}}
        if "topk_idx" in case["tensors"]:
            expected_controls.add("inputs.capture_topk_counts")
        require(set(controls) == expected_controls, "exact work-control set differs")
        require(type(row["occurrences"]) is int and row["occurrences"] > 0 and fingerprint(controls) == identity,
                "work-state identity/count differs")
        for name, value in controls.items():
            require(isinstance(value, dict) and isinstance(value.get("shape"), list)
                    and isinstance(value.get("runs"), list), "invalid exact work controls")
            require(all(type(n) is int and n >= 0 for n in value["shape"])
                    and all(type(v) is int and type(n) is int and n > 0 for v, n in value["runs"])
                    and sum(n for _, n in value["runs"]) == math.prod(value["shape"]) <= 65536, "invalid exact control RLE")
            tensor_name = name.removeprefix("inputs.")
            expected_shape = (case["tensors"]["topk_idx"]["shape"][:2] if tensor_name == "capture_topk_counts"
                              else case["tensors"][tensor_name]["shape"])
            require(value["shape"] == expected_shape, "work-control tensor shape differs")
            if name == "inputs.seq_lens":
                require(all(0 <= v <= case["tensors"]["req_to_token"]["shape"][1] for v, _ in value["runs"]),
                        "observed sequence length exceeds actual paging capacity")
        require(structural_work(controls, case["scalars"]) == case["scalars"].get("work.variant"),
                "observed work value escapes its structural class")
    metrics = [work_metric(row["tensor_controls"]) for row in distribution.values()]
    require(case.get("observed_extrema") == {"min": list(min(metrics)), "max": list(max(metrics))},
            "reported extrema differ from full observed work distribution")
    represented = {tuple(work_metric(state["tensor_controls"])) for state in case["states"]}
    require(min(metrics) in represented and max(metrics) in represented,
            "observed global work extrema lack actual fixture states")
    labels = set()
    for state in case["states"]:
        identity = fingerprint(state["tensor_controls"])
        require(identity in distribution, "representative state was not observed in the workload")
        labels.update(state["representative_labels"])
    require({"first", "min", "max"} <= labels, "first/min/max representative coverage is incomplete")


def validate_cases(manifest, definition, root=None):
    validate_manifest(manifest)
    require(manifest["runtime_image"] == definition["runtime_image"], "wrong runtime image")
    evidence = manifest.get("capture", {})
    require(evidence.get("served_only") is True and evidence.get("startup_values") is False,
            "actual served capture required; startup graph buffers are not fixtures")
    require(evidence.get("complete") is True and evidence.get("dropped_case_count") == 0,
            "capture is incomplete or omitted required contracts")
    require(evidence.get("source_sha256") == definition["source_sha256"]
            and evidence.get("run_id") and evidence.get("manifest_sha256"), "capture/source identity is missing")
    require(manifest["measurement"]["warmup_iterations"] == 10
            and manifest["measurement"]["benchmark_iterations"] == 100
            and manifest["measurement"]["correctness_seeds"] == [0, 1, 2],
            "frozen 10/100 timing and three-draw correctness policy changed")
    require(manifest["measurement"].get("refresh_addressing") == "translated_pages_and_legal_block_permutations",
            "fresh replay must vary concrete pages/routing while preserving captured work controls")
    ids = [case["case_id"] for case in manifest["cases"]]
    require(evidence.get("required_case_ids") == ids, "case set differs from complete observed capture")
    require(evidence.get("represented_calls") == sum(c["occurrences"] for c in manifest["cases"])
            == evidence.get("target_calls"), "case multiplicity does not cover target calls")
    meaningful_score = definition["kind"] != "decode_score"
    external = definition.get("fixture_format") == "minimax-representatives-v1"
    if external:
        require(evidence.get("coverage_method") == "captured_plus_native_replay_v1"
                and evidence.get("full_work_distributions") is True, "complete supplemented coverage is required")
        from minimax_coverage import validate_coverage
        validate_coverage(manifest, definition, root)
    for case in manifest["cases"]:
        require(case["kind"] == definition["kind"] and case["case_id"] == case_identity(case),
                "case kind or full ABI identity differs")
        require(case["calls_per_sample"] == 1, "one complete wrapper invocation is required per replay")
        names = input_names(case)
        require(set(case["input_aliases"]) == names, "all original input aliases must be declared")
        scalar_inputs = {k for k in case["scalars"] if not k.startswith(("result", "work."))}
        require(names | scalar_inputs == set(definition["arguments"])
                and not names & scalar_inputs, "incomplete or ambiguous wrapper argument ABI")
        require({"q", "k_cache", "req_to_token", "seq_lens", "slot_ids"} <= names,
                "query, cache and paging tensors are mandatory")
        for spec in case["tensors"].values():
            require(spec["dtype"] in DTYPE_BYTES, "unsupported tensor dtype")
            require(spec["role"] != "inout", "this source contract has no mutable input arguments")
            if spec["shape"]:
                require(spec["strides"][-1] == 1, "an observed non-unit inner stride needs an explicit adapter")
            span_bytes(spec)
        require(case.get("states") and isinstance(case["states"], list), "captured addressing states required")
        require(case.get("production_mode") in ("eager", "graph"), "unknown production mode")
        if external:
            require(root is not None and case.get("fixture") and case.get("launch_contract")
                    and set(case.get("original_storage_nbytes", {})) == names,
                    "external physical fixture/launch contract is missing")
            validate_distribution(case)
            from minimax_fixtures import load_bundle
            load_bundle(root, case, definition)
        seen_states = set()
        for state in case["states"]:
            require(state.get("served") is True and state.get("startup_values") is False,
                    "unserved or startup state cannot enter a task")
            require(state.get("state_id") not in seen_states and state.get("fixture_sha256"),
                    "duplicate state or missing original fixture hash")
            seen_states.add(state["state_id"])
            if case["production_mode"] == "graph":
                rank = str(state.get("served_context", {}).get("tp_rank"))
                replays = evidence.get("graph_replays_by_rank", {}).get(rank, {})
                require(type(replays.get(state.get("graph_id"))) is int
                        and replays[state["graph_id"]] >= 2,
                        "two actual served replays are required for every used graph")
            geometry = state.get("geometry", {})
            if not external:
                require(set(geometry) == names - FLOAT_INPUTS, "every integer/scale tensor needs captured bytes")
                for name, item in geometry.items():
                    spec = case["tensors"][name]
                    data = decode_bytes(item, root=root)
                    require(len(data) == math.prod(spec["shape"]) * DTYPE_BYTES[spec["dtype"]],
                            "captured geometry tensor byte count differs")
            if definition["kind"] == "decode_score":
                require(case["scalars"].get("disable_index_value") is True
                        and case["scalars"].get("use_dense_main_attn") is False
                        and case["scalars"].get("page_size") == 1
                        and case["scalars"].get("sink") is None
                        and case["scalars"].get("v_cache") is None,
                        "observed score branch needs its own independent oracle")
                if external:
                    lens = [value for value, _ in state["tensor_controls"]["inputs.seq_lens"]["runs"]]
                else:
                    import struct
                    spec = case["tensors"]["seq_lens"]
                    fmt = "q" if spec["dtype"] == "int64" else "i"
                    lens = [x[0] for x in struct.iter_unpack("<"+fmt, decode_bytes(geometry["seq_lens"], root=root))]
                meaningful_score |= any((n + case["scalars"]["block_size"]-1)//case["scalars"]["block_size"]
                                        > case["scalars"]["topk"] for n in lens)
    require(meaningful_score, "score task needs observed nontrivial top-k work, not warmup-only geometry")
    return manifest


def load_cases(root):
    root = Path(root)
    definition = strict_json((root / "task_definition.json").read_text())
    path = root / "cases.json"
    require(path.is_file(), "fresh served cases.json is missing; historical fixtures are not substituted")
    return validate_cases(strict_json(path.read_text()), definition, root), definition
