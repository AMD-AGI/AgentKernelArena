"""Pure-data native routing policy and complete graph report contract.

The editable file is parsed, never imported or executed. Keep this module
stdlib-only so both compile and authoritative quality checks can use it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

PROTOCOL = "qwen_native_route_graph_v1"
WARMUP = 10
SAMPLES = 100
CHECKED = WARMUP + SAMPLES + 4
PREPARED = WARMUP + SAMPLES + 6
POLICY_FILE = "source/routing.py"
CHOICES = {"dense": {"recorded", "torch", "triton"},
           "moe": {"captured", "block32", "block64", "block128"}}


def require(value, message):
    if not value:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    def invalid(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def specs(meta):
    kind = meta["native_routing"]["kind"]
    require(kind in CHOICES and meta["native_routing"]["protocol"] == PROTOCOL, "unknown native routing contract")
    rows = meta["workload"]["cases"]
    require(len(rows) == (5 if kind == "dense" else 2), "incomplete native routing case set")
    require(len({row["sig"] for row in rows}) == len(rows), "duplicate native routing case")
    if kind == "dense":
        require([(row["m"], row["n"], row["k"]) for row in rows] ==
                [(16384, 4608, 8192), (64, 32, 8192), (64, 512, 8192), (64, 4608, 8192), (64, 8192, 256)],
                "original five dense geometries changed")
    else:
        require([(row["regime"], row["m"]) for row in rows] == [("decode", 64), ("prefill", 8192)],
                "original two captured MoE geometries changed")
        require(meta["geometry"]["hidden_size"] == 8192 and meta["geometry"]["topk"] == 10
                and meta["geometry"]["local_experts"] == 64 and meta["geometry"]["expert_mask_size"] == 512,
                "captured MoE EP/shape contract changed")
    return rows


def parse_policy(raw, meta):
    require(isinstance(raw, bytes) and len(raw) <= 65536, "routing source exceeds the bounded literal grammar")
    tree = ast.parse(raw, filename=POLICY_FILE)
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body.pop(0)
    require(len(body) == 1 and isinstance(body[0], ast.FunctionDef), "routing source permits only select_routes")
    function = body[0]
    args = function.args
    require(function.name == "select_routes" and not function.decorator_list and function.returns is None
            and not args.posonlyargs and not args.args and not args.kwonlyargs and not args.defaults
            and not args.kw_defaults and args.vararg is None and args.kwarg is None,
            "routing selector signature/decorators changed")
    require(len(function.body) == 1 and isinstance(function.body[0], ast.Return)
            and isinstance(function.body[0].value, ast.Dict), "routing body must return one literal mapping")
    mapping = function.body[0].value
    result = {}
    for key, value in zip(mapping.keys, mapping.values):
        require(isinstance(key, ast.Constant) and type(key.value) is str
                and isinstance(value, ast.Constant) and type(value.value) is str,
                "routing choices must be string literals; Python expressions are never executed")
        require(key.value not in result, "duplicate routing case")
        result[key.value] = value.value
    rows = specs(meta)
    require(set(result) == {row["sig"] for row in rows}, "routing must name every original case exactly once")
    require(set(result.values()) <= CHOICES[meta["native_routing"]["kind"]], "unsupported native route")
    return result


def dense_route_config(spec, route):
    """Resolve a literal to the exact configuration passed to the native GEMM.

    Request identity, invocation and report validation share this resolver so
    aliases of a recorded native backend cannot create an apparent source gain.
    """
    require(route in CHOICES["dense"], "unapproved dense route")
    if route == "recorded":
        return {"libtype": spec["expected_backend"], "solidx": int(spec["expected_solidx"]),
                "splitK": int(spec["expected_split_k"]), "kernelName": spec["expected_kernel"]}
    return {"libtype": route, "solidx": 0, "splitK": 0, "kernelName": "native"}


def source_context(task, meta=None):
    task = Path(task).resolve()
    meta_path = task / "ut/meta.json"
    meta = strict_json(meta_path.read_text()) if meta is None else meta
    path = task / POLICY_FILE
    require(path.is_file() and not path.is_symlink() and not path.parent.is_symlink()
            and path.resolve().is_relative_to(task),
            "routing source must be a task-local regular file")
    raw = path.read_bytes()
    routes = parse_policy(raw, meta)
    native = task / "ut" / meta["source_provenance"]["baseline_ref"]
    native_sha = meta["source_provenance"]["source_sha256"]
    require(native.is_file() and not native.is_symlink() and file_sha(native) == native_sha,
            "frozen SG520 native source changed")
    context = {"protocol": PROTOCOL, "kind": meta["native_routing"]["kind"], "routes": routes,
               "source_file_sha256": hashlib.sha256(raw).hexdigest(), "meta_sha256": file_sha(meta_path),
               "native_source_sha256": native_sha, "fixture_sha256": meta.get("reference_io_sha256") or None}
    # Raw labels/bytes bind the request. Dense execution identity uses resolved
    # configs: hk05 recorded and torch, for example, invoke the same native op.
    identity = {key: context[key] for key in ("protocol", "kind", "native_source_sha256")}
    if context["kind"] == "dense":
        context["resolved_native_configs"] = {row["sig"]: dense_route_config(row, routes[row["sig"]])
                                              for row in specs(meta)}
        identity["resolved_native_configs"] = context["resolved_native_configs"]
    else:
        identity["routes"] = routes
    context["effective_source_sha256"] = fingerprint(identity)
    return context


def case_params(meta, spec):
    common = {"regime": spec["regime"], "m": int(spec["m"]), "dtype": "torch.bfloat16",
              "output_shape": [int(spec["m"]), int(spec.get("n", meta.get("geometry", {}).get("hidden_size", 0)))]}
    if meta["native_routing"]["kind"] == "dense":
        return {**common, "n": int(spec["n"]), "k": int(spec["k"]),
                "a_shape": [spec["m"], spec["k"]], "a_stride": spec["a_stride"],
                "b_shape": [spec["n"], spec["k"]], "b_stride": spec["b_stride"],
                "bias": None, "scale_a": None, "scale_b": None, "scale_c": None, "bpreshuffle": False}
    geo = meta["geometry"]
    return {**common, "hidden_shape": spec["hidden_shape"], "hidden_stride": spec["hidden_stride"],
            "topk": geo["topk"], "local_experts": geo["local_experts"],
            "expert_mask_size": geo["expert_mask_size"],
            **{key: geo[key] for key in ("w1_shape", "w2_shape", "w1_scale_shape", "w2_scale_shape")},
            "weight_dtype": "torch.float4_e2m1fn_x2", "scale_dtype": "torch.uint8",
            "fixture_sha256": meta["reference_io_sha256"]}


def make_request(task, mode, run_id):
    meta = strict_json((Path(task) / "ut/meta.json").read_text())
    require(mode in ("compile", "correctness", "performance"), "unknown evaluation phase")
    require(re.fullmatch(r"[0-9a-f]{48}", run_id) is not None, "fresh request ID required")
    return {"schema": "qwen-native-routing-request-v1", "run_id": run_id, "mode": mode,
            "source": source_context(task, meta), "cases": [row["sig"] for row in specs(meta)],
            "runtime_image": meta["source_provenance"]["runtime_image"]}


def validate_request(request, task, mode):
    expected = make_request(task, mode, request.get("run_id", ""))
    require(request == expected, "request/source/metadata/case binding changed")
    return expected


def validate_payload(payload, meta, request):
    require(payload.get("status") == "ok" and payload.get("protocol") == PROTOCOL,
            "native routing evaluation did not complete")
    require(payload.get("request") == request and payload.get("request_sha256") == fingerprint(request),
            "stale or mismatched evaluation request")
    rows = payload.get("cases")
    expected = specs(meta)
    require(isinstance(rows, list) and len(rows) == len(expected), "missing or extra native routing cases")
    require([row.get("sig") for row in rows] == [spec["sig"] + "|" + spec["regime"] for spec in expected],
            "native routing cases were dropped, duplicated, reordered or substituted")
    for row, spec in zip(rows, expected):
        require(row.get("params") == case_params(meta, spec), "case ABI differs from the protected workload")
        require(row.get("selected_route") == request["source"]["routes"][spec["sig"]], "selected route differs from source")
        validation = row.get("validation", {})
        require(validation.get("expected_device") == "cpu"
                and validation.get("inputs_unchanged") is True
                and validation.get("reference_after_candidate_snapshot") is True,
                "unprotected expected output or unchecked input mutation")
        require(validation.get("tolerance") == float(spec.get("tol", meta["tol"])), "tolerance changed")
        require(type(validation.get("max_rel_err")) in (int, float)
                and math.isfinite(validation["max_rel_err"]) and validation["max_rel_err"] >= 0,
                "missing/nonfinite numerical result")
        if meta["native_routing"]["kind"] == "moe":
            require(validation.get("captured_reference_checked") is True
                    and validation.get("routes_weights_scales_preserved") is True, "captured MoE binding not verified")
        if request["mode"] == "performance":
            require(payload.get("timer") == "cuda_graph" and payload.get("warmup") == WARMUP
                    and payload.get("iters") == SAMPLES, "wrong graph timing policy")
            require(row.get("benchmark_method") == "cuda_graph" and row.get("benchmark_warmup") == WARMUP
                    and row.get("benchmark_samples") == SAMPLES and row.get("benchmark_effective_repeats") == 1
                    and "benchmark_fallback_reason" not in row, "event fallback or changed graph work is not admitted")
            samples = row.get("samples_ms")
            require(isinstance(samples, list) and len(samples) == SAMPLES
                    and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in samples),
                    "every ordered graph sample is required")
            require(math.isclose(row["mean_ms"], math.fsum(samples) / SAMPLES, rel_tol=1e-13)
                    and row["median_ms"] == statistics.median(samples) and row["min_ms"] == min(samples)
                    and row["max_ms"] == max(samples), "timing summaries differ from raw samples")
            require(validation.get("checked_invocations") == CHECKED
                    and validation.get("fresh_input_sets") == PREPARED
                    and validation.get("outputs_poisoned_before_replay") is True
                    and validation.get("measured_graph_validated") is True,
                    "not every measured-graph invocation was refreshed/validated")
        else:
            require(validation.get("checked_invocations", 0) >= 4
                    and validation.get("negative_controls") == {"no_op": "rejected", "wrong_output": "rejected"},
                    "incomplete captured/fresh correctness or negative controls")
    return rows
