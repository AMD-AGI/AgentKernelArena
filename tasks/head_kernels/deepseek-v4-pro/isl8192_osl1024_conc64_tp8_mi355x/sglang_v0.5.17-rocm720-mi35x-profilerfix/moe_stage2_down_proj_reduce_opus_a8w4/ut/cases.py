"""Protected generated numerical inputs with the complete captured routing/case set."""

from __future__ import annotations
import importlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "meta.json").read_text())
_spec = importlib.util.spec_from_file_location(
    "generated_contract", HERE / "generated_contract.py"
)
GEN = sys.modules.get("generated_contract")
if (
    GEN is not None
    and Path(getattr(GEN, "__file__", "")).resolve()
    != (HERE / "generated_contract.py").resolve()
):
    raise RuntimeError("trusted generated_contract alias names a different file")
if GEN is None:
    GEN = importlib.util.module_from_spec(_spec)
    sys.modules["generated_contract"] = GEN
    _spec.loader.exec_module(GEN)
CONTRACT = GEN.load_contract(HERE)
TASK_KIND = CONTRACT["task_kind"]
_RECORDS = {record["sig"]: record for record in CONTRACT["records"]}
_ROW_ARGS = (
    "q",
    "indices",
    "topk_length",
    "extra_indices_in_kvcache",
    "extra_topk_length",
)
_KV_PAIRS = (("k_cache", "indices"), ("extra_k_cache", "extra_indices_in_kvcache"))
_UNDEF_KEY = "__undef__"
_SOURCE_KEY = "__source_sig__"


def _torch():
    import torch

    return torch


_BOUND = None


def _resolve(dotted=None):
    global _BOUND
    import os

    target_module, attr = (dotted or META["target_callable"]).split(":")
    if _BOUND is None:
        settings = META["generated_inputs"]
        candidate = bool(os.environ.get("GEAK_ACTIVE_TASK_CANDIDATE"))
        path = (
            HERE.parent / settings["candidate_source"]
            if candidate
            else HERE / settings["baseline_source"]
        )
        if not candidate and GEN.digest(path) != settings["baseline_sha256"]:
            raise RuntimeError("frozen baseline source changed")
        package = settings["source_package"]
        importlib.import_module(package)
        name = package + (
            "._deepseek_generated_candidate"
            if candidate
            else "._deepseek_generated_reference"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            from importlib.machinery import SourceFileLoader

            spec = importlib.util.spec_from_loader(
                name, SourceFileLoader(name, str(path))
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _BOUND = getattr(module, attr)
        setattr(importlib.import_module(target_module), attr, _BOUND)
    return getattr(importlib.import_module(target_module), attr)


def _specs(meta):
    return list(meta.get("case_specs", []))


def _spec_by_name(meta, name):
    return next((row for row in _specs(meta) if row["name"] == name), None)


def _undef_mask(sig, torch, device, m=None):
    return GEN.undefined_mask(_RECORDS[sig], torch, device, m)


def _build_args(
    sig, torch, device, m=None, rng=None, index_mode="recorded", undef=True
):
    pos, kw = GEN.build_record(_RECORDS[sig], META, torch, device)
    if pos:
        raise RuntimeError("DSA capture must use its recorded keyword ABI")
    for name in _ROW_ARGS:
        value = kw.get(name)
        if torch.is_tensor(value) and m is not None and value.shape[0] > int(m):
            kw[name] = value[: int(m)].contiguous()
    if rng is not None:
        value = kw["q"]
        kw["q"] = torch.randn(
            value.shape, generator=rng, device=device, dtype=torch.float32
        ).to(value.dtype)
    if index_mode == "shortctx":
        for _, name in _KV_PAIRS:
            if torch.is_tensor(kw.get(name)):
                kw[name] = kw[name][:1].expand(kw[name].shape).contiguous()
    elif index_mode != "recorded":
        raise ValueError(index_mode)
    if undef:
        mask = _undef_mask(sig, torch, device, m)
        if mask is not None:
            kw[_UNDEF_KEY] = mask
    kw[_SOURCE_KEY] = sig
    return kw


def _primary(args):
    return (
        args["args"][0]
        if args["args"]
        else args["kwargs"][META.get("primary_input_key", "a")]
    )


def _oracle():
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for record in CONTRACT["records"]:
        pos, kw = GEN.build_record(record, META, torch, device)
        args = {"args": pos, "kwargs": kw}
        rows.append(
            {
                "sig": record["sig"],
                "regime": record["regime"],
                "m": int(_primary(args).shape[0]),
                "args": args,
                "output_contract": record["output_contract"],
            }
        )
    return rows


def oracle_records(h=None, meta=None):
    return _oracle()


def call_full(args):
    torch = _torch()
    fn = _resolve()
    if TASK_KIND == "dsa":
        return fn(
            **{
                key: value
                for key, value in args.items()
                if key not in {_UNDEF_KEY, _SOURCE_KEY}
            }
        )
    pos = list(args["args"])
    kw = dict(args["kwargs"])
    out_key = META.get("out_key", "out")
    if torch.is_tensor(kw.get(out_key)):
        kw[out_key] = torch.zeros_like(kw[out_key])
    index = META.get("inplace_out_arg")
    if index is not None and torch.is_tensor(pos[index]):
        pos[index] = torch.zeros_like(pos[index])
    return fn(*pos, **kw)


def call(args):
    # Return the actual timed callable result; comparison transforms run outside
    # the device timing interval in the canonical runner.
    return call_full(args)


def comparison_output(output, args, active_tokens=None):
    torch = _torch()
    if TASK_KIND == "dsa":
        return GEN.dsa_comparison(output, args, _RECORDS[args[_SOURCE_KEY]], torch)
    if TASK_KIND == "moe1":
        return GEN.moe1_comparison(output, args, torch, active_tokens)
    return output


class _LazyCase(dict):
    def __init__(self, fields, builder):
        super().__init__(fields)
        self.builder = builder

    def __getitem__(self, key):
        if key == "args" and key not in self:
            super().__setitem__(key, self.builder())
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def scored_case_specs(meta):
    """Only observed capture geometries are eligible for the timing score."""
    specs = {row["name"]: row for row in _specs(meta)}
    observed = {
        name
        for name, row in specs.items()
        if row["source_sig"] in _RECORDS
        and row["m"] == _RECORDS[row["source_sig"]]["kwargs"]["q"]["shape"][0]
        and row.get("scored") is True
    }
    names = [row["name"] for row in meta["workload"]["cases"]]
    if set(names) != observed or len(names) != len(observed):
        raise RuntimeError("DSA score case set differs from observed captured calls")
    return [specs[name] for name in names]


def timing_cases(h, meta):
    if TASK_KIND == "dsa":
        selected = scored_case_specs(meta)
        torch = _torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return [
            _LazyCase(
                {"sig": row["name"], "regime": row["regime"], "m": row["m"]},
                lambda spec=row: _build_args(
                    spec["source_sig"], torch, device, m=spec["m"], undef=False
                ),
            )
            for row in selected
        ]
    rows = {row["sig"]: row for row in _oracle()}
    names = [row["sig"] for row in meta["workload"]["cases"]]
    if set(names) != set(rows) or len(names) != len(rows):
        raise RuntimeError("MoE timing case set changed")
    return [rows[name] for name in names]


def random_shapes(h, meta):
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shapes = []
    if TASK_KIND == "dsa":
        for spec in _specs(meta):
            rm = int(spec.get("rand_m") or min(spec["m"], 1024))
            shapes.append(
                {
                    "sig": spec["name"],
                    "make_inputs": lambda rng, s=spec, m=rm: _build_args(
                        s["source_sig"], torch, device, m=m, rng=rng
                    ),
                }
            )
        for spec in _specs(meta):
            if spec.get("replay"):
                shapes.append(
                    {
                        "sig": "replay_shortctx:" + spec["name"],
                        "make_inputs": lambda rng, s=spec: _build_args(
                            s["source_sig"],
                            torch,
                            device,
                            m=s["m"],
                            index_mode="shortctx",
                        ),
                    }
                )
        return shapes
    for row in _oracle():

        def make(rng, row=row):
            base = row["args"]
            prior = _primary(base)
            values = torch.randn(
                prior.shape, generator=rng, device=device, dtype=torch.float32
            ) * float(meta.get("act_scale", 1.0))
            fresh = torch.empty_strided(
                prior.shape, prior.stride(), dtype=prior.dtype, device=device
            )
            fresh.copy_(values.to(prior.dtype))
            fresh.__dict__.update(prior.__dict__)
            pos = list(base["args"])
            kw = dict(base["kwargs"])
            if pos:
                pos[0] = fresh
            else:
                kw[meta.get("primary_input_key", "a")] = fresh
            return {"args": pos, "kwargs": kw}

        shapes.append({"sig": row["sig"], "make_inputs": make})
    return shapes
