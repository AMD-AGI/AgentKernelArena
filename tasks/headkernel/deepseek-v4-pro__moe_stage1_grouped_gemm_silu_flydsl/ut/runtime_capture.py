"""Finite-family production tensor capture; owner hooks supply roles and replay markers.

Graph construction inserts D2D copies, never emits fixtures. The owner calls
``after_served_replay`` before another replay overwrites those snapshot buffers.
No pickle, inferred tensor values, implicit full KV clone, or performance claim.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any


class CaptureError(RuntimeError):
    pass


def require(ok, why):
    if not ok:
        raise CaptureError(why)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def torch_module():
    import torch
    return torch


def capturing():
    torch = torch_module()
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def controls_json(value):
    """Owners flatten finite launch-config/enum codecs; arbitrary objects fail."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        require(math.isfinite(value), "Nonfinite control scalar requires an explicit owner codec")
        return value
    if isinstance(value, (list, tuple)):
        return [controls_json(x) for x in value]
    if isinstance(value, dict):
        require(value.get("kind") != "opaque", "Opaque control placeholder is not a captured value")
        require(all(type(k) is str for k in value), "Control keys must be strings")
        return {k: controls_json(v) for k, v in value.items()}
    raise CaptureError("Opaque control state; provide a finite codec: " + type(value).__qualname__)


@dataclass(frozen=True)
class Paged:
    indices: str
    unit_count: int
    unit_stride_bytes: int
    unit_bytes: int
    storage_offset_bytes: int = 0
    index_divisor: int = 1
    invalid_indices: tuple[int, ...] = (-1,)
    max_index_entries: int = 131072
    access_contract: str = "only_indexed_units"
    unit_name: str = "physical_token_row"


@dataclass(frozen=True)
class Role:
    kind: str                         # mutable / readonly / paged
    footprint: str = "view_span"      # view_span / full_storage (explicit)
    optional: bool = False
    paged: Paged | None = None
    case_control: bool = False


@dataclass(frozen=True)
class Family:
    name: str
    source_sha256: str
    input_roles: dict[str, Role]
    output_roles: dict[str, Role]
    tensor_control_codec: Any = None
    tensor_control_codec_id: str | None = None
    tensor_attribute_codec: Any = None
    tensor_attribute_codec_id: str | None = None


@dataclass(frozen=True)
class Budget:
    max_snapshot_bytes: int = 128 << 20
    max_live_snapshot_bytes: int = 512 << 20
    max_artifact_bytes: int = 8 << 30
    max_control_tensor_elements: int = 4096
    max_cases: int = 64
    max_runtime_case_keys: int = 4096
    max_graph_slots: int = 4096
    max_replay_notifications: int = 100000
    max_failure_records: int = 256
    chunk_bytes: int = 8 << 20
    checkpoint_seconds: float = 30.0


@dataclass(frozen=True)
class Served:
    run_id: str
    replay_id: str
    stage: str                        # prefill / decode; never startup/warmup
    tp_rank: int
    active_requests: int
    active_tokens: int
    workload_marker: str              # owner's actual profiled-workload marker


@dataclass
class Handle:
    family: Family
    mode: str
    controls: dict
    selected: bool
    served: Served | None = None
    graph_id: str | None = None
    slot_id: str | None = None
    bucket: str | None = None
    aliases: dict = field(default_factory=dict)
    storage_roles: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    plans: dict = field(default_factory=lambda: {"inputs": {}, "outputs": {}})
    live_bytes: int = 0
    finished: bool = False
    temporary_input_refs: dict = field(default_factory=dict)
    tensor_controls: dict = field(default_factory=dict)
    resolved_tensor_controls: dict = field(default_factory=dict)


def storage_key(t):
    storage = t.untyped_storage()
    return (str(t.device), storage._cdata, storage.nbytes())


def raw_storage(t):
    torch = torch_module()
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(
        t.untyped_storage(), 0, (t.untyped_storage().nbytes(),), (1,))


def tensor_metadata(t, alias):
    torch = torch_module()
    require(torch.is_tensor(t) and t.layout == torch.strided and not t.is_quantized,
            "Only ordinary strided tensors are representable")
    require(not t.is_conj() and not t.is_neg(), "Conjugate/negative-bit views need an explicit owner codec")
    require(all(x >= 0 for x in t.stride()), "Negative strides are unsupported")
    return {"kind": "tensor", "alias": alias, "shape": list(t.shape), "stride": list(t.stride()),
            "dtype": str(t.dtype), "storage_offset": t.storage_offset(),
            "storage_nbytes": t.untyped_storage().nbytes(), "element_size": t.element_size(),
            "device": str(t.device)}


def view_span(meta):
    start = meta["storage_offset"] * meta["element_size"]
    if any(n == 0 for n in meta["shape"]):
        return start, start
    end = start + (1 + sum((n - 1) * s for n, s in zip(meta["shape"], meta["stride"]))) * meta["element_size"]
    require(0 <= start <= end <= meta["storage_nbytes"], "Tensor view escapes storage")
    return start, end


def union_ranges(parts):
    """Union actual byte ranges; overlapping views must agree at the same phase."""
    merged = []
    for offset, data in sorted(parts, key=lambda x: x[0]):
        if not data:
            continue
        if not merged or offset > merged[-1][0] + len(merged[-1][1]):
            merged.append((offset, bytearray(data)))
            continue
        base, current = merged[-1]
        overlap = min(len(data), base + len(current) - offset)
        require(current[offset-base:offset-base+overlap] == data[:overlap],
                "Aliased snapshots disagree; call timing or footprint is invalid")
        current.extend(data[overlap:])
    return [(offset, bytes(data)) for offset, data in merged]


class Recorder:
    def __init__(self, root, provenance, budget=Budget(), *, graph_buckets=(), snapshot_slots=None, capture_ranks=(0,)):
        self.root = Path(root)
        self.provenance = controls_json(provenance)
        require(self.provenance.get("run_id"), "Current run_id is required")
        require(type(self.provenance.get("tp_rank")) is int and self.provenance["tp_rank"] >= 0,
                "Recorder provenance requires its actual TP rank")
        require(re.search(r"@sha256:[0-9a-f]{64}$", self.provenance.get("image", "")), "Exact image digest required")
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        (self.root / "blobs").mkdir()
        self.budget, self.graph_buckets = budget, frozenset(graph_buckets)
        self.snapshot_slots = None if snapshot_slots is None else frozenset(snapshot_slots)
        self.capture_ranks = frozenset(capture_ranks)
        self.metadata_only = self.provenance["tp_rank"] not in self.capture_ranks
        self.schemas = {}
        self.graphs, self.graph_replays = {}, {}
        self.case_counts, self.cases, self.failures = Counter(), {}, []
        self.snapshot_selections = Counter()
        self.failure_overflow = 0
        self.metadata_case_overflow = 0
        self.live_bytes = self.artifact_bytes = 0
        self.static_blobs = {}
        self._last_checkpoint = 0.0
        self._closed = False
        self.flush(reason="created")

    def _failure(self, error):
        if len(self.failures) < self.budget.max_failure_records:
            self.failures.append({"type": type(error).__name__, "reason": str(error), "time": time.time()})
        else:
            self.failure_overflow += 1
        if not capturing():
            self.flush(reason="capture_failure")

    def _family(self, family):
        require(isinstance(family, Family) and re.fullmatch(r"[a-zA-Z0-9_.-]+", family.name), "Finite family name required")
        require(re.fullmatch(r"[0-9a-f]{64}", family.source_sha256), "Family source hash required")
        for role in [*family.input_roles.values(), *family.output_roles.values()]:
            require(role.kind in ("mutable", "readonly", "paged"), "Unsupported role")
            require(role.footprint in ("view_span", "full_storage"), "Unsupported footprint")
            require((role.kind == "paged") == (role.paged is not None), "Paged role requires a finite footprint contract")
            if role.case_control:
                require(callable(family.tensor_control_codec) and family.tensor_control_codec_id,
                        "Declared tensor controls require a finite owner codec and codec ID")
        if family.tensor_attribute_codec is not None:
            require(callable(family.tensor_attribute_codec) and family.tensor_attribute_codec_id,
                    "Tensor attributes require a finite owner codec and codec ID")

    def _served(self, served):
        require(isinstance(served, Served) and served.run_id == self.provenance["run_id"], "Wrong served-workload run")
        require(served.stage in ("prefill", "decode") and served.workload_marker and served.replay_id,
                "Only explicitly marked served workload may emit fixtures")
        require(served.tp_rank == self.provenance["tp_rank"] and served.active_requests > 0 and served.active_tokens > 0,
                "Actual rank/request/token metadata required")

    def _reserve(self, h, size):
        require(size >= 0 and h.live_bytes + size <= self.budget.max_snapshot_bytes,
                "Per-call snapshot byte budget exceeded")
        require(self.live_bytes + size <= self.budget.max_live_snapshot_bytes, "Live graph snapshot budget exceeded")
        h.live_bytes += size
        self.live_bytes += size

    def _describe(self, h, bindings, roles, phase):
        require(set(bindings) == set(roles), "Bindings must exactly match the declared family roles")
        result, groups = {}, {}
        for name in sorted(bindings):
            value, role = bindings[name], roles[name]
            if value is None:
                require(role.optional, "Undeclared None tensor: " + name)
                result[name] = None
                continue
            key = storage_key(value)
            alias = h.aliases.setdefault(key, "s" + str(len(h.aliases)))
            kinds = h.storage_roles.setdefault(alias, set())
            kinds.add("readonly" if role.kind == "readonly" else "mutable")
            require(len(kinds) == 1, "A readonly weight aliases a declared mutable input/output")
            meta = tensor_metadata(value, alias)
            meta["role"] = asdict(role)
            if h.family.tensor_attribute_codec is not None:
                meta["attributes"] = controls_json(h.family.tensor_attribute_codec(phase, name, value))
            result[name] = meta
            groups.setdefault(alias, []).append((name, value, role, meta))
        return result, groups

    def _snapshots(self, h, bindings, roles, phase):
        metadata, groups = self._describe(h, bindings, roles, phase)
        torch = torch_module()
        # Work-defining scalar/vector controls are tiny snapshots on ALL ranks/sites/buckets.
        # Their values are resolved after served replay, never from startup construction.
        for name, role in roles.items():
            if not role.case_control:
                continue
            value = bindings[name]
            if value is None:
                h.tensor_controls[phase+"."+name] = None
                continue
            require(value.numel() <= self.budget.max_control_tensor_elements,
                    "Case-control tensor exceeds the declared small-control bound")
            self._reserve(h, value.numel()*value.element_size())
            snapshot = torch.empty_like(value)
            snapshot.copy_(value)
            h.tensor_controls[phase+"."+name] = snapshot
        if not h.selected:
            return metadata
        for alias, values in groups.items():
            descriptors, spans, watches = [], [], []
            for name, value, role, meta in values:
                watches.append((value, value._version))
                raw = raw_storage(value)
                if role.kind == "paged":
                    contract = role.paged
                    require(contract.access_contract == "only_indexed_units", "Paged read footprint must be explicit")
                    require(contract.indices in bindings, "Paged physical indices binding missing")
                    indices = bindings[contract.indices]
                    require(torch.is_tensor(indices) and indices.dtype in (torch.int32, torch.int64)
                            and indices.device == value.device, "Physical indices must be int32/int64 on the tensor device")
                    n = indices.numel()
                    require(0 < contract.unit_count and 0 < contract.unit_bytes <= contract.unit_stride_bytes
                            and contract.index_divisor > 0 and n <= contract.max_index_entries, "Invalid or unbounded paged contract")
                    end = contract.storage_offset_bytes + (contract.unit_count-1)*contract.unit_stride_bytes + contract.unit_bytes
                    require(contract.storage_offset_bytes == meta["storage_offset"]*meta["element_size"],
                            "Paged contract base offset differs from the actual view pointer")
                    require(0 <= contract.storage_offset_bytes and end <= raw.numel(), "Paged physical storage footprint escapes allocation")
                    self._reserve(h, n * (contract.unit_bytes + 40))
                    ids = indices.reshape(-1).to(dtype=torch.int64).clone()
                    physical = torch.div(ids, contract.index_divisor, rounding_mode="floor")
                    safe = physical.clamp(0, contract.unit_count-1)
                    rows = torch.as_strided(raw, (contract.unit_count, contract.unit_bytes),
                                           (contract.unit_stride_bytes, 1), contract.storage_offset_bytes)
                    gathered = torch.empty((n, contract.unit_bytes), dtype=torch.uint8, device=value.device)
                    torch.index_select(rows, 0, safe, out=gathered)
                    descriptors.append({"kind": "paged", "data": gathered, "indices": ids,
                                        "contract": contract, "binding": name})
                else:
                    spans.append((0, raw.numel()) if role.footprint == "full_storage" else view_span(meta))
            merged_spans = []
            for start, end in sorted(spans):
                if merged_spans and start <= merged_spans[-1][1]:
                    merged_spans[-1] = (merged_spans[-1][0], max(end, merged_spans[-1][1]))
                else:
                    merged_spans.append((start, end))
            for start, end in merged_spans:
                raw = raw_storage(values[0][1])[start:end]
                readonly = values[0][2].kind == "readonly"
                if readonly:
                    require(end-start <= self.budget.max_artifact_bytes, "Readonly weight exceeds artifact budget")
                    data = raw                              # static reference, no D2D clone
                else:
                    self._reserve(h, end-start)
                    data = torch.empty_like(raw)
                    data.copy_(raw)                         # captured D2D node when inside graph capture
                descriptors.append({"kind": "span", "data": data, "offset": start,
                    "readonly": readonly, "watches": watches,
                    "static_key": (storage_key(values[0][1]), start, end, tuple(sorted({v for _,v in watches})))})
            h.plans[phase][alias] = {"storage_nbytes": values[0][3]["storage_nbytes"],
                                     "descriptors": descriptors}
        return metadata

    def _begin(self, family, inputs, controls, mode, selected, **kwargs):
        self._family(family)
        require(not self._closed, "Recorder already sealed")
        control = controls_json(controls)
        require(len(canonical(control)) <= 65536, "Unbounded control metadata")
        h = Handle(family, mode, control, selected, **kwargs)
        h.temporary_input_refs = dict(inputs)  # keep storage identities live through output alias registration
        try:
            h.inputs = self._snapshots(h, inputs, family.input_roles, "inputs")
            return h
        except Exception as error:
            self.live_bytes -= h.live_bytes
            self._failure(error)
            raise

    def begin_eager(self, family, inputs, controls, served):
        require(not capturing(), "Use begin_graph while capturing")
        self._served(served)
        return self._begin(family, inputs, controls, "eager", not self.metadata_only, served=served)

    def begin_graph(self, family, inputs, controls, *, graph_id, slot_id, bucket):
        # Called during startup graph construction irrespective of the later workload phase marker.
        require(capturing(), "begin_graph must run during actual CUDA graph construction")
        require(graph_id and slot_id and bucket, "Explicit graph, operator slot and batch bucket required")
        graph = self.graphs.setdefault(graph_id, {})
        require(slot_id not in graph, "Duplicate graph operator slot; provide a distinct owner call-site ID")
        require(sum(len(x) for x in self.graphs.values()) < self.budget.max_graph_slots, "Graph metadata slot budget exceeded")
        selected = not self.metadata_only and bucket in self.graph_buckets and (self.snapshot_slots is None or slot_id in self.snapshot_slots)
        h = self._begin(family, inputs, controls, "graph", selected,
                        graph_id=graph_id, slot_id=slot_id, bucket=bucket)
        graph[slot_id] = h
        return h

    def finish(self, h, outputs):
        require(not h.finished, "Capture handle already finished")
        require((h.mode == "graph") == capturing(), "Input/output hooks crossed a capture boundary")
        try:
            h.outputs, _ = self._describe(h, outputs, h.family.output_roles, "outputs")
            h.outputs = self._snapshots(h, outputs, h.family.output_roles, "outputs")
            if h.mode == "eager":
                self._resolve_tensor_controls(h, h.served)
                self.case_counts[self._case_key(h, h.served)] += 1
            h.finished = True
            h.temporary_input_refs.clear()
            if h.mode == "eager":
                self._observe(h, h.served, count=False)
                self._periodic()
        except Exception as error:
            self._failure(error)
            raise
        finally:
            if h.mode == "eager":
                self.live_bytes -= h.live_bytes
                h.plans.clear()
                h.tensor_controls.clear()
                h.temporary_input_refs.clear()

    def abort(self, handle, error):
        """Owner hook calls this if the native operator itself raises before finish."""
        self._failure(CaptureError(handle.family.name + " native operator failed: " + str(error)))
        if handle.mode == "eager":
            self.live_bytes -= handle.live_bytes
            handle.live_bytes = 0
            handle.plans.clear()
            handle.tensor_controls.clear()
            handle.temporary_input_refs.clear()
        handle.finished = False

    def after_served_replay(self, graph_id, served):
        require(not capturing(), "Serialize only outside graph capture")
        require(not self._closed, "Recorder already sealed")
        self._served(served)
        try:
            require(graph_id in self.graphs, "Served replay has no instrumented graph metadata")
            seen = self.graph_replays.setdefault(graph_id, set())
            require(served.replay_id not in seen, "Duplicate served replay notification")
            # No startup/warmup notifications are accepted. Counters reflect all notified served replays.
            require(sum(len(x) for x in self.graph_replays.values()) < self.budget.max_replay_notifications,
                    "Replay notification metadata budget exceeded")
            seen.add(served.replay_id)
            handles = list(self.graphs[graph_id].values())
            for h in handles:
                require(h.finished, "Graph operator output hook missing")
                self._resolve_tensor_controls(h, served)
                self.case_counts[self._case_key(h, served)] += 1
            # Selected representatives first; unselected sites may share an already captured exact schema.
            for h in sorted(handles, key=lambda item: not item.selected):
                self._observe(h, served, count=False)
            self._periodic()
        except Exception as error:
            self._failure(error)
            raise

    def _resolve_tensor_controls(self, h, served):
        if not h.tensor_controls:
            h.resolved_tensor_controls = {}
            return
        self._synchronize(h)
        cpu = {name: None if value is None else value.cpu() for name, value in h.tensor_controls.items()}
        h.resolved_tensor_controls = controls_json(h.family.tensor_control_codec(cpu, served))
        require(len(canonical(h.resolved_tensor_controls)) <= 65536, "Unbounded decoded case-control metadata")

    def _case_schema(self, h, served):
        # Geometry/control/alias case identity is shared across TP ranks. Never include addresses,
        # layer weight bytes, served sequence IDs, or device ordinal in the case key.
        def geometry(bindings):
            return {name: None if meta is None else {k:v for k,v in meta.items() if k != "device"}
                    for name,meta in bindings.items()}
        return {"family": h.family.name, "source": h.family.source_sha256,
            "inputs": geometry(h.inputs), "outputs": geometry(h.outputs), "controls": h.controls,
            "tensor_controls": h.resolved_tensor_controls, "tensor_control_codec_id": h.family.tensor_control_codec_id,
            "tensor_attribute_codec_id": h.family.tensor_attribute_codec_id,
            "mode": h.mode, "stage": served.stage, "active_requests": served.active_requests,
            "active_tokens": served.active_tokens}

    def _case_key(self, h, served):
        schema = self._case_schema(h, served)
        key = h.family.name + "-" + digest(schema)[:24]
        if key not in self.schemas and len(self.schemas) >= self.budget.max_runtime_case_keys:
            self.metadata_case_overflow += 1
            error = CaptureError("Runtime case-key metadata budget exhausted")
            self._failure(error)
            raise error
        self.schemas[key] = schema
        return key

    def _observe(self, h, served, *, count=True):
        key = self._case_key(h, served)
        if count:
            self.case_counts[key] += 1                   # separate from first-fixture selection
        for phase in h.plans.values():
            for group in phase.values():
                for d in group["descriptors"]:
                    if d["kind"] == "span" and d["readonly"]:
                        require(all(t._version == version for t, version in d["watches"]),
                                "Declared readonly weight mutated during served work")
        if self.metadata_only or key in self.cases:
            return
        require(h.selected, "Served graph bucket/site schema was not snapshot-selected: " + str(h.bucket))
        require(len(self.cases) < self.budget.max_cases, "Observed runtime case exceeds fixture budget")
        self._synchronize(h)
        payload = {phase: self._serialize_phase(h.plans[phase]) for phase in ("inputs", "outputs")}
        record = {"schema": "served-tensor-fixture-v1", "case_key": key, "family": h.family.name,
            "source_sha256": h.family.source_sha256, "origin": "served_" + h.mode,
            "served": asdict(served), "controls": h.controls, "tensor_controls": h.resolved_tensor_controls,
            "tensor_control_codec_id": h.family.tensor_control_codec_id,
            "tensor_attribute_codec_id": h.family.tensor_attribute_codec_id,
            "inputs": h.inputs, "outputs": h.outputs,
            "payload": payload, "graph_bucket": h.bucket, "graph_id": h.graph_id, "slot_id": h.slot_id,
            "sampling": "first actual served occurrence of this runtime case", "startup_values": False,
            "provenance": self.provenance, "capture_overhead_is_not_performance_gain": True}
        self._write(self.root/(key+".json"), record)
        self.cases[key] = {"path": key+".json", "sha256": file_sha(self.root/(key+".json"))}
        self.snapshot_selections[key] += 1

    def _synchronize(self, h):
        torch = torch_module()
        devices = {str(d["data"].device) for phase in h.plans.values() for group in phase.values() for d in group["descriptors"]}
        devices.update(str(t.device) for t in h.tensor_controls.values() if t is not None)
        for device in devices:
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)          # all streams before host reads

    def _serialize_phase(self, plans):
        result = {}
        for alias, group in plans.items():
            descriptors = group["descriptors"]
            if all(d["kind"] == "span" for d in descriptors):
                segments = []
                for d in descriptors:
                    for tensor, version in d["watches"] if d["readonly"] else []:
                        require(tensor._version == version, "Declared readonly weight mutated before serialization")
                    cache_key = d["static_key"] if d["readonly"] else None
                    cached = self.static_blobs.get(cache_key) if cache_key is not None else None
                    blob = cached["blob"] if cached is not None else None
                    if blob is None:
                        blob = self._blob(d["data"])
                        if cache_key is not None:
                            # Strong references prevent recycled storage identities from becoming false dedup hits.
                            self.static_blobs[cache_key] = {"blob": blob, "references": [t for t, _ in d["watches"]]}
                    segments.append({"offset_bytes": d["offset"], **blob})
            else:
                parts = []
                for d in descriptors:
                    if d["kind"] == "span":
                        parts.append((d["offset"], d["data"].cpu().numpy().tobytes()))
                        continue
                    c = d["contract"]
                    ids = d["indices"].cpu().tolist()
                    rows = d["data"].cpu().numpy()
                    for original, row in zip(ids, rows):
                        if original in c.invalid_indices:
                            continue
                        unit = original // c.index_divisor
                        require(original >= 0 and 0 <= unit < c.unit_count, "Actual served physical index violates paged contract")
                        parts.append((c.storage_offset_bytes + unit*c.unit_stride_bytes, row.tobytes()))
                segments = [{"offset_bytes": offset, **self._blob(data)} for offset, data in union_ranges(parts)]
            result[alias] = {"storage_nbytes": group["storage_nbytes"], "segments": segments,
                "coverage": "union_of_explicit_declared_read_footprints", "unrecorded_bytes": "must_not_be_read"}
        return result

    def _blob(self, value):
        size = len(value) if isinstance(value, bytes) else value.numel()
        require(self.artifact_bytes + size <= self.budget.max_artifact_bytes, "Artifact byte budget exceeded")
        temp = self.root/"blobs"/("pending-"+str(time.time_ns()))
        h = hashlib.sha256()
        with temp.open("xb") as f:
            for offset in range(0, size, self.budget.chunk_bytes):
                part = value[offset:offset+self.budget.chunk_bytes]
                data = part if isinstance(value, bytes) else part.cpu().numpy().tobytes()
                h.update(data)
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        checksum = h.hexdigest()
        target = self.root/"blobs"/(checksum+".bin")
        if target.exists():
            require(target.stat().st_size == size, "Blob collision/size mismatch")
            temp.unlink()
        else:
            os.replace(temp, target)
            self.artifact_bytes += size
        return {"blob": "blobs/"+target.name, "sha256": checksum, "bytes": size}

    def _write(self, path, value):
        temp = path.with_suffix(path.suffix+".tmp")
        with temp.open("w") as f:
            f.write(canonical(value)+"\n"); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)

    def _periodic(self):
        if time.monotonic() - self._last_checkpoint >= self.budget.checkpoint_seconds:
            self.flush(reason="periodic")

    def flush(self, *, reason="profile_stop", required_cases=None):
        """Call from each TP worker's native stop-profile handler before retirement."""
        require(not capturing(), "Manifest checkpoint cannot run inside graph capture")
        required = set(required_cases or ())
        missing = (set(self.case_counts) | required) - set(self.cases)
        metadata_missing = required - set(self.case_counts)
        sealed = required_cases is not None and bool(self.case_counts) and not metadata_missing and not self.failures
        complete = sealed and not missing
        record = {"schema": "served-capture-rank-manifest-v1", "provenance": self.provenance,
            "complete": complete, "sealed": sealed, "metadata_only": self.metadata_only,
            "payload_capture_ranks": sorted(self.capture_ranks), "case_schemas": self.schemas,
            "missing_metadata_cases": sorted(metadata_missing),
            "required_cases_supplied": required_cases is not None, "required_cases": sorted(required),
            "missing_fixture_cases": sorted(missing), "runtime_case_counts": dict(self.case_counts),
            "snapshot_selections": dict(self.snapshot_selections), "cases": self.cases, "failures": self.failures,
            "additional_failure_count": self.failure_overflow,
            "unrepresented_metadata_case_notifications": self.metadata_case_overflow,
            "graph_buckets_snapshot_selected": sorted(self.graph_buckets),
            "graph_slots_snapshot_selected": None if self.snapshot_slots is None else sorted(self.snapshot_slots),
            "graph_replay_notifications": {g: len(ids) for g,ids in self.graph_replays.items()},
            "metadata_counts_include_all_notified_served_calls": not self.failures, "unnotified_replays_claimed": False,
            "live_snapshot_bytes": self.live_bytes, "artifact_bytes": self.artifact_bytes,
            "budget": asdict(self.budget), "checkpoint_reason": reason, "observed_unix": time.time(),
            "capture_overhead_is_not_performance_gain": True}
        self._write(self.root/"manifest.json", record)
        self._last_checkpoint = time.monotonic()
        return record

    def seal(self, required_cases):
        record = self.flush(reason="profile_stop", required_cases=required_cases)
        require(record["sealed"] and (self.metadata_only or record["complete"]),
                "Capture cannot be sealed: missing cases or prior capture failures")
        self._closed = True
        return record


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def restore_phase(root, fixture, phase, *, device="cpu", max_storage_bytes=1 << 30,
                  tensor_attribute_decoder=None, tensor_attribute_codec_id=None):
    """Preserve original offsets/aliasing; allocate full physical address span only with explicit budget."""
    torch = torch_module()
    root = Path(root).resolve()
    groups = fixture["payload"][phase]
    require(sum(g["storage_nbytes"] for g in groups.values()) <= max_storage_bytes,
            "Physical-storage reconstruction exceeds explicit restore budget")
    raw = {alias: torch.zeros(g["storage_nbytes"], dtype=torch.uint8, device=device) for alias,g in groups.items()}
    for alias, group in groups.items():
        for segment in group["segments"]:
            path = (root/segment["blob"]).resolve()
            require(path.is_relative_to(root) and file_sha(path) == segment["sha256"], "Artifact path/hash mismatch")
            require(path.stat().st_size == segment["bytes"], "Artifact byte count differs")
            start = segment["offset_bytes"]
            require(0 <= start <= start+segment["bytes"] <= raw[alias].numel(), "Snapshot segment escapes physical storage")
            actual_hash = hashlib.sha256()
            copied = 0
            with path.open("rb") as stream:
                for part in iter(lambda: stream.read(8 << 20), b""):
                    actual_hash.update(part)
                    data = torch.frombuffer(bytearray(part), dtype=torch.uint8)
                    raw[alias][start+copied:start+copied+len(part)].copy_(data.to(device))
                    copied += len(part)
            require(copied == segment["bytes"] and actual_hash.hexdigest() == segment["sha256"],
                    "Bytes changed while reconstructing storage")
    result = {}
    for name, meta in fixture[phase].items():
        if meta is None:
            result[name] = None
            continue
        dtype = getattr(torch, meta["dtype"].removeprefix("torch."), None)
        require(isinstance(dtype, torch.dtype), "Unknown recorded tensor dtype")
        result[name] = torch.empty(0,dtype=dtype,device=device).set_(raw[meta["alias"]].untyped_storage(),
            meta["storage_offset"], tuple(meta["shape"]), tuple(meta["stride"]))
        if meta.get("attributes"):
            require(callable(tensor_attribute_decoder) and tensor_attribute_codec_id == fixture["tensor_attribute_codec_id"],
                    "Tensor packing attributes require the matching finite owner restore codec")
            tensor_attribute_decoder(phase, name, result[name], meta["attributes"])
    return result


def verify_rank_manifests(paths, *, run_id, required_ranks, required_cases_by_rank, max_artifact_bytes=16 << 30):
    """Controller gate: all rank metadata keys must have a real fixture on a declared capture rank."""
    ranks, manifests, available = {}, {}, {}
    for path in paths:
        path = Path(path)
        record = json.loads(path.read_text())
        require(record["provenance"]["run_id"] == run_id and record["sealed"], "Wrong run or unsealed rank manifest")
        rank = record["provenance"]["tp_rank"]
        require(rank not in ranks, "Duplicate rank receipt")
        require(set(required_cases_by_rank[rank]) <= set(record["runtime_case_counts"]), "Rank metadata misses required cases")
        for key, case in record["cases"].items():
            fixture_path = (path.parent/case["path"]).resolve()
            require(fixture_path.is_relative_to(path.parent.resolve()) and file_sha(fixture_path) == case["sha256"], "Fixture manifest changed")
            fixture = json.loads(fixture_path.read_text())
            require(fixture["case_key"] == key and fixture["served"]["tp_rank"] in record["payload_capture_ranks"],
                    "Fixture does not belong to a declared payload rank")
            for phase in fixture["payload"].values():
                for group in phase.values():
                    for segment in group["segments"]:
                        blob = (path.parent/segment["blob"]).resolve()
                        require(blob.is_relative_to(path.parent.resolve()) and blob.stat().st_size == segment["bytes"]
                                and file_sha(blob) == segment["sha256"], "Fixture tensor blob changed")
            available.setdefault(key, {"rank": rank, "path": str(fixture_path), "sha256": case["sha256"]})
        manifests[rank] = record
        ranks[rank] = {"path": str(path), "sha256": file_sha(path), "case_count": len(record["runtime_case_counts"])}
    require(set(ranks) == set(required_ranks), "Missing or unexpected TP rank receipts")
    artifact_bytes = sum(record["artifact_bytes"] for record in manifests.values())
    require(artifact_bytes <= max_artifact_bytes, "Combined model artifact budget exceeded")
    for rank, record in manifests.items():
        require(set(record["runtime_case_counts"]) <= set(available),
                "Another rank observed a schema without any captured representative: " + str(rank))
    return {"complete": True, "run_id": run_id, "ranks": ranks, "fixture_representatives": available,
            "artifact_bytes": artifact_bytes, "max_artifact_bytes": max_artifact_bytes,
            "payloads_are_representative_not_every_value_history": True, "performance_gain_claim": False}
