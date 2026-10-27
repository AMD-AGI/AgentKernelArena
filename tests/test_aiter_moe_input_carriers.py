"""Carrier checks and real CPU Torch storage tests; no workload qualification."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


TASK = Path(__file__).resolve().parents[1] / "tasks/aiter_workloads/deepseek_v4_pro_tp8_ep1_c64_8k1k_fused_moe"


@pytest.fixture
def carrier_loader(monkeypatch):
    monkeypatch.setitem(sys.modules, "reference", SimpleNamespace(validate_contract=lambda _: None))
    spec = importlib.util.spec_from_file_location("_moe_carrier_checks", TASK / "scripts/cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decoded, reads = {}, []

    class Tensor:
        layout = "dense"

    def tensor_load(stream, *, map_location, weights_only):
        reads.append((Path(stream.name).name, map_location, weights_only))
        return {key: Tensor() for key in decoded[stream.read()]}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        Tensor=Tensor, strided="dense", load=tensor_load))
    return module, decoded, reads


def _write_carrier(root, name, decoded, keys):
    contents = name.encode()
    (root / name).write_bytes(contents)
    decoded[contents] = keys
    return {"path": name, "sha256": hashlib.sha256(contents).hexdigest()}


def _case(activation, weights):
    return {"inputs": {**activation, "weights": weights, "input_weight_storages_disjoint": True, "call_kwargs": {
        "activation": "silu", "quant_type": "per_1x32", "gate_mode": "interleave",
        "doweight_stage1": False, "swiglu_limit": None,
        "hidden_pad": 0, "intermediate_pad": 0,
    }}}


def test_canonical_scale_shape_excludes_public_padding(carrier_loader):
    loader, _, _ = carrier_loader
    # TP-local down weights use logical K=384 and twelve scales per row;
    # the public a16w4 shuffle pads those scales to sixteen columns.
    weight = SimpleNamespace(shape=(384, 7168, 192))
    loader._validate_canonical_scale_shape(weight, SimpleNamespace(shape=(2752512, 12)))
    with pytest.raises(ValueError, match="unpadded per-1x32"):
        loader._validate_canonical_scale_shape(weight, SimpleNamespace(shape=(2752512, 16)))


def test_public_weight_attributes_preserve_captured_absence(carrier_loader):
    loader, _, _ = carrier_loader
    weights = {name: SimpleNamespace(is_shuffled=True) for name in ("w1", "w2")}
    absent = {name: {"is_shuffled": True} for name in weights}
    loader._restore_public_weight_attributes(weights, absent)
    assert all(value.is_shuffled is True and not hasattr(value, "gate_up_interleave") for value in weights.values())
    # A serialized attribute is not allowed to override the protected capture.
    weights["w1"].gate_up_interleave = True
    weights["w2"].gate_up_interleave = False
    loader._restore_public_weight_attributes(weights, absent)
    assert all(not hasattr(value, "gate_up_interleave") for value in weights.values())


def test_public_weight_attributes_restore_only_declared_markers(carrier_loader):
    loader, _, _ = carrier_loader
    weights = {name: SimpleNamespace() for name in ("w1", "w2")}
    declared = {"w1": {"is_shuffled": True}, "w2": {"is_shuffled": True, "gate_up_interleave": False}}
    loader._restore_public_weight_attributes(weights, declared)
    assert weights["w1"].is_shuffled is True and not hasattr(weights["w1"], "gate_up_interleave")
    assert weights["w2"].is_shuffled is True and weights["w2"].gate_up_interleave is False


@pytest.mark.parametrize("name,attributes", [
    ("w1", {}), ("w1", {"gate_up_interleave": True}),
    ("w1", {"is_shuffled": 1}), ("w1", {"is_shuffled": False}),
    ("w1", {"is_shuffled": True, "gate_up_interleave": False}),
    ("w2", {"is_shuffled": True, "gate_up_interleave": True}),
    ("w1", {"is_shuffled": True, "gate_up_interleave": 1}),
    ("w2", {"is_shuffled": True, "gate_up_interleave": None}),
    ("w1", {"is_shuffled": True, "unknown": True}),
])
def test_public_weight_attributes_reject_unknown_or_incorrect_markers(carrier_loader, name, attributes):
    loader, _, _ = carrier_loader
    weights = {key: SimpleNamespace() for key in ("w1", "w2")}
    declared = {key: {"is_shuffled": True} for key in weights}
    declared[name] = attributes
    with pytest.raises(ValueError, match="layout attributes"):
        loader._restore_public_weight_attributes(weights, declared)
    assert all(vars(value) == {} for value in weights.values())


def test_both_carriers_are_hash_checked_before_deserialization(tmp_path, carrier_loader):
    loader, decoded, reads = carrier_loader
    activation = _write_carrier(tmp_path, "case.pt", decoded, {"hidden_states", "topk_weight", "topk_ids"})
    weights = _write_carrier(tmp_path, "weights.pt", decoded, set())
    (tmp_path / "weights.pt").write_bytes(b"changed after binding")
    with pytest.raises(ValueError, match="digest differs"):
        loader.load_case(tmp_path, _case(activation, weights),
                         {"doweight_stage1": False, "swiglu_limit": None}, device="cpu")
    assert reads == [("case.pt", "cpu", True)]


@pytest.mark.parametrize("extra", [set(), {"hidden_states"}])
def test_weight_carrier_cannot_override_case_inputs(tmp_path, carrier_loader, extra):
    loader, decoded, reads = carrier_loader
    activation = _write_carrier(tmp_path, "case.pt", decoded, {"hidden_states", "topk_weight", "topk_ids"})
    keys = {"w1", "w2", "w1_scale", "w2_scale"}
    keys |= {"canonical_" + key for key in keys.copy()}
    weights = _write_carrier(tmp_path, "weights.pt", decoded, keys | extra)
    # A valid split proceeds to the next binding check; an overlap fails before
    # metadata or native imports. No tensor semantics are simulated here.
    reason = "Weight carrier has missing or unsupported" if extra else "Manifest must declare"
    with pytest.raises(ValueError, match=reason):
        loader.load_case(tmp_path, _case(activation, weights),
                         {"doweight_stage1": False, "swiglu_limit": None}, device="cpu")
    assert reads == [("case.pt", "cpu", True), ("weights.pt", "cpu", True)]


def test_weight_carrier_cannot_escape_task_root(tmp_path, carrier_loader):
    loader, decoded, reads = carrier_loader
    task = tmp_path / "task"
    task.mkdir()
    outside = _write_carrier(tmp_path, "outside.pt", decoded, set())
    with pytest.raises(ValueError, match="canonical task-relative path"):
        loader._load_tensor_file(task, {**outside, "path": "../outside.pt"}, "cpu")
    assert reads == []


def test_split_carriers_require_capture_attestation(tmp_path, carrier_loader):
    loader, _, reads = carrier_loader
    case = _case({}, {})
    del case["inputs"]["input_weight_storages_disjoint"]
    with pytest.raises(ValueError, match="attest disjoint"):
        loader.load_case(tmp_path, case, {"doweight_stage1": False, "swiglu_limit": None}, device="cpu")
    assert reads == []


@pytest.fixture
def storage_loader(monkeypatch):
    torch = pytest.importorskip("torch", reason="Real CPU Torch is required for storage semantics")
    monkeypatch.setitem(sys.modules, "reference", SimpleNamespace(validate_contract=lambda _: None))
    spec = importlib.util.spec_from_file_location("_moe_storage_checks", TASK / "scripts/cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, torch


def _metadata(shape, stride, offset, nbytes, group):
    return {
        "shape": shape, "stride": stride, "dtype": "torch.float32",
        "storage_offset": offset, "storage_nbytes": nbytes, "storage_group": group,
    }


def test_serialized_views_preserve_offsets_and_shared_storage(tmp_path, storage_loader):
    loader, torch = storage_loader
    backing = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    payload = {"left": backing[:, 1:5], "right": backing[1:, 2:]}
    path = tmp_path / "views.pt"
    torch.save(payload, path)
    descriptor = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    loaded = loader._load_tensor_file(tmp_path, descriptor, "cpu")
    loader._validate_public_tensors(loaded, {
        "left": _metadata([4, 4], [6, 1], 1, 96, 0),
        "right": _metadata([3, 4], [6, 1], 8, 96, 0),
    })
    loaded["left"][1, 1] = -3
    assert loaded["right"][0, 0].item() == -3


@pytest.mark.parametrize("shared,groups", [(False, (0, 0)), (True, (0, 1))])
def test_storage_group_equivalence_must_match_actual_aliases(storage_loader, shared, groups):
    loader, torch = storage_loader
    left = torch.arange(4, dtype=torch.float32)
    right = left.view(4) if shared else left.clone()
    with pytest.raises(ValueError, match="storage aliases"):
        loader._validate_public_tensors({"left": left, "right": right}, {
            "left": _metadata([4], [1], 0, 16, groups[0]),
            "right": _metadata([4], [1], 0, 16, groups[1]),
        })


@pytest.mark.parametrize("offset,nbytes", [(1, 32), (0, 40)])
def test_equal_shape_stride_values_do_not_hide_storage_changes(storage_loader, offset, nbytes):
    loader, torch = storage_loader
    backing = torch.zeros(nbytes // 4, dtype=torch.float32)
    value = backing[offset:offset + 4]
    with pytest.raises(ValueError, match="differs from protected manifest"):
        loader._validate_public_tensors({"value": value}, {"value": _metadata([4], [1], 0, 32, 0)})


def test_shared_storage_is_snapshotted_once(storage_loader):
    loader, torch = storage_loader
    backing = torch.arange(12, dtype=torch.float32)
    left, right = backing[1:6], backing[5:10]
    snapshots = loader.snapshot_inputs({"args": (left, right), "kwargs": {"repeated": left}})
    assert len(snapshots["tensors"]) == 2
    assert len(snapshots["storages"]) == 1
    loader.check_inputs_unchanged(snapshots)


def test_write_outside_public_view_is_rejected(storage_loader):
    loader, torch = storage_loader
    backing = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    value = backing[:, 1:5]
    visible_before = value.clone()
    snapshots = loader.snapshot_inputs({"args": (value,), "kwargs": {}})
    backing[:, 0].zero_()
    assert torch.equal(value, visible_before)
    with pytest.raises(AssertionError, match="backing storage changed"):
        loader.check_inputs_unchanged(snapshots)


@pytest.mark.parametrize("change", ["offset", "replacement", "size"])
def test_equal_visible_values_do_not_hide_storage_rebinding(storage_loader, change):
    loader, torch = storage_loader
    backing = torch.zeros(8, dtype=torch.float32)
    value = backing[:4]
    snapshots = loader.snapshot_inputs({"args": (value,), "kwargs": {}})
    if change == "offset":
        value.set_(value.untyped_storage(), 4, (4,), (1,))
    elif change == "replacement":
        value.set_(torch.zeros(4, dtype=torch.float32))
    else:
        value.untyped_storage().resize_(48)
    assert torch.equal(value, torch.zeros(4, dtype=torch.float32))
    with pytest.raises(AssertionError, match="storage"):
        loader.check_inputs_unchanged(snapshots)


@pytest.mark.parametrize("change", ["remove", "value", "type", "add"])
def test_public_layout_marker_changes_are_rejected(storage_loader, change):
    loader, torch = storage_loader
    value = torch.zeros(4, dtype=torch.float32)
    value.is_shuffled = True
    snapshots = loader.snapshot_inputs({"args": (value,), "kwargs": {}})
    if change == "remove":
        del value.is_shuffled
    elif change == "value":
        value.is_shuffled = False
    elif change == "type":
        value.is_shuffled = 1
    else:
        value.gate_up_interleave = True
    with pytest.raises(AssertionError, match="layout attributes"):
        loader.check_inputs_unchanged(snapshots)


def test_distinct_empty_storages_are_not_grouped_by_null_pointer(storage_loader):
    loader, torch = storage_loader
    left, right = torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)
    loader._validate_public_tensors({"left": left, "right": right}, {
        "left": _metadata([0], [1], 0, 0, 0), "right": _metadata([0], [1], 0, 0, 1),
    })
    snapshots = loader.snapshot_inputs({"args": (left, right), "kwargs": {}})
    assert len(snapshots["storages"]) == 2
    loader.check_inputs_unchanged(snapshots)
