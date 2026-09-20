"""Draft contract and integrity regressions; no GPU validation is implied."""

import array
import base64
import importlib.util
import json
from pathlib import Path
import sys
import types
import zlib

import pytest
from head_kernel_generated_test_utils import generated_helper, generated_task
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = sorted(
    path.parent
    for path in (ROOT / "tasks/head_kernels/deepseek-v4-pro").rglob("config.yaml")
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helper = load("_deepseek_contract_test", generated_helper("deepseek", "generated_contract.py"))
worker = load("_deepseek_worker_test", generated_helper("deepseek", "generated_worker.py"))
controller = load("_deepseek_controller_test", generated_helper("deepseek", "generated_correctness.py"))


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
def test_committed_record_contract_without_tensor_archives(task):
    meta = json.loads((task / "ut/meta.json").read_text())
    contract = helper.load_contract(task / "ut")
    assert len(contract["records"]) == meta["num_cases"]
    assert len(contract["records"]) == (4 if task.name == "dsa_sparse_mla_attn" else 5)
    assert not (task / "ut/reference_io.pt").exists()
    assert json.loads((task / "scripts/artifacts.json").read_text()) == []
    assert "reference_io_sha256" not in meta
    assert meta["generated_inputs"]["gpu_validated"] is False
    assert "source_byte_sample" not in (task / "ut/generated_cases.json").read_text()
    profiles = helper.profiles(meta)
    assert len(profiles["eager"]) == (6 if contract["task_kind"] == "dsa" else 5)
    assert len(profiles["random"]) == (7 if contract["task_kind"] == "dsa" else 5)
    assert len(profiles["sequence"]) == (6 if contract["task_kind"] == "dsa" else 256)
    assert (
        len(profiles["replay"]) == 3 and profiles["replay"][0] == profiles["replay"][-1]
    )


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
def test_protected_contract_hash_and_case_identity(task, tmp_path):
    meta = json.loads((task / "ut/meta.json").read_text())
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    original = (task / "ut/generated_cases.json").read_bytes()
    (tmp_path / "generated_cases.json").write_bytes(original + b" ")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        helper.load_contract(tmp_path)


def test_dsa_preserves_exact_packing_and_partial_masks():
    task = next(path for path in TASKS if path.name == "dsa_sparse_mla_attn")
    records = helper.load_contract(task / "ut")["records"]
    assert [len(row["undefined_partial_rows"]) for row in records] == [0, 4256, 0, 8]
    assert [len(row["undefined_rows"]) for row in records] == [0, 0, 0, 0]
    for row in records:
        for key in ("k_cache", "extra_k_cache"):
            node = row["kwargs"][key]
            assert node["shape"][2:] == [1, 584]
            assert node["stride0"] > node["shape"][1] * 584
            assert node["block_payload_bytes"] == node["shape"][1] * 576
            assert node["block_scale_bytes"] == node["shape"][1] * 8
            ids = array.array("q")
            ids.frombytes(helper.decode_bytes(node["rows_idx"]["payload"]))
            assert min(ids) >= 0 and max(ids) < node["shape"][0]
        for key in (
            "indices",
            "topk_length",
            "extra_indices_in_kvcache",
            "extra_topk_length",
        ):
            assert row["kwargs"][key]["recipe"] == "captured_structure"
        partial = helper.decode_bytes(row["undefined_partial_payload"])
        assert len(partial) == len(row["undefined_partial_rows"]) * 64 * 512
        assert set(partial) <= set((0, 1))
        if partial:
            assert 0 in partial and 1 in partial


@pytest.mark.parametrize(
    "task",
    [path for path in TASKS if path.name != "dsa_sparse_mla_attn"],
    ids=lambda path: path.name,
)
def test_moe_retains_routing_weights_and_storage_dedup(task):
    contract = helper.load_contract(task / "ut")
    weight_groups = []
    for row in contract["records"]:
        if contract["task_kind"] == "moe1":
            nodes = row["kwargs"]
            routing = [
                nodes[name]
                for name in (
                    "sorted_token_ids",
                    "sorted_expert_ids",
                    "num_valid_ids",
                    "topk_ids",
                )
            ]
            weights = [nodes["w1"]]
        else:
            routing = row["args"][3:6] + [row["kwargs"]["sorted_weights"]]
            weights = row["args"][1:3]
            assert row["args"][6]["recipe"] == "zero_output"
        for node in routing:
            assert node["recipe"] == "captured_structure"
            assert len(helper.decode_bytes(node["payload"])) == node["payload"]["bytes"]
        assert all(node["recipe"] == "generated_mxfp4_weight" for node in weights)
        weight_groups.append(tuple(node["storage_group"] for node in weights))
    assert len(set(weight_groups)) == 1


def test_codec_rejects_wrong_lengths_trailing_data_and_bombs():
    def node(raw, n):
        return {
            "codec": "zlib-base64",
            "bytes": n,
            "data": base64.b64encode(raw).decode(),
        }

    assert helper.decode_bytes(node(zlib.compress(b"abc"), 3)) == b"abc"
    for raw, n in [
        (zlib.compress(b"abc"), 2),
        (zlib.compress(b"abc") + b"extra", 3),
        (zlib.compress(b"a" * 1000), 10),
    ]:
        with pytest.raises(ValueError):
            helper.decode_bytes(node(raw, n))


def test_parent_rejects_missing_reordered_and_stale_results():
    payload = {
        "schema_version": 1,
        "profile": "sequence",
        "index": 0,
        "seed": 10,
        "reference": False,
        "rows": [{"id": "a", "output": 1}, {"id": "b", "output": 2}],
    }

    def proc(value):
        return types.SimpleNamespace(
            returncode=0, stdout=controller.PREFIX + json.dumps(value), stderr=""
        )

    assert (
        len(
            controller.parse_worker(proc(payload), "sequence", 0, 10, False, ["a", "b"])
        )
        == 2
    )
    for altered in [
        {**payload, "seed": 9},
        {**payload, "rows": payload["rows"][::-1]},
        {**payload, "rows": payload["rows"][:1]},
    ]:
        with pytest.raises(RuntimeError, match="stale, incomplete or reordered"):
            controller.parse_worker(proc(altered), "sequence", 0, 10, False, ["a", "b"])


def test_required_artifact_manifest_excludes_all_three_archives():
    data = json.loads((ROOT / "tasks/head_kernels/artifacts.json").read_text())
    assert not any(
        row["task"].startswith("deepseek-v4-pro/") for row in data["artifacts"]
    )
    assert data["artifact_count"] == len(data["artifacts"])
    assert data["total_size_bytes"] == sum(
        row["size_bytes"] for row in data["artifacts"]
    )


@pytest.mark.parametrize("task", TASKS, ids=lambda path: path.name)
def test_timing_runtime_controls_and_kernel_source_unchanged(task):
    # These shared controls are copied verbatim from the published base.
    for name in (
        "task_runner.py",
        "_bench.py",
        "runtime_preflight.py",
        "runtime_integrity.py",
        "_trusted_worker.py",
    ):
        assert (task / "scripts" / name).read_bytes() == (
            ROOT / "tasks/head_kernels/_support" / name
        ).read_bytes()
    cfg = yaml.safe_load((task / "config.yaml").read_text())
    assert all(
        "generated_task_runner.py" in cfg[mode + "_command"][0]
        for mode in ("compile", "correctness", "performance")
    )
    assert "WARMUP_ITERATIONS = 10" in (task / "scripts/task_runner.py").read_text()
    assert "BENCHMARK_ITERATIONS = 100" in (task / "scripts/task_runner.py").read_text()


def test_direct_binding_loads_frozen_and_edited_files_in_package_context(
    tmp_path, monkeypatch
):
    task = next(path for path in TASKS if path.name == "dsa_sparse_mla_attn")
    captured = json.loads((task / "ut/generated_cases.json").read_text())
    gen = types.ModuleType("generated_contract")
    gen.__file__ = str(task / "ut/generated_contract.py")
    gen.load_contract = lambda root: captured
    gen.digest = helper.digest
    monkeypatch.setitem(sys.modules, "generated_contract", gen)
    package = types.ModuleType("_fixture_deepseek")
    package.__path__ = []
    target = types.ModuleType("_fixture_deepseek.target")
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, target.__name__, target)
    (tmp_path / "ut").mkdir()
    (tmp_path / "source").mkdir()
    baseline = tmp_path / "ut/baseline.py.orig"
    source = tmp_path / "source/kernel.py"
    baseline.write_text("def selected():\n    return 'frozen-reference'\n")
    source.write_text("def selected():\n    return 'edited-candidate'\n")

    def prepare(name):
        module = load(name, task / "ut/cases.py")
        module.HERE = tmp_path / "ut"
        module.META["target_callable"] = "_fixture_deepseek.target:selected"
        module.META["generated_inputs"].update(
            baseline_source="baseline.py.orig",
            baseline_sha256=helper.digest(baseline),
            candidate_source="source/kernel.py",
            source_package="_fixture_deepseek",
        )
        return module

    monkeypatch.delenv("GEAK_ACTIVE_TASK_CANDIDATE", raising=False)
    reference = prepare("_case_reference_binding")
    assert reference._resolve()() == "frozen-reference"
    monkeypatch.setenv("GEAK_ACTIVE_TASK_CANDIDATE", "task")
    candidate = prepare("_case_candidate_binding")
    assert candidate._resolve()() == "edited-candidate"
    assert (
        candidate._BOUND.__package__
        if hasattr(candidate._BOUND, "__package__")
        else candidate._BOUND.__module__.startswith("_fixture_deepseek.")
    )


def test_shipped_contract_distinguishes_two_inherited_dsa_slices():
    task = next(path for path in TASKS if path.name == "dsa_sparse_mla_attn")
    meta = json.loads((task / "ut/meta.json").read_text())
    records = {row["sig"]: row for row in helper.load_contract(task / "ut")["records"]}
    derived = [row for row in meta["case_specs"] if row.get("derived_from")]
    assert {row["name"] for row in derived} == {"decode_m1_x128", "decode_m1_x1024"}
    for row in meta["case_specs"]:
        captured_m = records[row["source_sig"]]["kwargs"]["q"]["shape"][0]
        if row in derived:
            assert captured_m == 64 and row["m"] == 1 and not row["scored"]
        else:
            assert captured_m == row["m"] and row["scored"]


def test_dsa_scores_only_four_observed_calls_and_keeps_both_robustness_probes(
    monkeypatch,
):
    task = next(path for path in TASKS if path.name == "dsa_sparse_mla_attn")
    generator = load("_score_generator", task / "ut/generated_contract.py")
    monkeypatch.setitem(sys.modules, "generated_contract", generator)
    cases = load("_score_cases", task / "ut/cases.py")
    expected = [
        "prefill_m8192_x64",
        "prefill_m8192_x1024",
        "decode_m64_x128",
        "decode_m64_x1024",
    ]
    assert [row["name"] for row in cases.scored_case_specs(cases.META)] == expected
    assert [row["sig"] for row in cases.timing_cases(None, cases.META)] == expected
    profiles = generator.profiles(cases.META)
    probes = {"decode_m1_x128", "decode_m1_x1024"}
    assert probes <= set(profiles["eager"])
    assert probes <= set(profiles["random"])
    assert probes <= set(profiles["sequence"])
    assert all(
        not row["scored"] for row in cases.META["case_specs"] if row["name"] in probes
    )
    cases.META["workload"]["cases"].append({"name": "decode_m1_x128"})
    with pytest.raises(RuntimeError, match="observed captured calls"):
        cases.scored_case_specs(cases.META)


def test_all_generated_tasks_declare_the_actual_preloaded_worker_aliases():
    for task in TASKS:
        cfg = yaml.safe_load((task / "config.yaml").read_text())
        assert cfg["harness_path"] == "scripts/_bench.py"
        assert cfg["headkernel"]["trusted_worker_modules"] == {
            "generated_contract": "ut/generated_contract.py",
            "generated_worker": "scripts/generated_worker.py",
            "_deepseek_generated_cases": "ut/cases.py",
            "_headkernel_cases": "ut/cases.py",
            "_bench": "scripts/_bench.py",
        }
        assert (
            "class Attestation"
            not in (task / "scripts/generated_worker.py").read_text()
        )
