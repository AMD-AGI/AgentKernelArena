"""CPU evidence for generated Qwen values, preserved structures and parent gates."""
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from types import SimpleNamespace

import pytest
from head_kernel_generated_test_utils import generated_helper, generated_task
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "tasks/head_kernels/qwen3.8-2.4t-a95b-mxfp4"
NAMES = ("fused_moe_2stage_mxfp4", "fused_recurrent_gated_delta_rule_decode", "paged_attention_decode")
TASKS = {name: next(QWEN.glob(f"*/*/{name}")) for name in NAMES}


def load(name, path=None):
    path = path or generated_helper("qwen", name + ".py")
    alias = "test_qwen_" + name
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


contract = load("generated_contract")
worker = load("generated_worker")
controller = load("generated_correctness")
extractor = load("extract_contract")


def descriptor(shape, *, name="query", dtype="torch.float32", recipe="finite_normal", group=None, **extra):
    count = 1
    stride = []
    for size in reversed(shape):
        stride.insert(0, count)
        count *= size
    itemsize = torch.empty((), dtype=getattr(torch, dtype.removeprefix("torch."))).element_size()
    return {"tensor": True, "name": name, "shape": shape, "stride": stride, "dtype": dtype,
            "storage_offset": 0, "storage_nbytes": count * itemsize, "storage_group": group or name,
            "tensor_attrs": {}, "recipe": recipe, **extra}


def small_blob():
    return {"shared": {"weights": descriptor([4, 8], name="w1", dtype="torch.float4_e2m1fn_x2",
                                              recipe="packed_fp4", tensor_attrs={"is_shuffled": True})},
            "records": [{"source_sig": "case", "regime": "decode", "args": {"sequence": "tuple", "items": []},
                         "kwargs": {"query": descriptor([2, 4]), "w1": {"__shared__": "weights"},
                                    "w1_alias": {"__shared__": "weights"},
                                    "scale": descriptor([2, 4], name="w1_scale", dtype="torch.uint8", recipe="positive_e8m0_scale"),
                                    "topk_weight": descriptor([2, 3], name="topk_weight", recipe="normalized_route_weights"),
                                    "indices": descriptor([3], name="indices", dtype="torch.int32", recipe="captured_structure",
                                                          values=extractor.integer_encoding([5, 1, 3]))},
                         "kwargs_before": {}, "output_contract": None}]}


def test_all_five_case_identities_and_archival_hashes_are_preserved():
    assert sum((task / "ut/generated_cases.json").stat().st_size for task in TASKS.values()) == 507947
    for name, task in TASKS.items():
        data = contract.load_contract(task / "ut")
        meta = json.loads((task / "ut/meta.json").read_text())
        assert data["source_reference_sha256"] == meta["archival_capture"]["reference_io_sha256"]
        assert meta["generated_inputs"]["contract_sha256"] == contract.digest(task / "ut/generated_cases.json")
        assert len(data["records"]) == meta["num_cases"] == (1 if name == "paged_attention_decode" else 2)
        assert set(meta["generated_inputs"]["case_ids"]) == {r["sig"] for r in meta["workload"]["cases"]}


def test_moe_retains_full_packed_weights_shuffled_flags_and_real_expert_mapping():
    task = TASKS["fused_moe_2stage_mxfp4"]
    data = contract.load_contract(task / "ut")
    assert data["shared"]["w1"]["shape"] == [64, 4096, 4096]
    assert data["shared"]["w1"]["storage_nbytes"] == 1073741824
    assert data["shared"]["w2"]["shape"] == [64, 8192, 1024]
    assert data["shared"]["w2"]["storage_nbytes"] == 536870912
    assert [row["kwargs"]["hidden_states"]["shape"][0] for row in data["records"]] == [8192, 64]
    for row in data["records"]:
        values = row["kwargs"]
        assert values["w1_scale"]["dtype"] == values["w2_scale"]["dtype"] == "torch.uint8"
        assert values["w1_scale"]["recipe"] == values["w2_scale"]["recipe"] == "positive_e8m0_scale"
        mask = contract.decode_integers(values["expert_mask"]["values"])
        ids = contract.decode_integers(values["topk_ids"]["values"])
        assert len(mask) == 512 and len(ids) == values["hidden_states"]["shape"][0] * 10
        assert set(mask) == {0, 1}
        assert [index for index, selected in enumerate(mask) if selected] == list(range(64))
        assert min(ids) >= 0 and max(ids) < len(mask)
        assert values["doweight_stage1"] is False
    # Physical layouts allocate the complete production storage on meta, without
    # allocating gigabytes in the CPU unit test.
    built = contract.build_blob(data, 7, torch, "meta")
    assert built["shared"]["w1"].shape == (64, 4096, 4096)
    assert built["shared"]["w1"].untyped_storage().nbytes() == 1073741824
    cases_source = (task / "ut/cases.py").read_text()
    assert 'shared[key].is_shuffled = True' in cases_source


def test_gated_delta_keeps_199_slots_exact_state_indices_and_before_snapshots():
    task = TASKS["fused_recurrent_gated_delta_rule_decode"]
    data = contract.load_contract(task / "ut")
    meta = json.loads((task / "ut/meta.json").read_text())
    expected = {row["B"]: row["state_indices"] for row in meta["workload"]["cases"]}
    for row in data["records"]:
        kwargs = row["kwargs"]
        batch = kwargs["mixed_qkv"]["shape"][0]
        assert contract.decode_integers(kwargs["ssm_state_indices"]["values"]) == expected[batch]
        state = row["kwargs_before"]["initial_state"]
        assert state["shape"] == [199, 16, 128, 128]
        assert state["stride"] == [262144, 16384, 128, 1]
        assert state["storage_nbytes"] == 208666624
        assert row["output_contract"]["sequence"] == "tuple"
        assert row["output_contract"]["items"][1]["shape"] == state["shape"]
    plan = controller.expected_profiles(meta)
    assert plan["transitions"] == ["decode_b1_live|transition0", "decode_b64_live|transition1", "decode_b1_live|transition2"]
    assert plan["replay"] == ["decode_b64_live|replay0", "decode_b1_live|replay1", "decode_b64_live|replay2"]


def test_paged_attention_keeps_all_521351_pages_198_lengths_and_full_workspace():
    data = contract.load_contract(TASKS["paged_attention_decode"] / "ut")
    args = data["records"][0]["args"]["items"]
    assert len(args) == 19
    assert args[1]["shape"] == [71926272] and args[1]["storage_nbytes"] == 71926272
    assert args[3]["shape"] == args[4]["shape"] == [521351, 1, 1, 256]
    assert args[3]["stride"] == args[4]["stride"] == [256, 256, 256, 1]
    indices = contract.decode_integers(args[7]["values"])
    indptr = contract.decode_integers(args[6]["values"])
    lengths = contract.decode_integers(args[8]["values"])
    assert len(indices) == len(set(indices)) == 521351
    assert set(indices) == set(range(521351))
    assert len(indptr) == 65 and indptr[0] == 0 and indptr[-1] == 521351
    assert all(a < b for a, b in zip(indptr, indptr[1:]))
    assert len(lengths) == 198
    built = contract.build_blob(data, 3, torch, "meta")["records"][0]["args"]
    assert built[1].shape == (71926272,)
    assert built[3].shape == built[4].shape == (521351, 1, 1, 256)


def test_generated_bytes_are_seeded_valid_and_keep_exact_structure_and_aliases():
    data = small_blob()
    first = contract.build_blob(data, 11, torch, "cpu")["records"][0]["kwargs"]
    same = contract.build_blob(data, 11, torch, "cpu")["records"][0]["kwargs"]
    changed = contract.build_blob(data, 12, torch, "cpu")["records"][0]["kwargs"]
    assert torch.equal(first["query"], same["query"]) and not torch.equal(first["query"], changed["query"])
    assert torch.equal(first["w1"].view(torch.uint8), same["w1"].view(torch.uint8))
    assert not torch.equal(first["w1"].view(torch.uint8), changed["w1"].view(torch.uint8))
    assert first["w1"].is_shuffled is True and first["w1"].data_ptr() == first["w1_alias"].data_ptr()
    assert set(first["scale"].flatten().tolist()) <= set(range(120, 125))
    assert bool((first["topk_weight"] > 0).all())
    assert torch.allclose(first["topk_weight"].sum(-1), torch.ones(2))
    assert first["indices"].tolist() == changed["indices"].tolist() == [5, 1, 3]
    cloned = contract.clone_tree(first, torch)
    assert cloned["w1"].is_shuffled is True
    assert cloned["w1"].data_ptr() == cloned["w1_alias"].data_ptr() != first["w1"].data_ptr()


@pytest.mark.parametrize("values", [[], [2], [0, 1, 2, 4, 6], [7, 7, 7], [5, -1, 3, 2, -3]])
def test_affine_structural_encoding_is_lossless(values):
    assert contract.decode_integers(extractor.integer_encoding(values)) == values


def state_kwargs():
    return {"initial_state": torch.arange(7 * 2 * 3 * 4, dtype=torch.float32).reshape(7, 2, 3, 4) / 500,
            "out": torch.zeros(2, 1, 2, 3), "ssm_state_indices": torch.tensor([5, 1], dtype=torch.int32),
            "query": torch.arange(4, dtype=torch.float32)}


def state_kernel(**kwargs):
    indices = kwargs["ssm_state_indices"].long()
    kwargs["initial_state"][indices] += 0.1
    kwargs["out"].fill_(kwargs["initial_state"][indices].sum().item())
    return kwargs["out"], kwargs["initial_state"]


def test_state_aliases_all_unchanged_rows_and_repeated_transitions_are_checked():
    kwargs = state_kwargs()
    baseline = worker.invoke(NAMES[1], state_kernel, (), kwargs, torch, contract, [], lambda: None)
    candidate_kwargs = state_kwargs()
    candidate = worker.invoke(NAMES[1], state_kernel, (), candidate_kwargs, torch, contract, [], lambda: None)
    assert contract.compare_output(candidate, baseline, 0.02, torch)
    second = worker.invoke(NAMES[1], state_kernel, (), kwargs, torch, contract, [], lambda: None)
    assert not contract.compare_output(second, baseline, 0.02, torch)
    def damage_unselected(**values):
        out = state_kernel(**values)
        values["initial_state"][0] += 0.01
        return out
    with pytest.raises(RuntimeError, match="unselected"):
        worker.invoke(NAMES[1], damage_unselected, (), state_kwargs(), torch, contract, [], lambda: None)
    def break_alias(**values):
        out, state = state_kernel(**values)
        return out.clone(), state
    with pytest.raises(RuntimeError, match="aliases"):
        worker.invoke(NAMES[1], break_alias, (), state_kwargs(), torch, contract, [], lambda: None)


def test_parent_rejects_changed_state_values_and_preserves_full_state_rms_tolerance():
    kwargs = state_kwargs()
    expected = worker.invoke(NAMES[1], state_kernel, (), kwargs, torch, contract, [], lambda: None)
    changed = copy.deepcopy(expected)
    state = contract.decode_output(changed["items"][1]["selected_rows"], torch)
    state.flatten()[-1] += 1024
    changed["items"][1]["selected_rows"]["data"] = contract.encode_output(state, torch)["data"]
    assert not contract.compare_output(changed, expected, 0.02, torch)
    assert expected["items"][1]["full_state_rms"] == float(kwargs["initial_state"].square().mean().sqrt())
    changed = copy.deepcopy(expected)
    changed["items"][1]["unselected_rows_unchanged"] = False
    assert not contract.compare_output(changed, expected, 0.02, torch)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
def test_integer_and_boolean_outputs_compare_exactly(dtype):
    expected = torch.tensor([1, 1], dtype=dtype)
    changed = torch.tensor([0, 1], dtype=dtype)
    assert not contract.compare_output(contract.encode_output(changed, torch), contract.encode_output(expected, torch), 10, torch)


def test_parent_rejects_stale_case_sets_wrong_strides_and_truncated_outputs():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    encoded = contract.encode_output(tensor, torch)
    forged = {**encoded, "stride": [1, 2]}
    assert not contract.compare_output(forged, encoded, 0.02, torch)
    with pytest.raises(ValueError, match="byte count"):
        contract.decode_output({**encoded, "data": "AAAA"}, torch)
    result = {"schema_version": 1, "profile": "eager", "seed": 7, "reference": False,
              "rows": [{"id": "case", "output": encoded}]}
    proc = SimpleNamespace(returncode=0, stdout=controller.PREFIX + json.dumps(result), stderr="")
    assert len(controller.parse_worker(proc, "eager", 7, False, ["case"])) == 1
    for profile, seed, ids in [("replay", 7, ["case"]), ("eager", 8, ["case"]), ("eager", 7, ["other"])]:
        with pytest.raises(RuntimeError, match="stale, incomplete or reordered"):
            controller.parse_worker(proc, profile, seed, False, ids)


def test_numeric_source_edits_are_compared_in_separate_cpu_processes(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("import json, torch\nfrom pathlib import Path\nimport importlib.util\n"
                      f"spec=importlib.util.spec_from_file_location('contract',{str(generated_helper("qwen", "generated_contract.py"))!r})\n"
                      "c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)\n"
                      "namespace={};exec(Path(__import__('sys').argv[1]).read_text(),namespace)\n"
                      "x=torch.tensor([0.25,-0.5,0.875])\n"
                      "print(json.dumps(c.encode_output(namespace['kernel'](x),torch)))\n")
    reference = tmp_path / "reference.py"
    candidate = tmp_path / "candidate.py"
    reference.write_text("def kernel(x): return x.square()\n")
    candidate.write_text(reference.read_text())
    def run(path):
        return json.loads(subprocess.run([sys.executable, str(script), str(path)], check=True,
                                         capture_output=True, text=True).stdout)
    baseline = run(reference)
    assert contract.compare_output(run(candidate), baseline, 0.02, torch)
    candidate.write_text("def kernel(x): return x.square()+1\n")
    assert not contract.compare_output(run(candidate), baseline, 0.02, torch)


def protected_cpu_worker(tmp_path, candidate_source=None, preload_attack=False, declaration_attack=None):
    """Use the real protected worker, module preloader and monitor on CPU tensors."""
    task = tmp_path / "protected-task"
    for folder in ("scripts", "source", "ut/overlay", "build"):
        (task / folder).mkdir(parents=True)
    for name in ("_trusted_worker.py", "runtime_integrity.py"):
        shutil.copyfile(ROOT / "tasks/head_kernels/_support" / name, task / "scripts" / name)
    shutil.copyfile(generated_helper("qwen", "generated_worker.py"), task / "scripts/generated_worker.py")
    shutil.copyfile(generated_helper("qwen", "generated_contract.py"), task / "ut/generated_contract.py")
    shutil.copyfile(ROOT / "src/tools/perf/aka_benchmark.py", task / "scripts/_aka_benchmark.py")
    (task / "scripts/runtime_preflight.py").write_text(
        "def require_runtime(config, *, phase='complete'):\n"
        "    assert phase in ('environment', 'complete')\n"
        "    return {'cpu_fixture': True, 'phase': phase}\n"
    )
    (task / "ut/harness_lib.py").write_text(
        "import torch\ndef _torch(): return torch\n"
        "def correct(out, ref, tol): return (bool(torch.allclose(out, ref)), 0)\n"
        "def _correct_one(out, ref, tol): return correct(out, ref, tol)\n"
        "def flatten_outputs(out): return [out]\n"
        "def to_device_like(ref, dev): return ref\n")
    (task / "ut/cases.py").write_text(
        "import sys, torch\n"
        "def baseline(hidden_states): return hidden_states.square()\n"
        "def current_callable():\n"
        "    return sys.modules['candidate_kernel'].kernel if 'candidate_kernel' in sys.modules else baseline\n"
        "def load_live_cases(device='cuda', seed=0):\n"
        "    x=torch.randn((2,3), generator=torch.Generator(device='cpu').manual_seed(seed))\n"
        "    return {('decode',2): {'sig':'decode_m2_live','m':2,'regime':'decode','args':{'kwargs':{'hidden_states':x}}}}\n")
    if preload_attack:
        with (task / "ut/cases.py").open("a") as stream:
            stream.write(f"open({str(task / 'source/kernel.py')!r}).read()\n")
    compact = {"schema_version": 1, "task_id": "cpu-test", "source_reference_sha256": "a" * 64,
               "correctness_case_count": 1, "records": [{"regime": "decode", "kwargs": {"hidden_states": descriptor([2, 3])},
                                                          "output_contract": descriptor([2, 3])}]}
    raw = json.dumps(compact)
    (task / "ut/generated_cases.json").write_text(raw)
    (task / "ut/meta.json").write_text(json.dumps({"task_id": "cpu-test", "num_cases": 1,
        "archival_capture": {"reference_io_sha256": "a" * 64},
        "generated_inputs": {"kernel": NAMES[0], "case_ids": ["decode_m2_live"], "contract_file": "generated_cases.json",
                             "contract_sha256": hashlib.sha256(raw.encode()).hexdigest()}}))
    config = {"source_file_path": ["source/kernel.py"], "target_kernel_functions": ["kernel"],
              "headkernel": {"generated_input_revision": "generated-v1", "trusted_worker_modules": {
                  "generated_contract": "ut/generated_contract.py", "_qwen_generated_cases": "ut/cases.py",
                  "_headkernel_cases": "ut/cases.py", "_qwen_generated_worker": "scripts/generated_worker.py"}}}
    if declaration_attack == "missing-map":
        config["headkernel"].pop("trusted_worker_modules")
    elif declaration_attack == "undeclared-entrypoint":
        config["headkernel"]["trusted_worker_modules"].pop("_qwen_generated_worker")
    (task / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (task / "source/kernel.py").write_text(candidate_source or "def kernel(hidden_states): return hidden_states.square()\n")
    (task / "ut/overlay/sitecustomize.py").write_text(
        "import importlib.util, sys\n"
        f"s=importlib.util.spec_from_file_location('candidate_kernel',{str(task / 'source/kernel.py')!r})\n"
        "m=importlib.util.module_from_spec(s);sys.modules['candidate_kernel']=m;s.loader.exec_module(m)\n")
    command = [sys.executable, str(task / "scripts/_trusted_worker.py"),
               "--task-root", str(task), "--script", str(task / "scripts/generated_worker.py"),
               "--completion", str(task / "build/completion.json"), "--nonce", "cpu-test-nonce"]
    if candidate_source is not None:
        command += ["--overlay", str(task / "ut/overlay")]
    command += ["--", "--ut", str(task / "ut"), "--profile", "eager", "--seed", "7"]
    if candidate_source is None:
        command += ["--reference"]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
    return proc, task


def test_actual_protected_worker_accepts_baseline_and_rejects_wrong_numeric_source(tmp_path):
    base, base_task = protected_cpu_worker(tmp_path / "baseline")
    assert base.returncode == 0, base.stderr
    same, same_task = protected_cpu_worker(tmp_path / "same", "def kernel(hidden_states): return hidden_states.square()\n")
    assert same.returncode == 0, same.stderr
    wrong, _ = protected_cpu_worker(tmp_path / "wrong", "def kernel(hidden_states): return hidden_states.square()+1\n")
    assert wrong.returncode == 0, wrong.stderr
    ids = ["decode_m2_live|call0", "decode_m2_live|call1"]
    expected = controller.parse_worker(base, "eager", 7, True, ids)
    controller.compare_workers(expected, controller.parse_worker(same, "eager", 7, False, ids), contract, 0.02, torch)
    with pytest.raises(RuntimeError, match="correctness mismatch"):
        controller.compare_workers(expected, controller.parse_worker(wrong, "eager", 7, False, ids), contract, 0.02, torch)
    for task in (base_task, same_task):
        assert json.loads((task / "build/completion.json").read_text())["nonce"] == "cpu-test-nonce"


@pytest.mark.parametrize("attack", [
    "import generated_contract\ngenerated_contract.tensor_signature = lambda value: {}\n",
    "import generated_contract\ngenerated_contract.encode_output = lambda value, torch: None\n",
    "import sys\nsys.modules['_qwen_generated_worker'].unchanged = lambda *args: None\n",
    "import sys\nsys.modules['generated_contract'] = object()\n",
    "import torch\ntorch.Tensor.stride = lambda self: (3, 1)\n",
])
def test_actual_preloaded_helpers_cannot_be_rebound_or_forge_strides(tmp_path, attack):
    source = "def kernel(hidden_states):\n" + "".join("    " + line + "\n" for line in attack.splitlines())
    source += "    return hidden_states.square()\n"
    proc, task = protected_cpu_worker(tmp_path, source)
    assert proc.returncode != 0
    assert "IntegrityError" in proc.stderr
    assert not (task / "build/completion.json").exists()


def test_helper_preload_cannot_access_editable_source_before_attestation(tmp_path):
    proc, task = protected_cpu_worker(tmp_path, preload_attack=True)
    assert proc.returncode != 0
    assert "preload attempted candidate access before attestation" in proc.stderr
    assert not (task / "build/completion.json").exists()


@pytest.mark.parametrize("declaration_attack,message", [
    ("missing-map", "generated_input_revision requires"),
    ("undeclared-entrypoint", "worker entrypoint is missing"),
])
def test_generated_mode_requires_declared_actual_entrypoint(tmp_path, declaration_attack, message):
    proc, task = protected_cpu_worker(tmp_path, declaration_attack=declaration_attack)
    assert proc.returncode != 0
    assert message in proc.stderr
    assert not (task / "build/completion.json").exists()


def test_numpy_serialization_forgery_cannot_hide_wrong_square_plus_one(tmp_path):
    attack = (
        "def kernel(hidden_states):\n"
        "    import torch\n"
        "    actual_numpy = torch.Tensor.numpy\n"
        "    expected_bytes = actual_numpy(hidden_states.square().detach().cpu().contiguous().view(torch.uint8))\n"
        "    torch.Tensor.numpy = lambda self: expected_bytes\n"
        "    return hidden_states.square() + 1\n"
    )
    proc, task = protected_cpu_worker(tmp_path, attack)
    assert proc.returncode != 0
    assert "IntegrityError" in proc.stderr and "numpy" in proc.stderr
    assert not (task / "build/completion.json").exists()


class FakeCuda:
    def __init__(self, failure=None):
        self.failure = failure
        self.active = False
        self.operation = None

    def is_available(self):
        return self.failure != "no_device"

    def Stream(self):
        return SimpleNamespace(wait_stream=lambda other: None)

    current_stream = Stream

    def stream(self, stream):
        return nullcontext()

    def synchronize(self):
        pass

    def CUDAGraph(self):
        def replay():
            if self.failure == "replay":
                raise RuntimeError("injected replay failure")
            if self.failure != "stale":
                self.operation()
        return SimpleNamespace(replay=replay)

    @contextmanager
    def graph(self, graph):
        if self.failure == "capture":
            raise RuntimeError("injected capture failure")
        self.active = True
        try:
            yield
        finally:
            self.active = False


@pytest.mark.parametrize("failure", [None, "stale", "capture", "replay", "no_device"])
def test_actual_graph_worker_requires_changed_input_and_restore_replay(monkeypatch, failure):
    cuda = FakeCuda(failure)
    monkeypatch.setattr(torch, "cuda", cuda)
    def fn(hidden_states):
        result = hidden_states.square()
        if cuda.active:
            cuda.operation = lambda: result.copy_(hidden_states.square())
        return result
    case = {"sig": "small", "args": {"kwargs": {"hidden_states": torch.arange(6, dtype=torch.float32).reshape(2, 3)}}}
    expected = worker.run_graph(NAMES[0], fn, case, torch, contract, True, 1, lambda: None)
    if failure in {"capture", "replay", "no_device"}:
        with pytest.raises(RuntimeError):
            worker.run_graph(NAMES[0], fn, case, torch, contract, False, 1, lambda: None)
    else:
        actual = worker.run_graph(NAMES[0], fn, case, torch, contract, False, 1, lambda: None)
        if failure == "stale":
            with pytest.raises(RuntimeError, match="correctness mismatch"):
                controller.compare_workers(expected, actual, contract, 0.02, torch)
        else:
            controller.compare_workers(expected, actual, contract, 0.02, torch)
            assert actual[0]["output"] == actual[2]["output"]
            assert actual[0]["output"] != actual[1]["output"]


@pytest.mark.parametrize("failure", [None, "stale"])
def test_state_graph_preserves_masked_slots_and_large_small_large_order(monkeypatch, failure):
    cuda = FakeCuda(failure)
    monkeypatch.setattr(torch, "cuda", cuda)
    rows = []
    for batch, indices in ((2, [5, 1]), (1, [2])):
        kwargs = state_kwargs()
        kwargs.update(mixed_qkv=torch.ones(batch, 4), a=torch.ones(batch, 2), b=torch.ones(batch, 2),
                      A_log=torch.ones(2), dt_bias=torch.ones(2), scale=0.5, use_qk_l2norm_in_kernel=True,
                      out=torch.zeros(batch, 1, 2, 3), ssm_state_indices=torch.tensor(indices, dtype=torch.int32))
        rows.append({"sig": f"batch{batch}", "m": batch, "args": {"kwargs": kwargs}})
    def fn(**kwargs):
        selected = kwargs["ssm_state_indices"]
        selected = selected[selected >= 0].long()
        kwargs["initial_state"][selected] += 0.1
        kwargs["out"].fill_(kwargs["initial_state"][selected].sum().item())
        if cuda.active:
            cuda.operation = lambda: fn(**kwargs)
        return kwargs["out"], kwargs["initial_state"]
    expected = worker.run_state_graph(fn, rows, torch, contract, True, lambda: None)
    actual = worker.run_state_graph(fn, rows, torch, contract, False, lambda: None)
    assert [row["id"] for row in actual] == ["batch2|replay0", "batch1|replay1", "batch2|replay2"]
    if failure:
        with pytest.raises(RuntimeError, match="correctness mismatch"):
            controller.compare_workers(expected, actual, contract, 0.02, torch)
    else:
        controller.compare_workers(expected, actual, contract, 0.02, torch)
        assert actual[0]["output"] == actual[2]["output"]


def test_controller_passes_only_profiles_and_fresh_seeds_to_independent_workers():
    task = TASKS[NAMES[1]]
    meta = json.loads((task / "ut/meta.json").read_text())
    profiles = controller.expected_profiles(meta)
    calls = []
    def run_worker(script, arguments, overlay, remaining, candidate, **kwargs):
        profile = arguments[arguments.index("--profile") + 1]
        seed = int(arguments[arguments.index("--seed") + 1])
        calls.append((profile, seed, candidate, arguments, kwargs))
        value = {"schema_version": 1, "profile": profile, "seed": seed, "reference": not candidate,
                 "rows": [{"id": case_id, "output": contract.encode_output(torch.ones(3), torch)} for case_id in profiles[profile]]}
        return SimpleNamespace(returncode=0, stdout=controller.PREFIX + json.dumps(value), stderr="")
    runner = SimpleNamespace(TASK_DIR=task, UT_DIR=task / "ut", overlays=lambda: ("baseline", "candidate"),
                             run_worker=run_worker, write_report=lambda *args: None)
    ok, error = controller.run_correctness(runner, {}, 30)
    assert ok, error
    assert len(calls) == 2 * (meta["random_draws"] + 2)
    for reference, candidate in zip(calls[::2], calls[1::2]):
        assert reference[:2] == candidate[:2]
        assert reference[2] is False and candidate[2] is True
        assert "--reference" in reference[3] and "--reference" not in candidate[3]
        assert all("output" not in argument and "expected" not in argument for argument in candidate[3])
        assert set(candidate[4]) == {"cwd"}


def test_generated_source_abi_timing_and_rmsnorm_contract_are_unchanged():
    for task in TASKS.values():
        config = yaml.safe_load((task / "config.yaml").read_text())
        for relative in [*config["source_file_path"], "scripts/source_abi.json", "scripts/task_runner.py"]:
            path = task / relative
            original = subprocess.run(["git", "show", "b66e373d:" + path.relative_to(ROOT).as_posix()],
                                      cwd=ROOT, check=True, capture_output=True).stdout
            assert path.read_bytes() == original
        for mode in ("compile", "correctness", "performance"):
            assert "generated_task_runner.py" in config[f"{mode}_command"][0]
    for name in ("gemma_fused_add_rmsnorm",):
        task, = QWEN.glob(f"*/*/{name}")
        for relative in ("config.yaml", "ut/meta.json", "ut/cases.py"):
            path = task / relative
            if path.exists():
                original = subprocess.run(["git", "show", "b66e373d:" + path.relative_to(ROOT).as_posix()],
                                          cwd=ROOT, check=True, capture_output=True).stdout
                if relative == "config.yaml":
                    current_config = yaml.safe_load(path.read_text())
                    original_config = yaml.safe_load(original)
                    assert current_config["headkernel"].pop("trusted_worker_modules") == {
                        "fused_add_rmsnorm_cases": "ut/cases.py",
                        "_headkernel_cases": "ut/cases.py",
                        "_legacy_correctness": "ut/unittest.py",
                        "_bench": "scripts/_bench.py",
                    }
                    for record in (current_config, original_config):
                        for key in ("docker", "runtime", "capture_runtime"):
                            record["headkernel"].pop(key, None)
                    assert current_config == original_config
                else:
                    assert path.read_bytes() == original


def test_framework_materializes_timer_for_the_copied_generated_worker(tmp_path):
    from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
    task = TASKS["paged_attention_decode"]
    workspace = tmp_path / "workspace"
    shutil.copytree(task, workspace, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "build"))
    materialized = materialize_perf_helpers_in_workspace(workspace, root=ROOT)
    helper = workspace / "scripts/_aka_benchmark.py"
    assert helper in materialized
    assert helper.read_bytes() == (ROOT / "src/tools/perf/aka_benchmark.py").read_bytes()
    assert (workspace / "scripts/_trusted_worker.py").is_file()
    assert not (workspace / "scripts/generated_trusted_worker.py").exists()
    assert contract.load_contract(workspace / "ut")["correctness_case_count"] == 1
