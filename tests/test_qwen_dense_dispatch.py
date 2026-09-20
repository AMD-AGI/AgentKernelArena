"""CPU regression for the retained Qwen dispatcher table and actual candidate map."""
import ast
import csv
import functools
import importlib.util
import json
import logging
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = next((ROOT / "tasks/head_kernels/qwen3.8-2.4t-a95b-mxfp4").glob("*/*/dense_bf16_gemm_cluster"))
META = json.loads((TASK / "ut/meta.json").read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dispatch(monkeypatch):
    module = load("dense_dispatch_contract_test", TASK / "ut/dispatch_contract.py")
    monkeypatch.delenv("AITER_CONFIG_GEMM_BF16", raising=False)
    return module


class CsvFrame:
    """Supply only pandas' CSV/index surface; execute AITER's real lookup code."""
    def __init__(self, path):
        self.rows = []
        with open(path) as stream:
            for row in csv.DictReader(stream):
                for key in ("cu_num", "M", "N", "K", "solidx", "splitK"):
                    row[key] = int(row[key])
                for key in ("bias", "scaleAB", "bpreshuffle"):
                    assert row[key] in {"False", "True"}
                    row[key] = row[key] == "True"
                self.rows.append(row)

    def drop_duplicates(self):
        assert len({tuple(row.items()) for row in self.rows}) == len(self.rows)
        return self

    def set_index(self, keys):
        self.keys = keys
        return self

    def to_dict(self, orientation):
        assert orientation == "index"
        return {tuple(row[key] for key in self.keys): {key: value for key, value in row.items() if key not in self.keys}
                for row in self.rows}


class TensorShape:
    def __init__(self, shape, dtype=torch.bfloat16):
        self.shape, self.dtype = tuple(shape), dtype

    def dim(self):
        return len(self.shape)


def native_fixture(monkeypatch, tmp_path, *, missing_flydsl=False):
    api = load("qwen_aiter_api_test", ROOT / "tests/fixtures/qwen_dense_dispatch/aiter_config_api_fixture.py")
    api.AITER_ROOT_DIR = str(tmp_path)
    default = tmp_path / "default.csv"
    default.write_text((TASK / "ut/live_dispatch_rows.csv").read_text().splitlines()[0] + "\n")
    api.AITER_CONFIG_GEMM_BF16 = str(default)
    config = api.AITER_CONFIG()
    # Warm the same process-lifetime property that runtime preflight cached.
    assert config.AITER_CONFIG_GEMM_BF16_FILE == str(default)
    core = ModuleType("aiter.jit.core")
    core.AITER_CONFIGS = config
    monkeypatch.setitem(sys.modules, "aiter.jit.core", core)
    tree = ast.parse((TASK / "ut/baseline_ref/tuned_gemm.py.orig").read_text())
    selected = {"get_GEMM_A16W16_config_", "get_GEMM_A16W16_config", "is_skinny_default_shape", "gemm_a16w16"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in selected:
            if node.name == "gemm_a16w16":
                node.decorator_list = []
            body.append(node)
    code = ast.Module(body=body, type_ignores=[])
    parser = lambda name: None if missing_flydsl else {"kernelName": name}
    aiter = SimpleNamespace(ops=SimpleNamespace(flydsl=SimpleNamespace(gemm_kernels=SimpleNamespace(
        get_flydsl_splitk_hgemm_kernel_params=parser))))
    modules = []
    for name, filename in (("aiter.tuned_gemm", str(tmp_path / "native/tuned_gemm.py")),
                           ("tuned_gemm_candidate", str(TASK / "source/tuned_gemm_candidate.py"))):
        module = ModuleType(name)
        module.__file__ = filename
        events = []
        def backend(kind, events=events):
            def call(a, b, solution, bias, otype, *args, config=None):
                events.append((kind, solution, config["splitK"], config["kernelName"]))
                return TensorShape((a.shape[0], b.shape[0]), otype)
            return call
        module.__dict__.update(functools=functools, os=__import__("os"), pd=SimpleNamespace(read_csv=CsvFrame),
                               AITER_CONFIGS=config, AITER_LOG_TUNED_CONFIG=False, torch=torch, Tensor=torch.Tensor,
                               dtypes=SimpleNamespace(fp16=torch.float16, bf16=torch.bfloat16),
                               get_cu_num=lambda: 256, get_gfx=lambda: "gfx950", get_padded_m=lambda m, n, k, gl: m,
                               is_flydsl_available=lambda: True, aiter=aiter, logger=logging.getLogger(name),
                               save_shapes=lambda *args: None,
                               solMap={kind: backend(kind) for kind in ("torch", "hipblaslt", "asm", "flydsl")}, events=events)
        exec(compile(code, filename, "exec"), module.__dict__)
        modules.append(module)
        monkeypatch.setitem(sys.modules, name, module)
    native, candidate = modules
    native.original_dispatcher = native.gemm_a16w16
    native.gemm_a16w16 = candidate.gemm_a16w16
    return config, native, candidate, default


def test_five_authoritative_rows_are_complete_and_not_rewritten(dispatch):
    path = dispatch.prepare_environment()
    assert path == TASK / "ut/live_dispatch_rows.csv"
    original = subprocess.run(["git", "show", "189394fe:" + path.relative_to(ROOT).as_posix()],
                              cwd=ROOT, check=True, capture_output=True).stdout
    assert path.read_bytes() == original
    assert [(row["ledger_id"], row["expected_backend"], row["expected_solidx"], row["expected_split_k"])
            for row in META["workload"]["cases"]] == [
        ("hk04", "hipblaslt", 438243, 0), ("hk05", "torch", 0, 0),
        ("hk06", "flydsl", 3358, 8), ("hk07", "asm", 5, 3), ("hk08", "flydsl", 2138, 1)]


def test_late_environment_assignment_keeps_stale_cached_default(dispatch, monkeypatch, tmp_path):
    config, _, _, default = native_fixture(monkeypatch, tmp_path)
    dispatch.prepare_environment()
    assert config.AITER_CONFIG_GEMM_BF16_FILE == str(default)
    config.get_config_file.cache_clear()
    assert config.AITER_CONFIG_GEMM_BF16_FILE == str(TASK / "ut/live_dispatch_rows.csv")


def test_prepare_runtime_restores_exact_cfg_and_keeps_candidate_backend_map(dispatch, monkeypatch, tmp_path):
    config, native, candidate, _ = native_fixture(monkeypatch, tmp_path)
    candidate_map = candidate.solMap
    assert dispatch.prepare_runtime() is native
    assert candidate.solMap is candidate_map
    assert candidate.get_GEMM_A16W16_config is native.get_GEMM_A16W16_config
    assert candidate.get_GEMM_A16W16_config_ is native.get_GEMM_A16W16_config_
    assert dispatch.dispatch_table(native) is candidate_map
    monkeypatch.setattr(native.pd, "read_csv", lambda path: pytest.fail("warmed native dispatch re-read the protected CSV"))
    monkeypatch.setattr(candidate.pd, "read_csv", lambda path: pytest.fail("candidate opened the protected dispatch CSV"))
    for case in META["workload"]["cases"]:
        cfg = native.get_GEMM_A16W16_config(case["m"], case["n"], case["k"], False,
                                          "torch.bfloat16", "torch.bfloat16", False, False)
        assert dispatch.config_matches(cfg, case)
        result = candidate.gemm_a16w16(TensorShape((case["m"], case["k"])), TensorShape((case["n"], case["k"])))
        assert result.shape == (case["m"], case["n"])
    assert native.events == []
    assert candidate.events == [(case["expected_backend"], case["expected_solidx"], case["expected_split_k"], case["expected_kernel"])
                                for case in META["workload"]["cases"]]


def test_hk05_fallback_is_still_rejected_for_missing_native_kernel_name(dispatch):
    case = META["workload"]["cases"][1]
    assert case["ledger_id"] == "hk05"
    assert not dispatch.config_matches({"libtype": "torch", "solidx": 0}, case)
    assert dispatch.config_matches({"libtype": "torch", "solidx": 0, "kernelName": "native"}, case)


def test_missing_flydsl_catalog_entry_still_fails_exact_dispatch_check(dispatch, monkeypatch, tmp_path):
    native_fixture(monkeypatch, tmp_path, missing_flydsl=True)
    with pytest.raises(RuntimeError, match="live dispatcher config drift for hk06"):
        dispatch.prepare_runtime()


def test_modified_table_fails_before_setting_environment(dispatch, monkeypatch, tmp_path):
    path = tmp_path / "live_dispatch_rows.csv"
    path.write_text((TASK / "ut/live_dispatch_rows.csv").read_text().replace("438243", "0"))
    monkeypatch.setattr(dispatch, "HERE", tmp_path)
    with pytest.raises(RuntimeError, match="SHA-256"):
        dispatch.prepare_environment()
    assert "AITER_CONFIG_GEMM_BF16" not in __import__("os").environ


def test_original_source_abi_and_mismatch_predicate_are_unchanged():
    for relative in ("source/tuned_gemm_candidate.py", "ut/baseline_ref/tuned_gemm.py.orig", "scripts/source_abi.json"):
        path = TASK / relative
        original = subprocess.run(["git", "show", "189394fe:" + path.relative_to(ROOT).as_posix()],
                                  cwd=ROOT, check=True, capture_output=True).stdout
        assert path.read_bytes() == original
    path = TASK / "ut/unittest.py"
    old = subprocess.run(["git", "show", "189394fe:" + path.relative_to(ROOT).as_posix()],
                         cwd=ROOT, check=True, capture_output=True, text=True).stdout
    node = lambda text: next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_config_matches")
    assert ast.dump(node(old)) == ast.dump(node(path.read_text()))


def test_runner_sets_environment_before_loading_common_runtime(dispatch, monkeypatch):
    runner = load("dense_task_runner_test", TASK / "scripts/dense_task_runner.py")
    order = []
    def fake_load(name, path):
        order.append(name)
        if name == "dense_dispatch_contract":
            return dispatch
        assert __import__("os").environ[dispatch.ENVIRONMENT_KEY] == str(TASK / "ut/live_dispatch_rows.csv")
        return SimpleNamespace(main=lambda: 0)
    monkeypatch.setattr(runner, "load", fake_load)
    assert runner.main() == 0
    assert order == ["dense_dispatch_contract", "_dense_arena_runner"]


@pytest.mark.parametrize("case", META["workload"]["cases"], ids=lambda case: case["ledger_id"])
@pytest.mark.parametrize("matching_kernel", [True, False])
def test_current_backend_and_device_kernel_are_both_required(dispatch, monkeypatch, tmp_path, case, matching_kernel):
    _, native, candidate, _ = native_fixture(monkeypatch, tmp_path)
    dispatch.prepare_runtime()
    source = ast.parse((TASK / "ut/validate_selection.py").read_text())
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in {"validate_one", "_plain"}]
    fake_torch = SimpleNamespace(bfloat16=torch.bfloat16,
                                 cuda=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None))
    dispatch._baseline_callable = native.original_dispatcher
    def call(args):
        return dispatch.baseline_callable()(args["A"], args["B"])
    def profile(torch_module, function):
        output = function()
        names = [case["profile_match"] or "native_device_kernel"] if matching_kernel else []
        return output, names
    values = {"dispatch": dispatch, "_profile_names": profile,
              "importlib": SimpleNamespace(import_module=lambda name: fake_torch),
              "cases": SimpleNamespace(make_args=lambda row, seed: {
                  "A": TensorShape((row["m"], row["k"])), "B": TensorShape((row["n"], row["k"]))}, native_baseline_call=call)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(TASK / "ut/validate_selection.py"), "exec"), values)
    report = values["validate_one"](case)
    assert report["config_exact"] is True
    assert report["backend_hook_calls"] == 2
    assert report["output_shape_ok"] is True
    assert report["ok"] is matching_kernel
    assert report["runtime_config"]["kernelName"] == case["expected_kernel"]
    assert len(native.events) == 2 and candidate.events == []


def test_correctness_rejects_failed_fresh_kernel_selection_before_numerical_checks(dispatch, monkeypatch, tmp_path):
    _, native, _, _ = native_fixture(monkeypatch, tmp_path)
    dispatch.prepare_runtime()
    case = META["workload"]["cases"][0]
    tree = ast.parse((TASK / "ut/unittest.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"_run_case", "_runtime_config", "_config_matches"}]
    values = {"time": SimpleNamespace(time=lambda: 0), "META": META, "dispatch": dispatch,
              "_verify_selection": lambda case_id: {"ok": True},
              "cases": SimpleNamespace(case_map=lambda meta: {case["ledger_id"]: case}),
              "importlib": SimpleNamespace(import_module=lambda name: SimpleNamespace(
                  bfloat16=torch.bfloat16, cuda=SimpleNamespace(is_available=lambda: True))),
              "selection_validator": SimpleNamespace(validate_one=lambda case: {"ok": False, "device_events": []})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TASK / "ut/unittest.py"), "exec"), values)
    with pytest.raises(RuntimeError, match="live backend/kernel selection drift for hk04"):
        values["_run_case"]("hk04")


def test_case_contract_and_source_provenance_are_unchanged():
    path = TASK / "ut/meta.json"
    old = json.loads(subprocess.run(["git", "show", "189394fe:" + path.relative_to(ROOT).as_posix()],
                                   cwd=ROOT, check=True, capture_output=True, text=True).stdout)
    for key in ("workload", "case_entrypoints", "ledger_ids", "source_provenance", "baseline_frozen_by",
                "tol", "random_draws", "target_callable", "candidate_bind"):
        assert META[key] == old[key]


def test_real_shared_preload_reuses_actual_aliases_without_importing_candidate():
    code = f'''import importlib.util, json, sys
from pathlib import Path
import yaml
task = Path({str(TASK)!r})
spec = importlib.util.spec_from_file_location("dense_preload_bootstrap", task / "scripts/_trusted_worker.py")
bootstrap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bootstrap
spec.loader.exec_module(bootstrap)
harness = bootstrap.load("harness_lib", task / "ut/harness_lib.py")
cfg = yaml.safe_load((task / "config.yaml").read_text())
modules, files = bootstrap.load_declared_modules(task, cfg)
assert modules["dense_bf16_gemm_cases"] is modules["_headkernel_cases"]
assert sys.modules["harness_lib"] is harness
assert "tuned_gemm_candidate" not in sys.modules
assert "aiter.tuned_gemm" not in sys.modules
assert (task / "ut/unittest.py").resolve() in files
assert (task / "scripts/_bench.py").resolve() in files
print(json.dumps({{"aliases": sorted(modules), "candidate_imported": False}}))
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["candidate_imported"] is False
