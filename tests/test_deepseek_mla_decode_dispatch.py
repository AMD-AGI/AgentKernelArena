"""CPU proof of the captured decode branch and its protected edit boundary."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/headkernel/deepseek-v4-pro__unified_paged_attention_decode"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, shape, dtype="bfloat16", strides=None):
        self.shape = tuple(shape) if not isinstance(shape, int) else (shape,)
        self.dtype, self.device, self.is_cuda = dtype, "cuda", True
        if strides is None:
            strides, step = [], 1
            for dimension in reversed(self.shape):
                strides.insert(0, step)
                step *= dimension
        self.strides = strides

    def stride(self, axis):
        return self.strides[axis]

    def new_empty(self, shape, dtype):
        return Tensor(shape, dtype)


def native_control_flow():
    source = ast.parse((TASK / "source/native.py").read_text())
    names = {"_prev_pow2", "_kv_splits_heuristic", "_kernel_config",
             "_sparse_attn_v4_paged_decode_triton", "sparse_attn_v4_paged_decode"}
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((self.name, grid, args, kwargs))

    namespace = {"_TARGET_WG_PER_CU": 1.5, "_MAX_KV_SPLITS": 16, "_cu_count": lambda: 256,
                 "_is_gfx1250_supported": False, "_is_hip": True, "_FP8_GROUP_SIZE": 64,
                 "_FP8_DTYPE": "float8_e4m3fn", "LOG2E": 1.4426950408889634,
                 "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
                 "torch": SimpleNamespace(bfloat16="bfloat16", float16="float16", float32="float32",
                     empty_like=lambda t: Tensor(t.shape, t.dtype),
                     empty=lambda shape, dtype, device: Tensor(shape, dtype))}
    for name in ("_paged_decode_fused_kernel", "_paged_decode_split_kernel", "_paged_decode_reduce_kernel"):
        namespace[name] = Kernel(name)
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + nodes, type_ignores=[]), "<unchanged-native-wrapper>", "exec"), namespace)
    return namespace, calls


class DecodeDispatchTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((TASK / "cases.json").read_text())
        self.contract = load(TASK / "ut/dispatch_contract.py", "mla_decode_dispatch_contract")

    def test_three_actual_cases_select_split_then_reduce_in_unchanged_wrapper(self):
        self.assertTrue(self.contract.validate_dispatch(self.manifest))
        self.assertEqual(len(self.manifest["cases"]), 3)
        for case in self.manifest["cases"]:
            native, calls = native_control_flow()
            inputs = {}
            for name in ("q", "unified_kv", "kv_indices", "kv_indptr", "attn_sink"):
                row = case["tensors"]["arg." + name]
                inputs[name] = Tensor(row["shape"], row["dtype"], row["strides"])
            inputs["softmax_scale"] = case["scalars"]["arguments"]["softmax_scale"]
            inputs["kv_scales"] = None
            proof = self.contract.validate_runtime_dispatch(SimpleNamespace(**native), inputs)
            native["sparse_attn_v4_paged_decode"](**inputs)
            self.assertEqual(proof["kv_splits"], 4)
            self.assertEqual([call[0] for call in calls], list(self.contract.TARGETS))
            self.assertEqual(calls[0][1], (64, 1, 4))
            self.assertEqual(calls[1][1], (64, 16, 1))
            self.assertFalse(calls[0][3]["QUANT_KV"])

    def test_uncaptured_shapes_and_fp8_controls_are_rejected(self):
        for change in ("T", "H", "fp8", "scales"):
            altered = copy.deepcopy(self.manifest)
            case = altered["cases"][0]
            if change == "T": case["tensors"]["arg.q"]["shape"][0] = 1
            if change == "H": case["tensors"]["arg.q"]["shape"][1] = 32
            if change == "fp8": case["tensors"]["arg.unified_kv"]["dtype"] = "float8_e4m3fn"
            if change == "scales": case["scalars"]["arguments"]["kv_scales"] = {"tensor": "arg.kv_scales"}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "outside the captured"):
                self.contract.validate_dispatch(altered)
        native, _ = native_control_flow()
        native["_cu_count"] = lambda: 64
        with self.assertRaisesRegex(ValueError, "kv_splits=4"):
            self.contract.validate_runtime_dispatch(SimpleNamespace(**native), {"q": Tensor((64,16,512)), "kv_scales": None})

    def test_fused_body_remains_packaged_but_is_not_editable(self):
        config = yaml.safe_load((TASK / "config.yaml").read_text())
        source = json.loads((TASK / "provenance/SOURCE.json").read_text())
        policy = json.loads((TASK / "ut/source_guard_policy.json").read_text())
        targets = list(self.contract.TARGETS)
        self.assertEqual(config["target_kernel_functions"], targets)
        self.assertEqual(source["editable_sources"]["source/native.py"]["targets"], targets)
        self.assertEqual(policy["sources"]["source/native.py"]["targets"], targets)
        original = (TASK / "source/native.py").read_text()
        self.assertEqual(original, (TASK / "ut/baseline/native.py").read_text())
        sys.path.insert(0, str(TASK / "ut"))
        try:
            guard = load(TASK / "ut/source_guard.py", "mla_decode_source_guard")
        finally:
            sys.path.pop(0)
        for target in [*targets, "_paged_decode_fused_kernel"]:
            altered = ast.parse(original)
            function = next(n for n in altered.body if isinstance(n, ast.FunctionDef) and n.name == target)
            function.body.append(ast.Pass())
            candidate = ast.unparse(altered)
            if target in targets:
                guard.validate_python(candidate, original, targets, "triton")
            else:
                with self.assertRaisesRegex(ValueError, "frozen"):
                    guard.validate_python(candidate, original, targets, "triton")

    def test_evidence_binds_existing_cases_and_native_bytes(self):
        evidence = json.loads((TASK / "provenance/EDITABLE-DISPATCH.json").read_text())
        self.assertEqual(evidence["case_manifest_sha256"], hashlib.sha256((TASK / "cases.json").read_bytes()).hexdigest())
        self.assertEqual([x["case_id"] for x in evidence["cases"]], [x["case_id"] for x in self.manifest["cases"]])
        self.assertEqual([x["occurrences"] for x in evidence["cases"]], [x["occurrences"] for x in self.manifest["cases"]])
        for name, expected in evidence["dispatch_sources"].items():
            self.assertEqual(hashlib.sha256((TASK / name).read_bytes()).hexdigest(), expected)
        self.assertFalse(evidence["new_cases_added"])
        self.assertFalse(evidence["native_implementation_bytes_changed"])

    def test_complete_replay_policy_and_performance_budget_are_preserved(self):
        from agents.task_validator.launch_agent import _resolve_validation_timeouts
        config = yaml.safe_load((TASK / "config.yaml").read_text())
        limits = _resolve_validation_timeouts(config, {"compile_timeout":600,"correctness_timeout":600,"performance_timeout":600})
        self.assertEqual(limits[:3], (600,600,3600))
        policy = self.manifest["measurement"]
        self.assertEqual((policy["warmup_iterations"], policy["benchmark_iterations"]), (10,100))
        self.assertEqual(policy["correctness_seeds"], [42,43])
        self.assertEqual(policy["negative_controls"], ["no_op","wrong_output"])
        self.assertEqual([policy[k] for k in ["refresh_inputs","initialize_outputs","validate_outputs"]], ["each_replay"]*3)
        self.assertEqual(self.manifest["unresolved_legacy_m"], [1,8192])

    def test_registered_protocol_pins_the_dispatch_guard(self):
        from src.tools import custom_perf_protocols as protocols
        entrypoints = {TASK / "scripts/task_runner.py"}
        self.assertEqual(protocols.custom_protocol_family(TASK, entrypoints), "portable_case_contract")
        original = protocols._read_file
        def changed(task, name):
            value = original(task, name)
            return value + b"\n# changed guard\n" if name == "ut/dispatch_contract.py" else value
        with mock.patch.object(protocols, "_read_file", side_effect=changed), self.assertRaisesRegex(ValueError, "reviewed implementation"):
            protocols.custom_protocol_family(TASK, entrypoints)


if __name__ == "__main__":
    unittest.main()
