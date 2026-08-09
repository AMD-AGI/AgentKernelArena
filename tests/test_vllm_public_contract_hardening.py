import ast
import hashlib
import struct
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "tasks" / "triton2triton" / "vllm"

RUNNER_PATHS = {
    name: TASK_ROOT / name / "scripts" / "task_runner.py"
    for name in (
        "triton_rms_norm",
        "triton_scaled_mm",
        "triton_topk_log_softmax",
    )
}


def _parse(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text()
    return source, ast.parse(source)


def _literal_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment: {name}")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function: {name}")


class VllmPublicContractHardeningTests(unittest.TestCase):
    def test_scoring_paths_remain_byte_for_byte_unchanged(self) -> None:
        expected = {
            "triton_rms_norm": {
                "shapes": [
                    (32, 128),
                    (64, 512),
                    (128, 1024),
                    (256, 2048),
                    (512, 4096),
                ],
                "performance_sha256": (
                    "7dfcbf9870cb9744cb5040d044d87737e7a8749715f714090d63993b176dbd31"
                ),
            },
            "triton_scaled_mm": {
                "shapes": [
                    (32, 64, 64, True, True, False),
                    (64, 128, 128, True, True, True),
                    (128, 256, 256, False, False, False),
                    (256, 512, 512, True, True, True),
                    (64, 256, 128, True, False, False),
                ],
                "performance_sha256": (
                    "8fc5b5e98d38c44aea109fb87d52ae9ec49b94eafe09ab4029b02eb111e9cd21"
                ),
            },
            "triton_topk_log_softmax": {
                "shapes": [
                    (4, 256, 3),
                    (8, 1024, 5),
                    (16, 4096, 10),
                    (32, 8192, 20),
                    (64, 32768, 10),
                ],
                "performance_sha256": (
                    "b333c359ccba4b524c025bc07229b775f0f105484533dd3b12f74ff68e57e526"
                ),
            },
        }

        for name, contract in expected.items():
            with self.subTest(task=name):
                source, tree = _parse(RUNNER_PATHS[name])
                self.assertEqual(
                    _literal_assignment(tree, "TEST_SHAPES"), contract["shapes"]
                )
                self.assertEqual(_literal_assignment(tree, "WARMUP_ITERATIONS"), 10)
                self.assertEqual(_literal_assignment(tree, "BENCHMARK_ITERATIONS"), 100)
                performance = ast.get_source_segment(
                    source, _function(tree, "run_performance")
                )
                self.assertIsNotNone(performance)
                digest = hashlib.sha256(performance.encode()).hexdigest()
                self.assertEqual(digest, contract["performance_sha256"])
                self.assertEqual(source.count('f"perf{test_idx + 1}"'), 2)

    def test_rms_norm_non_scoring_cases_cover_public_domain(self) -> None:
        _, tree = _parse(RUNNER_PATHS["triton_rms_norm"])
        cases = _literal_assignment(tree, "CORRECTNESS_CASES")

        by_name = {case["name"]: case for case in cases}
        self.assertEqual(
            set(by_name),
            {
                "non_power_float16",
                "non_power_bfloat16",
                "noncontiguous_float32",
                "large_dynamic_fallback",
            },
        )
        self.assertEqual(by_name["non_power_float16"]["shape"], (257,))
        self.assertEqual(
            by_name["non_power_bfloat16"]["shape"], (2, 3, 1025)
        )
        self.assertEqual(by_name["noncontiguous_float32"]["layout"], "padded")
        self.assertEqual(len(by_name["noncontiguous_float32"]["shape"]), 3)
        self.assertGreater(
            by_name["large_dynamic_fallback"]["shape"][1], 1 << 20
        )
        correctness_ast = ast.unparse(_function(tree, "run_correctness"))
        self.assertIn("result.untyped_storage().data_ptr()", correctness_ast)
        self.assertIn("frozen_backing", correctness_ast)

    def test_scaled_mm_cases_and_reference_preserve_upstream_order(self) -> None:
        _, tree = _parse(RUNNER_PATHS["triton_scaled_mm"])
        cases = _literal_assignment(tree, "CORRECTNESS_CASES")
        by_name = {case["name"]: case for case in cases}

        self.assertEqual(
            set(by_name),
            {
                "bias_after_output_cast",
                "sequential_low_precision_scales",
                "weak_contiguous_custom_tiles",
                "weak_contiguous_column_weight",
            },
        )
        self.assertEqual(
            by_name["bias_after_output_cast"]["data"], "bias_rounding"
        )
        self.assertEqual(
            by_name["sequential_low_precision_scales"]["scale_dtype"],
            "float16",
        )
        self.assertEqual(
            by_name["sequential_low_precision_scales"]["data"],
            "scale_product_underflow",
        )
        self.assertEqual(
            by_name["weak_contiguous_custom_tiles"]["layout"],
            "column_input_row_weight",
        )
        self.assertEqual(
            by_name["weak_contiguous_custom_tiles"]["tiles"], (16, 32, 32)
        )
        self.assertEqual(
            by_name["weak_contiguous_column_weight"]["layout"],
            "row_input_column_weight",
        )

        def fp16(value: float) -> float:
            return struct.unpack("e", struct.pack("e", value))[0]

        accumulator = fp16(1.0 / 3.0) * fp16(0.1875) * 16
        bias = fp16(-1.0)
        self.assertEqual(fp16(fp16(accumulator) + bias), 0.0)
        self.assertGreater(abs(fp16(accumulator + bias)), 5e-5)

        reference = _function(tree, "reference_scaled_mm")
        result_values = []
        for node in reference.body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "result"
            ):
                result_values.append(ast.unparse(node.value))
            elif isinstance(node, ast.If):
                for child in node.body:
                    if (
                        isinstance(child, ast.Assign)
                        and isinstance(child.targets[0], ast.Name)
                        and child.targets[0].id == "result"
                    ):
                        result_values.append(ast.unparse(child.value))
        self.assertEqual(
            result_values,
            [
                "a @ b",
                "sa * result",
                "result * sb.reshape(1, -1)",
                "result.to(out_dtype)",
                "result + bias",
            ],
        )

        correctness_ast = ast.unparse(_function(tree, "run_correctness"))
        self.assertIn("use_heuristic=False", correctness_ast)
        self.assertIn("_KernelLaunchRecorder", correctness_ast)
        self.assertIn(
            "(block_m, block_n, block_k) not in recorder.tiles",
            correctness_ast,
        )
        self.assertIn("protected_inputs", correctness_ast)
        self.assertIn("protected_backing", correctness_ast)
        self.assertIn("result.untyped_storage().data_ptr()", correctness_ast)

    def test_topk_cases_cover_stride_dtype_vocab_and_ids(self) -> None:
        _, tree = _parse(RUNNER_PATHS["triton_topk_log_softmax"])
        cases = _literal_assignment(tree, "CORRECTNESS_CASES")
        by_name = {case["name"]: case for case in cases}

        fp16_case = by_name["padded_stride_fp16_nonpower_duplicates"]
        self.assertEqual(fp16_case["logits_dtype"], "float16")
        self.assertNotEqual(fp16_case["vocab"] & (fp16_case["vocab"] - 1), 0)
        self.assertGreater(fp16_case["row_padding"], 0)

        large_case = by_name["large_vocab_batch1_bfloat16_int32_duplicates"]
        self.assertEqual(large_case["batch"], 1)
        self.assertEqual(large_case["logits_dtype"], "bfloat16")
        self.assertEqual(large_case["token_ids_dtype"], "int32")
        self.assertGreater(large_case["vocab"], 32768)

        factory_ast = ast.unparse(
            _function(tree, "_make_topk_correctness_inputs")
        )
        self.assertIn("logits_storage[:, :vocab]", factory_ast)
        self.assertIn("duplicate_unsorted_ids", factory_ast)

        factory = _function(tree, "_make_topk_correctness_inputs")
        ids = None
        for node in factory.body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "duplicate_unsorted_ids"
            ):
                ids = [ast.unparse(element) for element in node.value.elts]
        self.assertIsNotNone(ids)
        self.assertNotEqual(ids, sorted(ids))
        self.assertLess(len(set(ids)), len(ids))

        correctness_ast = ast.unparse(_function(tree, "run_correctness"))
        self.assertIn("original_logits_storage", correctness_ast)
        self.assertIn("token_ids.untyped_storage().data_ptr()", correctness_ast)


if __name__ == "__main__":
    unittest.main()
