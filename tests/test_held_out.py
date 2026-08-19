import tempfile
import unittest
import logging
from pathlib import Path
from unittest import mock

import yaml

from src.held_out.generate_heldout import discover_tasks
from src.held_out.injection import (
    apply_injection,
    replace_get_inputs,
    replace_test_shapes,
)
from src.held_out.run_heldout_eval import (
    _select_heldout_comparison_cases,
    evaluate_single_task,
    resolve_task_id,
)
from src.testcases import TestCaseResult


class HeldOutInjectionTests(unittest.TestCase):
    def test_replaces_nested_test_shapes_without_touching_following_code(self) -> None:
        source = """TEST_SHAPES = [
    (64, [128, 256]),
]
AFTER = True
"""
        replacement = """TEST_SHAPES = [
    (37, [131, 257]),
]"""

        modified = replace_test_shapes(source, replacement)

        self.assertIn("(37, [131, 257])", modified)
        self.assertNotIn("(64, [128, 256])", modified)
        self.assertIn("AFTER = True", modified)

    def test_replaces_get_inputs_function(self) -> None:
        source = """def get_inputs():
    return [1]

def get_init_inputs():
    return []
"""
        replacement = """def get_inputs():
    return [37, 131]"""

        modified = replace_get_inputs(source, replacement)

        self.assertIn("return [37, 131]", modified)
        self.assertNotIn("return [1]", modified)
        self.assertIn("def get_init_inputs():", modified)

    def test_applies_raw_replacement_to_workspace_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            target = workspace / "test_kernel.py"
            target.write_text("SHAPES = [(32, 32)]\n")

            applied = apply_injection(
                workspace,
                {
                    "file": "test_kernel.py",
                    "find_marker": "raw_replace",
                    "old_code": "[(32, 32)]",
                    "replacement_code": "[(37, 131)]",
                },
            )

            self.assertTrue(applied)
            self.assertEqual(target.read_text(), "SHAPES = [(37, 131)]\n")


class HeldOutDiscoveryTests(unittest.TestCase):
    def test_discovers_only_supported_task_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tasks_root = Path(temporary_directory)
            supported = tasks_root / "hip2hip" / "gpumode" / "GELU"
            unsupported = tasks_root / "repository" / "aiter" / "example"
            supported.mkdir(parents=True)
            unsupported.mkdir(parents=True)
            (supported / "config.yaml").write_text("task_type: hip2hip\n")
            (unsupported / "config.yaml").write_text("task_type: repository\n")

            discovered = discover_tasks(tasks_root)

            self.assertEqual(discovered, [("hip2hip/gpumode/GELU", supported)])

    def test_resolves_task_id_from_task_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "task_result.yaml").write_text(
                yaml.safe_dump({"task_name": "triton2triton/vllm/triton_rms_norm"})
            )

            self.assertEqual(
                resolve_task_id(workspace),
                "triton2triton/vllm/triton_rms_norm",
            )


class HeldOutBenchmarkMethodTests(unittest.TestCase):
    @staticmethod
    def _case(time_ms: float, method: str) -> TestCaseResult:
        return TestCaseResult(
            test_case_id="case",
            shape=[1],
            execution_time_ms=time_ms,
            metadata={"benchmark_method": method},
        )

    def test_candidate_event_fallback_cannot_replace_graph_baseline(self) -> None:
        baseline = [self._case(2.0, "cuda_graph")]
        optimized = [self._case(3.0, "cuda_event_fallback")]

        selected, valid_optimized, consistent, mismatches = (
            _select_heldout_comparison_cases(
                baseline, optimized, logging.getLogger(__name__)
            )
        )

        self.assertEqual(selected[0].execution_time_ms, 2.0)
        self.assertEqual(
            selected[0].metadata["benchmark_method"],
            "cuda_graph",
        )
        self.assertEqual(valid_optimized, optimized)
        self.assertFalse(consistent)
        self.assertEqual(len(mismatches), 1)

    def test_missing_event_alternate_remains_unscoreable(self) -> None:
        baseline = [self._case(2.0, "cuda_graph")]
        optimized = [self._case(3.0, "cuda_event_fallback")]

        selected, _valid_optimized, consistent, mismatches = (
            _select_heldout_comparison_cases(
                baseline, optimized, logging.getLogger(__name__)
            )
        )

        self.assertEqual(selected[0].execution_time_ms, 2.0)
        self.assertFalse(consistent)
        self.assertEqual(len(mismatches), 1)

    def test_evaluation_refuses_to_score_candidate_event_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original_workspace = root / "original"
            output_workspace = root / "heldout"
            task_dir = root / "task"
            (original_workspace / "hip").mkdir(parents=True)
            task_dir.mkdir()
            (original_workspace / "hip" / "candidate.hip").write_text("// candidate\n")
            (original_workspace / "config.yaml").write_text(yaml.safe_dump({
                "task_type": "torch2hip",
                "target_file_path": "hip/candidate.hip",
            }))
            (original_workspace / "task_result.yaml").write_text(yaml.safe_dump({
                "task_name": "torch2hip/example",
            }))

            baseline = [self._case(2.0, "cuda_graph")]
            optimized = [self._case(3.0, "cuda_event_fallback")]

            with (
                mock.patch(
                    "src.held_out.run_heldout_eval.apply_all_injections",
                    return_value=True,
                ),
                mock.patch(
                    "src.held_out.run_heldout_eval.measure_baseline",
                    return_value=baseline,
                ),
                mock.patch(
                    "src.held_out.run_heldout_eval.evaluate_compilation",
                    return_value=(True, None),
                ),
                mock.patch(
                    "src.held_out.run_heldout_eval.evaluate_correctness",
                    return_value=(True, None),
                ),
                mock.patch(
                    "src.held_out.run_heldout_eval.measure_performance",
                    return_value=optimized,
                ),
            ):
                result = evaluate_single_task(
                    original_workspace,
                    output_workspace,
                    {"injections": []},
                    task_dir,
                    logging.getLogger(__name__),
                )

            self.assertEqual(result["orig_heldout_execution_time"], 2.0)
            self.assertEqual(result["opt_execution_time"], 3.0)
            self.assertEqual(result["speedup_ratio"], 0.0)
            self.assertFalse(result["benchmark_method_consistent"])
            self.assertEqual(len(result["benchmark_method_mismatches"]), 1)
            self.assertEqual(result["score"], 120.0)
            self.assertFalse(
                (output_workspace / "orig" / "comparison_baseline_perf.yaml")
                .exists()
            )


if __name__ == "__main__":
    unittest.main()
