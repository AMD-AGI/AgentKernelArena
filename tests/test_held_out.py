import tempfile
import unittest
from pathlib import Path

import yaml

from src.held_out.generate_heldout import discover_tasks
from src.held_out.injection import (
    apply_injection,
    replace_get_inputs,
    replace_test_shapes,
)
from src.held_out.run_heldout_eval import resolve_task_id


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


if __name__ == "__main__":
    unittest.main()
