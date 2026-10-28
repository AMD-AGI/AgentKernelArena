import logging
import tempfile
import unittest
from pathlib import Path

import yaml

from src.preprocessing import _resolve_gfx_arch
from src.prompt_builder import _load_cheatsheet
from src.tasks import get_task_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GPUArchitectureConfigTests(unittest.TestCase):
    def test_all_mi300_models_resolve_to_gfx942(self) -> None:
        for model in ("MI300", "MI300X", "MI325", "MI325X", "MI300A", "mi300x", "Mi325X", "mi300a"):
            with self.subTest(model=model):
                self.assertEqual(_resolve_gfx_arch(model), "gfx942")

    def test_exact_mi300_models_load_shared_and_model_specific_context(self) -> None:
        expected_headings = {
            "MI300X": "# AMD Instinct MI300X Model Profile",
            "MI325X": "# AMD Instinct MI325X Model Profile",
            "MI300A": "# AMD Instinct MI300A Model Profile",
        }

        for model, profile_heading in expected_headings.items():
            with self.subTest(model=model):
                prompt, gfx_arch = _load_cheatsheet(
                    "hip2hip", model, PROJECT_ROOT, {}, logging.getLogger(__name__)
                )

                self.assertEqual(gfx_arch, "gfx942")
                self.assertIn("# AMD CDNA 3 (`gfx942`) Kernel Optimization Context", prompt)
                self.assertIn(profile_heading, prompt)
                self.assertLess(
                    prompt.index("# AMD CDNA 3 (`gfx942`) Kernel Optimization Context"),
                    prompt.index(profile_heading),
                )

    def test_generic_mi300_target_loads_non_sku_specific_profile(self) -> None:
        prompt, gfx_arch = _load_cheatsheet(
            "triton2triton", "MI300", PROJECT_ROOT, {}, logging.getLogger(__name__)
        )

        self.assertEqual(gfx_arch, "gfx942")
        self.assertIn("# Generic AMD Instinct MI300-Series Profile", prompt)
        self.assertNotIn("# AMD Instinct MI300X Model Profile", prompt)

    def test_mi325_alias_keeps_the_mi325x_profile(self) -> None:
        logger = logging.getLogger(__name__)
        self.assertEqual(
            _load_cheatsheet("hip2hip", "MI325", PROJECT_ROOT, {}, logger),
            _load_cheatsheet("hip2hip", "MI325X", PROJECT_ROOT, {}, logger),
        )

    def test_v2_tasks_keep_destination_language_knowledge(self) -> None:
        for model in ("MI300X", "MI325X", "MI300A"):
            with self.subTest(model=model):
                prompt, arch = _load_cheatsheet(
                    "Aiter-task", model, PROJECT_ROOT,
                    {"schema_version": 2, "candidate": {"language": "flydsl"}},
                    logging.getLogger(__name__),
                )
                self.assertEqual(arch, "gfx942")
                self.assertIn(f"# AMD Instinct {model} Model Profile", prompt)
                self.assertIn("# FlyDSL Kernel Best Practices", prompt)

    def test_model_quickstarts_select_retained_tasks(self) -> None:
        for model in ("MI300X", "MI325X", "MI300A"):
            with self.subTest(model=model):
                path = PROJECT_ROOT / f"example_configs/quickstart_claude_{model.lower()}.yaml"
                config = yaml.safe_load(path.read_text())
                self.assertEqual(config["target_gpu_model"], model)
                self.assertEqual(config["tasks"], ["hip2hip/GELU"])
                for task in config["tasks"]:
                    self.assertTrue(get_task_config(str(PROJECT_ROOT / "tasks"), task))

    def test_legacy_file_and_composed_files_are_both_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            config_dir = project_root / "src" / "prompts" / "cheatsheet"
            config_dir.mkdir(parents=True)
            (project_root / "legacy.md").write_text("legacy architecture")
            (project_root / "shared.md").write_text("shared architecture")
            (project_root / "profile.md").write_text("model profile")
            (project_root / "hip.md").write_text("hip knowledge")
            config = {
                "architecture": {
                    "LEGACY": {"gfx_arch": "gfx900", "file": "legacy.md"},
                    "COMPOSED": {
                        "gfx_arch": "gfx942",
                        "files": ["shared.md", "profile.md"],
                    },
                },
                "knowledge": {"hip": "hip.md"},
            }
            (config_dir / "default_cheatsheet.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )

            legacy_prompt, legacy_arch = _load_cheatsheet(
                "hip2hip", "LEGACY", project_root, {}, logging.getLogger(__name__)
            )
            composed_prompt, composed_arch = _load_cheatsheet(
                "hip2hip", "COMPOSED", project_root, {}, logging.getLogger(__name__)
            )

            self.assertEqual(legacy_arch, "gfx900")
            self.assertEqual(legacy_prompt, "legacy architecture\n\n---\n\nhip knowledge")
            self.assertEqual(composed_arch, "gfx942")
            self.assertEqual(
                composed_prompt,
                "shared architecture\n\n---\n\nmodel profile\n\n---\n\nhip knowledge",
            )


if __name__ == "__main__":
    unittest.main()
