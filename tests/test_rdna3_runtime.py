"""Radeon PRO W7900 (gfx1100) runtime recipe and smoke identity checks."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
RDNA3 = ROOT / "docker/rdna3"
RDNA4 = ROOT / "docker/rdna4"
NAVI31_ROCMINFO = ROOT / "tests/fixtures/rocminfo_navi31_4x_w7900_dual_slot.txt"


class RecipeTests(unittest.TestCase):
    def test_shared_recipe_inputs_are_byte_identical_copies(self):
        # Diverging copies would silently give W7900 a different ROCm or Python stack.
        for name in ("prepare_runtime.py", "requirements.lock", ".dockerignore"):
            with self.subTest(name=name):
                self.assertEqual((RDNA3 / name).read_bytes(), (RDNA4 / name).read_bytes())

    def test_base_image_is_the_same_digest_pin(self):
        def base(recipe):
            return re.findall(r"(?m)^FROM\s+(\S+)", (recipe / "Dockerfile").read_text())

        self.assertEqual(base(RDNA3), base(RDNA4))
        self.assertRegex(base(RDNA3)[0], r"@sha256:[0-9a-f]{64}$")

    def test_image_ships_rocminfo_for_the_identity_check(self):
        self.assertIn("rocminfo missing", (RDNA3 / "Dockerfile").read_text())


class SmokeIdentityTests(unittest.TestCase):
    def run_smoke(self, rocminfo_text=None, config_text=None, selected="gfx1100"):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / "python").symlink_to(sys.executable)
            for command in ("dirname", "id", "cat"):
                (bin_dir / command).symlink_to(shutil.which(command))
            (bin_dir / "hipcc").write_text("#!/bin/sh\nexit 0\n")
            (bin_dir / "hipcc").chmod(0o755)
            if rocminfo_text is not None:
                (root / "rocminfo.txt").write_text(rocminfo_text)
                (bin_dir / "rocminfo").write_text(f"#!/bin/sh\ncat '{root / 'rocminfo.txt'}'\n")
                (bin_dir / "rocminfo").chmod(0o755)
            # PyYAML stays real: the identity check reads the cheatsheet registry.
            # The bare python symlink loses any venv, so expose the package directly.
            (root / "yaml").symlink_to(Path(yaml.__file__).parent)
            for name in ("triton", "pytest", "numpy", "flydsl"):
                (root / f"{name}.py").write_text("__version__ = 'mock'\n")
            (root / "torch.py").write_text(
                "from types import SimpleNamespace\nimport os\n"
                "float32 = 'float32'\n"
                "class Tensor:\n"
                "    def __init__(self, values, device='cpu'):\n"
                "        self.values, self.device = list(values), device\n"
                "    def to(self, device):\n"
                "        return Tensor(self.values, device)\n"
                "    def cpu(self):\n"
                "        return self.to('cpu')\n"
                "    def __add__(self, scalar):\n"
                "        return Tensor([x + scalar for x in self.values], self.device)\n"
                "def arange(n, *, dtype):\n"
                "    return Tensor(range(n))\n"
                "def equal(left, right):\n"
                "    return left.values == right.values\n"
                "cuda = SimpleNamespace(is_available=lambda: True, "
                "get_device_name=lambda i: 'mock GPU', synchronize=lambda: None, "
                "get_device_properties=lambda i: SimpleNamespace(gcnArchName=os.environ['MOCK_GPU_ARCH']))\n"
            )
            env = {**os.environ, "PATH": str(bin_dir), "PYTHONPATH": str(root),
                   "HOME": str(root), "AGENT_KERNEL_ARENA_GPU_ARCH": selected,
                   "AKA_REQUIRED_PROFILERS": "", "MOCK_GPU_ARCH": selected}
            env.pop("AKA_SMOKE_CONFIG", None)
            if config_text is not None:
                (root / "run.yaml").write_text(config_text)
                env["AKA_SMOKE_CONFIG"] = str(root / "run.yaml")
            return subprocess.run(
                [shutil.which("bash"), str(ROOT / "src/scripts/docker_benchmark.sh"), "_container_smoke"],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
            )

    def test_w7900_passes_and_reports_name_and_pci(self):
        result = self.run_smoke(NAVI31_ROCMINFO.read_text())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("gpu_marketing_name=AMD Radeon Pro W7900 Dual Slot;", result.stdout)
        self.assertIn("gpu_pci_device_id=1002:744a;", result.stdout)
        self.assertIn("gpu_identity=PASS target_gpu_model=W7900", result.stdout)
        self.assertIn("torch_gpu_numerical_smoke=PASS", result.stdout)

    def test_other_gfx1100_board_is_refused_before_numerical_smoke(self):
        for name in ("AMD Radeon Pro W7800", "AMD Radeon RX 7900 XTX", ""):
            with self.subTest(name=name):
                text = NAVI31_ROCMINFO.read_text().replace(
                    "AMD Radeon Pro W7900 Dual Slot", name)
                result = self.run_smoke(text)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("GPU identity refused", result.stderr)
                self.assertIn(f"'{name or '<empty>'}' gfx1100 PCI 1002:744a", result.stderr)
                self.assertNotIn("torch_gpu_numerical_smoke=PASS", result.stdout)

    def test_missing_rocminfo_is_refused_on_gfx1100(self):
        result = self.run_smoke(None)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rocminfo was not found", result.stderr)

    def test_preflight_config_with_evaluation_tools_is_refused(self):
        config = "target_gpu_model: W7900\nevaluation_tools:\n  enabled: [gpu_asan]\n"
        result = self.run_smoke(NAVI31_ROCMINFO.read_text(), config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("verified only for gfx950", result.stderr)

    def test_preflight_rdna4_config_on_a_w7900_is_refused(self):
        result = self.run_smoke(NAVI31_ROCMINFO.read_text(), "target_gpu_model: RDNA4\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Use target_gpu_model W7900", result.stderr)


if __name__ == "__main__":
    unittest.main()
