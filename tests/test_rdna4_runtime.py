"""CPU regression tests for the RDNA4 image layout and Docker smoke policy."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "rdna4_prepare_runtime", ROOT / "docker/rdna4/prepare_runtime.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class RuntimeLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        self.private = root / "root"
        self.private.mkdir(mode=0o700)
        self.base = self.private / "uv/python"
        (self.base / "bin").mkdir(parents=True)
        (self.base / "bin/python3.14").write_text("interpreter bytes")
        (self.base / "lib").mkdir()
        (self.base / "lib/stdlib").write_text("standard library bytes")
        self.prefix = root / "opt"
        self.venv = self.prefix / "python"
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "bin/python").symlink_to(self.base / "bin/python3.14")
        (self.venv / "bin/python3").symlink_to("python")
        (self.venv / "pyvenv.cfg").write_text(
            f"home = {self.base / 'bin'}\ninclude-system-site-packages = false\n"
        )
        self.sdk = self.venv / "lib/site-packages/_rocm_sdk_devel"
        (self.sdk / "bin").mkdir(parents=True)
        (self.sdk / "bin/hipcc").write_text("compiler bytes")

    def prepare(self):
        runtime.prepare_runtime(self.venv, self.base, self.sdk, self.prefix)

    def test_relocates_interpreter_and_stdlib_without_exposing_root(self):
        self.prepare()
        destination = self.prefix / "aka-python-runtime"
        for name in ("python", "python3"):
            link = self.venv / "bin" / name
            self.assertEqual(link.resolve(), destination / "bin/python3.14")
            self.assertEqual(link.read_text(), "interpreter bytes")
        self.assertEqual((destination / "lib/stdlib").read_text(), "standard library bytes")
        self.assertEqual(self.private.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.base / "bin/python3.14").read_text(), "interpreter bytes")
        self.assertEqual((self.venv / "pyvenv.cfg").read_text(),
                         f"home = {destination / 'bin'}\ninclude-system-site-packages = false\n")
        self.assertEqual((self.prefix / "venv").resolve(), self.venv)
        self.assertEqual((self.prefix / "rocm/bin/hipcc").read_text(), "compiler bytes")

    def test_existing_runtime_paths_are_not_replaced(self):
        (self.prefix / "venv").symlink_to("missing-target")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(os.readlink(self.prefix / "venv"), "missing-target")
        self.assertFalse((self.prefix / "aka-python-runtime").exists())

    def test_missing_sdk_fails_before_relocating(self):
        (self.sdk / "bin/hipcc").unlink()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.prefix / "aka-python-runtime").exists())
        self.assertEqual((self.venv / "bin/python").resolve(), self.base / "bin/python3.14")

    def test_unexpected_interpreter_location_is_rejected(self):
        (self.venv / "bin/python3").unlink()
        (self.venv / "bin/python3").symlink_to(sys.executable)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.prefix / "aka-python-runtime").exists())


class SmokeProfilerTests(unittest.TestCase):
    def run_smoke(self, selected, commands, actual=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / "python").symlink_to(sys.executable)
            for command in ("dirname", "id"):
                (bin_dir / command).symlink_to(shutil.which(command))
            for command in commands:
                path = bin_dir / command
                path.write_text("#!/bin/sh\nexit 0\n")
                path.chmod(0o755)
            for name in ("triton", "pytest", "yaml", "numpy", "flydsl"):
                (root / f"{name}.py").write_text("__version__ = 'mock'\n")
            (root / "torch.py").write_text(
                "from types import SimpleNamespace\nimport os\n"
                "cuda = SimpleNamespace(is_available=lambda: True, "
                "get_device_name=lambda i: 'mock GPU', "
                "get_device_properties=lambda i: SimpleNamespace("
                "gcnArchName=os.environ['MOCK_GPU_ARCH']))\n"
            )
            env = {**os.environ, "PATH": str(bin_dir), "PYTHONPATH": str(root),
                   "HOME": str(root), "AGENT_KERNEL_ARENA_GPU_ARCH": selected,
                   "MOCK_GPU_ARCH": actual or selected}
            return subprocess.run(
                [shutil.which("bash"), str(ROOT / "src/scripts/docker_benchmark.sh"),
                 "_container_smoke"], env=env, capture_output=True, text=True, timeout=30,
            )

    def test_rdna4_accepts_rocprofv3_without_rocprof_compute(self):
        result = self.run_smoke("gfx1201", ("hipcc", "rocprofv3"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rocprofv3=", result.stdout)

    def test_rdna4_rejects_missing_rocprofv3(self):
        result = self.run_smoke("gfx1201", ("hipcc", "rocprof-compute"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing command: rocprofv3", result.stderr)

    def test_cdna_still_requires_rocprof_compute(self):
        for arch in ("gfx942", "gfx950"):
            with self.subTest(arch=arch):
                result = self.run_smoke(arch, ("hipcc", "rocprofv3"))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("missing command: rocprof-compute", result.stderr)
                result = self.run_smoke(arch, ("hipcc", "rocprof-compute"))
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_rdna4_still_requires_compiler(self):
        result = self.run_smoke("gfx1201", ("rocprofv3",))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing command: hipcc", result.stderr)

    def test_rdna4_still_rejects_architecture_mismatch(self):
        result = self.run_smoke("gfx1201", ("hipcc", "rocprofv3"), actual="gfx950")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match visible device arch gfx950", result.stderr)


if __name__ == "__main__":
    unittest.main()
