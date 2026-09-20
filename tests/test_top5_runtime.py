"""CPU fixtures for head-kernel runtime selection and fail-closed preflight."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import contextlib
from collections import Counter
import io
import json
import os
import tempfile
import unittest
from unittest import mock
import yaml


from head_kernel_test_utils import task_directory

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_module("top5_launcher", "src/scripts/top5_head_kernels.py")
runtime = load_module("head_runtime", "tasks/head_kernels/_support/runtime_preflight.py")


def fixture_config(tmp_path, images, *, gpu="MI355X"):
    selectors = []
    for index, image in enumerate(images):
        selector = f"head_kernels/task_{index}"
        directory = tmp_path / "tasks" / selector
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text(yaml.safe_dump({"headkernel": {"docker": image}}))
        selectors.append(selector)
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump({"target_gpu_model": gpu, "tasks": selectors}))
    return config


class CohortTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_examples_cover_each_task_once_per_agent(self):
        expected = {p.parent.relative_to(ROOT / "tasks").as_posix()
                    for p in (ROOT / "tasks/head_kernels").rglob("config.yaml")}
        expected_counts = Counter(yaml.safe_load(path.read_text())["headkernel"]["docker"]
                                  for path in (ROOT / "tasks/head_kernels").rglob("config.yaml"))
        for agent in ("validator", "claude"):
            plans = [launcher.plan_run(path) for path in
                     sorted((ROOT / "example_configs").glob(f"top5_{agent}_*_mi355x.yaml"))]
            self.assertEqual(len(plans), 3)
            selected = [task for plan in plans for task in plan["tasks"]]
            self.assertEqual(len(selected), len(set(selected)))
            self.assertEqual(set(selected), expected)
            by_image = {plan["image"]: plan["task_count"] for plan in plans}
            self.assertEqual(by_image, dict(expected_counts))

    def test_rejects_mixed_capture_images(self):
        config = fixture_config(self.tmp_path, ["registry/sglang:v0.5.17", "registry/sglang:v0.5.18"])
        with self.assertRaisesRegex(ValueError, "Mixed capture runtimes"):
            launcher.plan_run(config, self.tmp_path)

    def test_rejects_unversioned_or_invalid_image(self):
        for image in ["registry/sglang:latest", "registry/sglang", "registry/sglang:v 1"]:
            with self.subTest(image=image), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = fixture_config(root, [image])
                with self.assertRaisesRegex(ValueError, "versioned image"):
                    launcher.plan_run(config, root)

    def test_rejects_wrong_gpu(self):
        config = fixture_config(self.tmp_path, ["registry/sglang:v0.5.18"], gpu="MI300")
        with self.assertRaisesRegex(ValueError, "MI355X"):
            launcher.plan_run(config, self.tmp_path)

    def test_rejects_external_task_selector(self):
        config = fixture_config(self.tmp_path, ["registry/sglang:v0.5.18"])
        config.write_text(yaml.safe_dump({"target_gpu_model": "MI355X", "tasks": ["../external"]}))
        with self.assertRaisesRegex(ValueError, "below tasks/head_kernels"):
            launcher.plan_run(config, self.tmp_path)

    def test_rejects_conflicting_override(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            launcher.runtime_environment({"image": "capture:v0.5.18"},
                                         {"AKA_DOCKER_IMAGE": "default:v0.5.14"})

    def test_launcher_uses_existing_docker_runner(self):
        config = ROOT / "example_configs/top5_validator_sglang_v0518_mi355x.yaml"
        with mock.patch.dict(os.environ, {"AKA_DOCKER_IMAGE": ""}), \
                mock.patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=17)) as run:
            result = launcher.main(["run", "--config", str(config), "--", "--run-suffix", "fixture"])
        self.assertEqual(result, 17)
        command, = run.call_args.args
        kwargs = run.call_args.kwargs
        self.assertEqual(command[:3], ["bash", "src/scripts/docker_benchmark.sh", "run"])
        self.assertEqual(command[-2:], ["--run-suffix", "fixture"])
        self.assertEqual(kwargs["env"]["AKA_DOCKER_IMAGE"], launcher.plan_run(config)["image"])
        self.assertEqual(kwargs["env"]["AKA_VERIFY_RUNTIME_IMAGE"], "1")
        self.assertEqual(kwargs["env"]["TVM_FFI_DISABLE_TORCH_C_DLPACK"], "1")
        self.assertEqual(kwargs["env"]["AKA_EXPECTED_IMAGE_ID"],
                         "sha256:760dd38b9b6f2bd11c13011d470eb8e377c3f0d71284a090a710d64a23bd789f")
        self.assertEqual(kwargs["cwd"], ROOT)

    def test_plan_never_launches_process(self):
        config = ROOT / "example_configs/top5_validator_kimi_k3_mi355x.yaml"
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"AKA_DOCKER_IMAGE": ""}), \
                mock.patch.object(launcher.subprocess, "run", side_effect=AssertionError("unexpected process")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(launcher.main(["plan", "--config", str(config)]), 0)
        self.assertEqual(json.loads(output.getvalue())["task_count"], 3)

    def test_every_task_has_self_contained_runtime_preflight(self):
        canonical = (ROOT / "tasks/head_kernels/_support/runtime_preflight.py").read_bytes()
        for config in (ROOT / "tasks/head_kernels").rglob("config.yaml"):
            self.assertEqual((config.parent / "scripts/runtime_preflight.py").read_bytes(), canonical)

    def test_environment_matrix_matches_task_runtime_contracts(self):
        self.assertEqual((ROOT / "docs/reference/top5-head-kernel-environments.md").read_text(),
                         launcher.environment_matrix())

    def test_plan_reads_capture_image_id_evidence(self):
        image = "registry/sglang:v0.5.18"
        expected_id = "sha256:" + "b" * 64
        config = fixture_config(self.tmp_path, [image])
        ut_dir = self.tmp_path / "tasks/head_kernels/task_0/ut"
        ut_dir.mkdir()
        (ut_dir / "meta.json").write_text(json.dumps({"source_provenance": {
            "runtime_image": image, "runtime_image_id": expected_id}}))
        self.assertEqual(launcher.plan_run(config, self.tmp_path)["expected_image_id"], expected_id)

    def test_rejects_capture_evidence_from_different_image(self):
        config = fixture_config(self.tmp_path, ["registry/sglang:v0.5.18"])
        ut_dir = self.tmp_path / "tasks/head_kernels/task_0/ut"
        ut_dir.mkdir()
        (ut_dir / "meta.json").write_text(json.dumps({"source_provenance": {
            "runtime_image": "registry/sglang:v0.5.17", "runtime_image_id": "sha256:" + "b" * 64}}))
        with self.assertRaisesRegex(ValueError, "evidence disagrees"):
            launcher.plan_run(config, self.tmp_path)

    def test_capture_dispatch_backends_contribute_runtime_requirements(self):
        task_dir = task_directory("qwen3.8-2.4t__dense_bf16_gemm_cluster")
        config = yaml.safe_load((task_dir / "config.yaml").read_text())
        requirements = runtime.runtime_requirements(config, task_dir)
        self.assertIn("flydsl", requirements["required_modules"])
        self.assertEqual(requirements["source_commits"]["aiter"],
                         "d9e5ef7ce08ee7045d583aed768cff41aa9210fe")


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.task_dir = Path(directory.name)
        image = "registry/sglang:v0.5.18-rocm720-mi35x-profilerfix"
        self.config = {"headkernel": {"docker": image, "target_callable": "fixture.kernel:run"}}
        self.modules = {name: SimpleNamespace(__version__="1.0") for name in ("sglang", "triton", "aiter")}
        self.modules["sglang"].__version__ = "0.5.18"
        self.properties = SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        self.modules["torch"] = SimpleNamespace(
            __version__="2.9.1+rocm7.2.0", version=SimpleNamespace(hip="7.2.26015"),
            cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                                 get_device_properties=lambda index: self.properties))
        self.modules["fixture.kernel"] = SimpleNamespace(run=lambda: None)
        self.enterContext(mock.patch.dict(os.environ, {
            "AGENT_KERNEL_ARENA_DOCKER": "1", "AGENT_KERNEL_ARENA_DOCKER_IMAGE": image,
            "AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID": "sha256:" + "a" * 64,
            "TVM_FFI_DISABLE_TORCH_C_DLPACK": "1",
            "AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS": "[]"}))
        self.enterContext(mock.patch.object(runtime.importlib, "import_module", side_effect=lambda name: self.modules[name]))

    def assert_runtime_rejected(self, message):
        report = runtime.preflight(self.config)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any(message in error for error in report["errors"]), report)
        with self.assertRaisesRegex(RuntimeError, "Runtime preflight failed"):
            runtime.require_runtime(self.config, self.task_dir)
        persisted = json.loads((self.task_dir / "build/runtime_preflight.json").read_text())
        self.assertEqual(persisted["status"], "fail")

    def test_preflight_accepts_matching_runtime(self):
        report = runtime.preflight(self.config)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["architecture"], "gfx950")
        self.assertEqual(report["versions"]["hip"], "7.2.26015")
        self.assertEqual(report["selected_image_id"], "sha256:" + "a" * 64)
        self.assertEqual(report["registry_repo_digests"], [])

    def test_preflight_image_mismatch_precedes_imports(self):
        with mock.patch.dict(os.environ, {"AGENT_KERNEL_ARENA_DOCKER_IMAGE": "default:v0.5.14"}), \
                mock.patch.object(runtime.importlib, "import_module", side_effect=AssertionError("unexpected import")):
            self.assert_runtime_rejected("Capture image mismatch")

    def test_rejects_wrong_sglang(self):
        self.modules["sglang"].__version__ = "0.5.17"
        self.assert_runtime_rejected("SGLang version mismatch")

    def test_rejects_wrong_hip(self):
        self.modules["torch"].version.hip = "7.1.0"
        self.assert_runtime_rejected("ROCm/HIP 7.2")

    def test_rejects_wrong_architecture(self):
        self.properties.gcnArchName = "gfx942"
        self.assert_runtime_rejected("GPU architecture mismatch")

    def test_rejects_missing_gpu(self):
        self.modules["torch"].cuda.is_available = lambda: False
        self.assert_runtime_rejected("GPU is not available")

    def test_rejects_missing_target(self):
        self.modules["fixture.kernel"].run = None
        self.assert_runtime_rejected("target is not callable")

    def test_rejects_missing_dependency(self):
        self.config["headkernel"]["backend"] = "TileLang"
        self.assert_runtime_rejected("tilelang")

    def test_rejects_missing_glm_architecture(self):
        self.config["headkernel"]["runtime"] = {"required_model_types": ["glm5_next"]}
        self.modules["transformers.models.auto.configuration_auto"] = SimpleNamespace(CONFIG_MAPPING={})
        self.assert_runtime_rejected("Required model architecture")

    def test_rejects_known_package_version_mismatch(self):
        self.config["headkernel"]["runtime"] = {"package_versions": {"triton": "3.7.1"}}
        self.assert_runtime_rejected("triton version mismatch")

    def test_rejects_missing_host_image_attestation(self):
        with mock.patch.dict(os.environ, {"AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID": ""}):
            self.assert_runtime_rejected("did not supply a verified image ID")

    def test_v0518_requires_documented_tvm_ffi_workaround(self):
        with mock.patch.dict(os.environ, {"TVM_FFI_DISABLE_TORCH_C_DLPACK": "0"}):
            self.assert_runtime_rejected("TVM_FFI_DISABLE_TORCH_C_DLPACK=1")

    def test_tvm_ffi_workaround_is_scoped_to_v0518(self):
        config = {"headkernel": {"docker": "registry/sglang:v0.5.17-rocm720-mi35x-profilerfix"}}
        self.assertEqual(runtime.runtime_requirements(config, self.task_dir)["environment"], {})

    def test_rejects_capture_image_id_mismatch(self):
        self.config["headkernel"]["runtime"] = {"expected_image_id": "sha256:" + "b" * 64}
        self.assert_runtime_rejected("Capture image ID mismatch")

    def test_records_registry_digests_separately_from_image_id(self):
        digests = ["registry/sglang@sha256:" + "c" * 64]
        with mock.patch.dict(os.environ, {"AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS": json.dumps(digests)}):
            report = runtime.preflight(self.config)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["registry_repo_digests"], digests)
        self.assertEqual(report["selected_image_id"], "sha256:" + "a" * 64)

    def test_persists_verified_identity_with_observed_runtime(self):
        report = runtime.require_runtime(self.config, self.task_dir)
        persisted = json.loads((self.task_dir / "build/runtime_preflight.json").read_text())
        self.assertEqual(report, persisted)
        self.assertEqual(persisted["status"], "ok")
        self.assertEqual(persisted["selected_image_id"], "sha256:" + "a" * 64)


if __name__ == "__main__":
    unittest.main()
