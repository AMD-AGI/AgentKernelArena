"""JSON-only CPU fixtures for native prefix reuse; no GPU validation claimed."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

import yaml

from src.tools import native_phase_resume as resume
from src.tools import verify_head_kernels as verifier


class NativePrefixTests(unittest.TestCase):
    IMAGE = "docker.io/rocm/hyperloom@sha256:" + "b" * 64
    CONFIG = "sha256:" + "a" * 64

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        self.source = self.repo / "tasks/head_kernels/example"
        self.source.mkdir(parents=True)
        self.selector = "head_kernels/example"
        self.config = {"headkernel": {"docker": self.IMAGE, "runtime": {
            "expected_image_id": self.CONFIG, "profile": "public_test"}}}
        for phase in verifier.PHASES:
            events = "internal-correctness,performance" if phase == "performance" else phase
            code = ("import pathlib,json; p=pathlib.Path('build'); p.mkdir(exist_ok=True); "
                    f"pathlib.Path('executions.txt').write_text({events!r}); "
                    f"(p/{(phase + '_report.json')!r}).write_text(json.dumps({{'status':'ok'}}))")
            self.config[phase + "_command"] = [shlex.quote(sys.executable) + " -c " + shlex.quote(code)]
            self.config[phase + "_timeout"] = 5
        (self.source / "config.yaml").write_text(yaml.safe_dump(self.config))
        (self.source / "kernel.py").write_text("value = 1\n")
        self.enterContext(mock.patch.object(verifier, "materialize_perf_helpers_in_workspace", return_value=[]))
        self.gpu = self.enterContext(mock.patch.object(resume, "current_gpu_arch", return_value="gfx950"))
        self.bundle = self.repo / "evidence"
        self.old = self.bundle / "repo/original/worker-000/000-example"
        identity = verifier.copy_task(self.source, self.old, self.repo)
        self.identity = {"AGENT_KERNEL_ARENA_DOCKER_IMAGE": self.IMAGE,
                         "AGENT_KERNEL_ARENA_DOCKER_CONFIG_DIGEST": self.CONFIG,
                         "AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID": self.CONFIG,
                         "AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS": "[]",
                         "AGENT_KERNEL_ARENA_DOCKER_IDENTITY": "{}",
                         "AGENT_KERNEL_ARENA_GPU_ARCH": "gfx950",
                         "TVM_FFI_DISABLE_TORCH_C_DLPACK": "1",
                         "AGENT_KERNEL_ARENA_HOST_GPU_ID": "2"}
        self.task_report = {"schema": "aka-direct-task-verification-v1", **verifier.FRAMEWORK,
                            "task": self.selector, "workspace": "original/worker-000/000-example",
                            "source_identity": identity, "status": "running", "phases": []}
        for phase in resume.PREFIX:
            native = self.old / "direct-native-reports" / (phase + "_report.json")
            native.parent.mkdir(exist_ok=True)
            native.write_text(json.dumps({"status": "ok", "phase": phase}))
            self.task_report["phases"].append({
                "phase": phase, "status": "native_phase_succeeded", "native_status": "ok",
                "timeout_seconds": 5, "started_utc": "2026-09-20T18:00:00+00:00",
                "commands": [{"command": self.config[phase + "_command"][0], "returncode": 0,
                              "timed_out": False, "elapsed_seconds": 1}],
                "native_report": {"path": native.relative_to(self.old).as_posix(), **verifier.fingerprint(native)}})
        self.task_report["phases"].append({"phase": "performance", "status": "running", "commands": []})
        self.run_report = {"schema": "aka-direct-verification-v1", "status": "running", "tasks": [],
                           "plan": {"tasks": [self.selector], "image": self.IMAGE, "expected_image_id": self.CONFIG,
                                    "target_gpu_model": "MI355X", "required_environment": {"TVM_FFI_DISABLE_TORCH_C_DLPACK": "1"}},
                           "shard": {"assigned_tasks": [self.selector]},
                           "runtime_identity": self.identity}
        self.runtime_report = {"status": "ok", "phase": "complete", "native_resolution_complete": True,
                               "selected_image": self.IMAGE, "verified_config_digest": self.CONFIG,
                               "architecture": "gfx950", "profile": "public_test",
                               "environment": {"TVM_FFI_DISABLE_TORCH_C_DLPACK": "1"}}
        self.current_plan = copy.deepcopy(self.run_report["plan"])
        self.counter = 0
        self.repin()

    def repin(self):
        paths = {"task_report": (self.old.parent / "000-example.direct.json", self.task_report),
                 "run_report": (self.old.parent / "direct-verification.json", self.run_report),
                 "runtime_report": (self.old / "build/runtime_preflight.json", self.runtime_report)}
        manifest = {"schema": resume.SCHEMA, "repository_root": "repo",
                    "workspace": self.old.relative_to(self.bundle).as_posix(),
                    "origin": {"job_id": "159222"}}
        for key, (path, value) in paths.items():
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(value))
            manifest[key] = {"path": path.relative_to(self.bundle).as_posix(), **verifier.fingerprint(path)}
        self.manifest = self.bundle / "prefix.json"
        self.manifest.write_text(json.dumps(manifest))
        self.sha = verifier.fingerprint(self.manifest)["sha256"]

    def execute(self, *, sha=None, identity=None):
        self.counter += 1
        workspace = self.repo / f"fresh-{self.counter}/000-example"
        workspace.parent.mkdir()
        result = verifier.verify_task(self.source, workspace, self.repo,
                                      resume_request={"path": self.manifest, "sha256": sha or self.sha,
                                                      "plan": self.current_plan},
                                      runtime_identity=identity or self.identity)
        return result, workspace

    def assert_rejected(self, fragment):
        result, workspace = self.execute()
        self.assertEqual(result["status"], "verification_error")
        self.assertIn(fragment, result["error"])
        self.assertFalse((workspace / "executions.txt").exists())

    def test_good_prefix_runs_only_unchanged_full_performance_command(self):
        current = {**self.identity, "AGENT_KERNEL_ARENA_HOST_GPU_ID": "7"}
        result, workspace = self.execute(identity=current)
        self.assertEqual(result["status"], "native_prefix_reused_performance_succeeded", result)
        self.assertEqual((workspace / "executions.txt").read_text(), "internal-correctness,performance")
        self.assertEqual([p["status"] for p in result["phases"]],
                         ["native_phase_reused", "native_phase_reused", "native_phase_succeeded"])
        self.assertFalse(result["phases"][0]["executed_here"])
        self.assertEqual(result["phases"][0]["prior_phase"]["started_utc"], "2026-09-20T18:00:00+00:00")
        self.assertEqual(result["resume_provenance"]["origin"]["job_id"], "159222")
        self.assertFalse((workspace / "build/correctness_report.json").exists())
        self.assertEqual(result["phases"][2]["commands"][0]["command"], self.config["performance_command"][0])

    def test_wrong_manifest_hash_is_rejected(self):
        result, workspace = self.execute(sha="0" * 64)
        self.assertEqual(result["status"], "verification_error")
        self.assertFalse((workspace / "executions.txt").exists())

    def test_tampered_direct_or_native_report_is_rejected(self):
        (self.old.parent / "000-example.direct.json").write_text("{}")
        self.assert_rejected("hash/size mismatch")
        self.repin()
        (self.old / "direct-native-reports/correctness_report.json").write_text('{"status":"fail"}')
        self.assert_rejected("hash/size mismatch")

    def test_missing_report_is_rejected(self):
        (self.old / "direct-native-reports/compile_report.json").unlink()
        self.assert_rejected("No such file")

    def test_changed_materialized_source_is_rejected(self):
        (self.source / "kernel.py").write_text("value = 2\n")
        self.assert_rejected("source_identity differs")

    def test_failed_partial_or_noninteger_success_is_rejected(self):
        original = copy.deepcopy(self.task_report)
        for change in ("failed", "partial", "boolean", "timeout", "command"):
            self.task_report = copy.deepcopy(original)
            phase = self.task_report["phases"][1]
            if change == "failed": phase["status"] = "command_failed"
            if change == "partial": phase["commands"] = []
            if change == "boolean": phase["commands"][0]["returncode"] = False
            if change == "timeout": phase["commands"][0]["timed_out"] = True
            if change == "command": phase["commands"][0]["command"] = "untrusted prior command"
            self.repin()
            with self.subTest(change=change): self.assert_rejected("failed, partial, reused")

    def test_runtime_image_or_architecture_mismatch_is_rejected(self):
        self.runtime_report["architecture"] = "gfx942"
        self.repin()
        self.assert_rejected("gfx950 preflight differs")
        self.runtime_report["architecture"] = "gfx950"
        self.run_report["plan"]["image"] = "example.invalid/other@sha256:" + "c" * 64
        self.repin()
        self.assert_rejected("runtime image")

    def test_wrong_current_gpu_is_rejected(self):
        self.gpu.return_value = "gfx942"
        self.assert_rejected("current physical GPU")

    def test_profile_environment_worker_and_full_workspace_must_match(self):
        original_task = copy.deepcopy(self.task_report)
        original_run = copy.deepcopy(self.run_report)
        original_runtime = copy.deepcopy(self.runtime_report)
        for change in ("profile", "environment", "target", "shard", "workspace", "reused_marker"):
            self.task_report = copy.deepcopy(original_task)
            self.run_report = copy.deepcopy(original_run)
            self.runtime_report = copy.deepcopy(original_runtime)
            if change == "profile": self.runtime_report["profile"] = "other_profile"
            if change == "environment": self.runtime_report["environment"]["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "0"
            if change == "target": self.run_report["plan"]["target_gpu_model"] = "MI300"
            if change == "shard": self.run_report["shard"]["assigned_tasks"] = ["other"]
            if change == "workspace": self.task_report["workspace"] = "wrong/worker-000/000-example"
            if change == "reused_marker": self.task_report["phases"][0]["prior_phase"] = {}
            self.repin()
            with self.subTest(change=change):
                result, workspace = self.execute()
                self.assertEqual(result["status"], "verification_error")
                self.assertFalse((workspace / "executions.txt").exists())

    def test_prior_performance_is_never_reused(self):
        self.task_report["phases"][2].update(status="native_phase_succeeded", native_status="ok")
        self.repin()
        (self.old / "direct-native-reports/performance_report.json").write_bytes(b"not even JSON")
        result, workspace = self.execute()
        self.assertEqual(result["status"], "native_prefix_reused_performance_succeeded")
        self.assertEqual((workspace / "executions.txt").read_text(), "internal-correctness,performance")
        self.assertFalse((workspace / "resume-evidence/performance_report.json").exists())

    def test_pickle_inputs_and_escaping_paths_are_rejected(self):
        (self.source / "oracle.pt").write_bytes(b"not deserialized")
        self.assert_rejected(".pt payloads are not supported")
        (self.source / "oracle.pt").unlink()
        manifest = json.loads(self.manifest.read_text())
        manifest["task_report"]["path"] = "../outside.json"
        self.manifest.write_text(json.dumps(manifest))
        self.sha = verifier.fingerprint(self.manifest)["sha256"]
        self.assert_rejected("escapes")


if __name__ == "__main__":
    unittest.main()
