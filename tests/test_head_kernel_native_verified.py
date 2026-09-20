"""CPU checks for native-verified selections and their source identities."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest

import yaml

from src.scripts.top5_head_kernels import plan_run
from src.tools.verify_head_kernels import PHASES, copy_task


ROOT = Path(__file__).resolve().parents[1]


class NativeVerifiedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.index = json.loads((ROOT / "tasks/head_kernels/native_verified.json").read_text())

    def test_configs_select_exact_index_entries_in_separate_public_images(self):
        tasks = self.index["tasks"]
        self.assertTrue(tasks)
        self.assertEqual(len(tasks), len({entry["task"] for entry in tasks}))
        selected = []
        for cohort, declaration in self.index["cohorts"].items():
            with self.subTest(cohort=cohort):
                config_path = ROOT / declaration["config"]
                config = yaml.safe_load(config_path.read_text())
                expected = sorted(entry["task"] for entry in tasks if entry["cohort"] == cohort)
                self.assertTrue(expected)
                self.assertEqual(sorted(config["tasks"]), expected)
                self.assertTrue(all((ROOT / "tasks" / task / "config.yaml").is_file()
                                    for task in expected))
                plan = plan_run(config_path, repo_root=ROOT)
                self.assertEqual(plan["tasks"], expected)
                self.assertEqual(plan["image"], declaration["image"])
                self.assertEqual(plan["expected_image_id"], declaration["expected_image_id"])
                selected.extend(expected)
        self.assertEqual(sorted(selected), sorted(entry["task"] for entry in tasks))

    def test_materialized_task_files_match_the_native_verified_digest(self):
        for entry in self.index["tasks"]:
            with self.subTest(task=entry["task"]), tempfile.TemporaryDirectory() as temporary:
                current = copy_task(ROOT / "tasks" / entry["task"], Path(temporary) / "task", ROOT)
                encoded = json.dumps(current["files"], sort_keys=True,
                                     separators=(",", ":"), ensure_ascii=True).encode("utf-8")
                expected = entry["source_identity"]
                self.assertEqual(len(current["files"]), expected["file_count"])
                self.assertEqual(current["materialized_perf_helpers"], expected["materialized_perf_helpers"])
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), expected["sha256"],
                                 "Native evidence must be refreshed for changed task files")

    def test_every_entry_records_complete_native_phases_without_broader_claims(self):
        self.assertEqual(self.index["framework_task_validator"], "NOT_RUN")
        self.assertIs(self.index["framework_PASS_claimed"], False)
        self.assertIs(self.index["exact_serving_equivalence_claimed"], False)
        self.assertEqual(self.index["required_phases"], list(PHASES))

        def report_pin(report):
            self.assertRegex(report["sha256"], r"^[a-f0-9]{64}$")
            self.assertIs(type(report["bytes"]), int)
            self.assertGreater(report["bytes"], 0)
            path = PurePosixPath(report["member"])
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)

        for entry in self.index["tasks"]:
            with self.subTest(task=entry["task"]):
                self.assertIn(entry["native_status"], {
                    "all_native_phases_succeeded", "native_prefix_reused_performance_succeeded"})
                resumed = entry["native_status"] == "native_prefix_reused_performance_succeeded"
                evidence = entry["evidence"]
                self.assertRegex(evidence["source_commit"], r"^[a-f0-9]{40}$")
                self.assertRegex(evidence["archive_sha256"], r"^[a-f0-9]{64}$")
                self.assertEqual([phase["phase"] for phase in evidence["phases"]], list(PHASES))
                report_pin(evidence["direct_report"])
                config = yaml.safe_load((ROOT / "tasks" / entry["task"] / "config.yaml").read_text())
                if resumed:
                    self.assertIn("prefix_evidence", evidence)
                    proof = evidence["prefix_evidence"]
                    self.assertEqual(proof["schema"], "aka-native-prefix-resume-v1")
                    self.assertIs(proof["performance_reused"], False)
                    self.assertEqual(proof["source_identity_sha256"], entry["source_identity"]["sha256"])
                    origin = proof["origin"]
                    self.assertTrue(origin["job_id"])
                    self.assertNotEqual(origin["job_id"], evidence["job_id"])
                    self.assertRegex(origin["source_commit"], r"^[a-f0-9]{40}$")
                    self.assertRegex(origin["archive_sha256"], r"^[a-f0-9]{64}$")
                    self.assertRegex(proof["origin_source"]["source_archive_sha256"], r"^[a-f0-9]{64}$")
                    report_pin(proof["origin_source"]["stage_receipt"])
                    report_pin(proof["origin_source"]["source_inventory"])
                    report_pin(proof["manifest"])
                    self.assertTrue(proof["manifest"]["member"].endswith("/resume-evidence/manifest.json"))
                    self.assertEqual(set(proof["snapshots"]), {"task_report", "run_report", "runtime_report"})
                    for snapshot in proof["snapshots"].values():
                        report_pin(snapshot)
                        self.assertIn("resume-evidence", PurePosixPath(snapshot["member"]).parts)
                        original = PurePosixPath(snapshot["origin_member"])
                        self.assertFalse(original.is_absolute())
                        self.assertNotIn("..", original.parts)
                else:
                    self.assertNotIn("prefix_evidence", evidence)
                for position, phase in enumerate(evidence["phases"]):
                    self.assertEqual(phase["native_status"], "ok")
                    self.assertIs(phase["timed_out"], False)
                    report_pin(phase["report"])
                    if resumed and position < 2:
                        self.assertEqual(phase["status"], "native_phase_reused")
                        self.assertIs(phase["executed_here"], False)
                        self.assertIs(phase["executed_in_this_run"], False)
                        self.assertEqual(phase["command_returncodes"], [])
                        prior = phase["prior_execution"]
                        for key in ("job_id", "source_commit", "archive_sha256"):
                            self.assertEqual(prior[key], origin[key])
                        report_pin(prior["report"])
                        original_phase = prior["phase"]
                        self.assertEqual(original_phase["phase"], phase["phase"])
                        self.assertEqual(original_phase["status"], "native_phase_succeeded")
                        self.assertEqual(original_phase["native_status"], "ok")
                        self.assertNotIn("prior_phase", original_phase)
                        self.assertIs(original_phase.get("executed_here", True), True)
                        self.assertIs(original_phase.get("executed_in_this_run", True), True)
                        commands = original_phase["commands"]
                        self.assertTrue(commands)
                        self.assertEqual([command["command"] for command in commands], config[phase["phase"] + "_command"])
                        self.assertEqual(original_phase["timeout_seconds"], config.get(phase["phase"] + "_timeout", 3600))
                        for command in commands:
                            self.assertIs(type(command["returncode"]), int)
                            self.assertEqual(command["returncode"], 0)
                            self.assertIs(command["timed_out"], False)
                        for key in ("sha256", "bytes"):
                            self.assertEqual(phase["report"][key], prior["report"][key])
                            self.assertEqual(prior["report"][key], original_phase["native_report"][key])
                        self.assertIn("resume-evidence", PurePosixPath(phase["report"]["member"]).parts)
                    else:
                        self.assertEqual(phase["status"], "native_phase_succeeded")
                        self.assertNotIn("prior_execution", phase)
                        self.assertTrue(phase["command_returncodes"])
                        self.assertTrue(all(type(code) is int and code == 0 for code in phase["command_returncodes"]))
                        if resumed:
                            self.assertEqual(phase["phase"], "performance")
                            self.assertIs(phase["executed_here"], True)
                            self.assertIs(phase["executed_in_this_run"], True)
                            self.assertGreater(phase["case_count"], 0)
                            self.assertEqual(phase["warmup_iterations"], 10)
                            self.assertEqual(phase["benchmark_iterations"], 100)

if __name__ == "__main__":
    unittest.main()
