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
        for entry in self.index["tasks"]:
            with self.subTest(task=entry["task"]):
                self.assertEqual(entry["native_status"], "all_native_phases_succeeded")
                evidence = entry["evidence"]
                self.assertRegex(evidence["source_commit"], r"^[a-f0-9]{40}$")
                self.assertRegex(evidence["archive_sha256"], r"^[a-f0-9]{64}$")
                self.assertEqual([phase["phase"] for phase in evidence["phases"]], list(PHASES))
                reports = [evidence["direct_report"]]
                for phase in evidence["phases"]:
                    self.assertEqual(phase["status"], "native_phase_succeeded")
                    self.assertEqual(phase["native_status"], "ok")
                    self.assertTrue(phase["command_returncodes"])
                    self.assertEqual(set(phase["command_returncodes"]), {0})
                    self.assertIs(phase["timed_out"], False)
                    reports.append(phase["report"])
                for report in reports:
                    self.assertRegex(report["sha256"], r"^[a-f0-9]{64}$")
                    self.assertGreater(report["bytes"], 0)
                    path = PurePosixPath(report["member"])
                    self.assertFalse(path.is_absolute())
                    self.assertNotIn("..", path.parts)


if __name__ == "__main__":
    unittest.main()
