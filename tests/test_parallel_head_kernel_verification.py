"""CPU shard/aggregation tests; no GPU execution or framework PASS is claimed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.tools import verify_head_kernels as verifier


class ParallelVerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = {"tasks": [f"head_kernels/task-{index:02d}" for index in range(11)],
                     "image": "example.invalid/runtime:fixed", "expected_image_id": "sha256:" + "a" * 64}

    def declaration(self, gpu_ids="0,1,2,3,4,5,6,7"):
        with mock.patch.object(verifier, "plan_run", return_value=self.plan):
            return verifier.parallel_plan(self.root / "config.yaml", gpu_ids, self.root)

    def reports(self, declaration):
        (self.root / "parallel-plan.json").write_text(json.dumps(declaration))
        for worker in declaration["workers"]:
            path = self.root / worker["directory"]
            path.mkdir()
            report = {"status": "all_native_phases_succeeded", "plan": self.plan,
                      "shard": {"index": worker["index"], "count": len(declaration["workers"]),
                                "assigned_tasks": worker["tasks"], "host_gpu_id": worker["gpu_id"],
                                "taskset_sha256": declaration["taskset_sha256"]},
                      "tasks": [{"task": task, "status": "all_native_phases_succeeded"}
                                for task in worker["tasks"]]}
            (path / "direct-verification.json").write_text(json.dumps(report))

    def test_eleven_tasks_are_disjoint_across_eight_workers(self):
        declaration = self.declaration()
        shards = [worker["tasks"] for worker in declaration["workers"]]
        self.assertEqual([len(shard) for shard in shards], [2, 2, 2, 1, 1, 1, 1, 1])
        combined = [task for shard in shards for task in shard]
        self.assertEqual(len(combined), len(set(combined)))
        self.assertEqual(set(combined), set(self.plan["tasks"]))

    def test_public_matrix_configs_cover_all_eighteen_tasks_once(self):
        root = Path(__file__).resolve().parents[1]
        collected = []
        for version, task_count, worker_count in (("v0517", 11, 8), ("v0518", 7, 7)):
            config = root / "example_configs" / f"top5_parallel_verify_public_{version}_mi355x.yaml"
            declaration = verifier.parallel_plan(config, "0,1,2,3,4,5,6,7", root)
            self.assertEqual(len(declaration["plan"]["tasks"]), task_count)
            self.assertEqual(len(declaration["workers"]), worker_count)
            self.assertEqual(declaration["unused_gpu_ids"], [] if version == "v0517" else ["7"])
            collected.extend(task for worker in declaration["workers"] for task in worker["tasks"])
        expected = {path.parent.relative_to(root / "tasks").as_posix()
                    for path in (root / "tasks/head_kernels").rglob("config.yaml")}
        self.assertEqual(len(collected), 18)
        self.assertEqual(len(collected), len(set(collected)))
        self.assertEqual(set(collected), expected)

    def test_surplus_gpus_are_idle_without_empty_workers(self):
        self.plan["tasks"] = self.plan["tasks"][:7]
        declaration = self.declaration()
        self.assertEqual(len(declaration["workers"]), 7)
        self.assertEqual(declaration["unused_gpu_ids"], ["7"])
        self.assertTrue(all(worker["tasks"] for worker in declaration["workers"]))

    def test_invalid_gpu_selection_is_rejected(self):
        for devices in ("", "0,0", "0,00", "-1", "gpu0"):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                self.declaration(devices)

    def test_empty_duplicate_or_invalid_shard_is_rejected(self):
        for tasks, index, count in (([], 0, 1), (["a", "a"], 0, 1), (["a"], 0, 2),
                                    (["a"], 1, 1), (["a"], -1, 1)):
            with self.subTest(tasks=tasks, index=index, count=count), self.assertRaises(ValueError):
                verifier.select_task_shard(tasks, index, count)

    def test_aggregation_requires_complete_successful_coverage(self):
        declaration = self.declaration()
        self.reports(declaration)
        code, path = verifier.aggregate_parallel(self.root, [0] * 8)
        result = json.loads(path.read_text())
        self.assertEqual(code, 0)
        self.assertFalse(result["framework_PASS_claimed"])
        self.assertEqual(result["status"], "all_native_phases_succeeded")
        self.assertEqual(verifier.aggregate_parallel(self.root, [0, 1, 0, 0, 0, 0, 0, 0])[0], 1)

    def test_missing_or_duplicate_task_report_fails(self):
        declaration = self.declaration()
        self.reports(declaration)
        path = self.root / "worker-000/direct-verification.json"
        report = json.loads(path.read_text())
        report["tasks"][0]["task"] = declaration["workers"][1]["tasks"][0]
        path.write_text(json.dumps(report))
        self.assertEqual(verifier.aggregate_parallel(self.root, [0] * 8)[0], 1)
        path.unlink()
        self.assertEqual(verifier.aggregate_parallel(self.root, [0] * 8)[0], 1)

    def test_worker_task_failure_fails_even_with_zero_exit(self):
        declaration = self.declaration()
        self.reports(declaration)
        path = self.root / "worker-004/direct-verification.json"
        report = json.loads(path.read_text())
        report["tasks"][0]["status"] = "native_report_failed_or_missing"
        path.write_text(json.dumps(report))
        self.assertEqual(verifier.aggregate_parallel(self.root, [0] * 8)[0], 1)


if __name__ == "__main__":
    unittest.main()
