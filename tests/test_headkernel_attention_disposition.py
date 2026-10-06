"""Current-workload discovery policy for superseded attention packages."""

import json
import logging
from pathlib import Path
import unittest

import yaml

from main import should_run_task_for_platform


ROOT = Path(__file__).resolve().parents[1]


class AttentionDispositionTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads(
            (ROOT / "docs/reference/headkernel-superseded-attention.json").read_text()
        )
        self.logger = logging.getLogger(__name__)

    def test_superseded_repository_tasks_are_filtered_before_setup(self):
        for record in self.catalog["records"]:
            if record["repository_task"] is None:
                continue
            with self.subTest(task=record["task_id"]):
                task = ROOT / record["repository_task"]
                config = yaml.safe_load((task / "config.yaml").read_text())
                self.assertFalse(
                    should_run_task_for_platform(
                        record["task_id"], config, "gfx950", self.logger
                    )
                )
                self.assertEqual(
                    config["headkernel"]["replacement_tasks"],
                    record["replacement_tasks"],
                )
                evidence = task / config["headkernel"]["disposition_evidence"]
                self.assertEqual(
                    json.loads(evidence.read_text())["disposition"],
                    "SUPERSEDED_FOR_CURRENT_WORKLOAD",
                )

    def test_replacement_and_current_kimi_moe_tasks_remain_selectable(self):
        paths = {
            path
            for record in self.catalog["records"]
            for path in record["replacement_paths"]
        }
        paths.update(
            f"tasks/headkernel/kimi-k3__{suffix}"
            for suffix in ("moe_gemm1_stage1", "moe_gemm2_stage2")
        )
        for path in sorted(paths):
            with self.subTest(task=path):
                config = yaml.safe_load((ROOT / path / "config.yaml").read_text())
                self.assertTrue(
                    should_run_task_for_platform(path, config, "gfx950", self.logger)
                )

    def test_dispatch_disposition_does_not_claim_qualification(self):
        self.assertFalse(self.catalog["ready"])
        for record in self.catalog["records"]:
            self.assertFalse(record["qualification_claim"])
        self.assertEqual(
            set(self.catalog["required_current_heads_not_retired"]),
            {
                "kimi-k3__dense_bf16_gemm_cijk",
                "kimi-k3__score_combine_mix_fused",
                "minimax-m3__gemm_afp4wfp4_kernel",
            },
        )


if __name__ == "__main__":
    unittest.main()
