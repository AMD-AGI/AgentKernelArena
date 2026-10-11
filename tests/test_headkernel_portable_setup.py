import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_headkernel_run', ROOT/'tools/prepare_headkernel_run.py')
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PortableSetupTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT/'tools/headkernel-runtime-targets.json').read_text())

    def test_manifest_native_staging_selection_resolves_task_contracts(self):
        current = prepare.selected_tasks(self.catalog, None)
        self.assertEqual(len(current), 18)
        self.assertEqual(len(current), self.catalog['fixture_preparation_task_count'])
        for row in current:
            config = yaml.safe_load((ROOT/row['capture_config']).read_text())
            self.assertEqual(config['platform_support']['status'], 'active')
            self.assertEqual(row['capture_image'], self.catalog['images']['sglang_v0520']['pull_reference'])
            if row['fixture_manifest']:
                path = ROOT/row['fixture_manifest']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['fixture_manifest_sha256'])
                manifest = json.loads(path.read_text())
                self.assertEqual(manifest['oci_prefix'], row['fixture_oci_prefix'])
                self.assertEqual(len(manifest['assets']), row['fixture_asset_count'])
                self.assertEqual(sum(item['bytes'] for item in manifest['assets']), row['fixture_declared_bytes'])

    def test_source_refreshed_qwen_keeps_separate_setup_and_capture_provenance(self):
        image = self.catalog['images']['sglang_v0520']['pull_reference']
        qwen = [row for row in self.catalog['tasks'] if 'qwen3.8' in row['task']]
        self.assertEqual(len(qwen), 5)
        for row in qwen:
            self.assertEqual(row['scope'], 'source_refreshed_sg520')
            self.assertEqual(row['runtime_image'], image)
            self.assertEqual(row['primary_target'], 'sglang_v0520')
            self.assertIn('v0.5.18-', row['capture_image'])
            self.assertEqual(row['current_runtime_workload_capture'], 'PENDING_FULL_SG0520_CAPTURE')
            path = ROOT / row['source_refresh_manifest']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['source_refresh_manifest_sha256'])
            self.assertEqual(json.loads(path.read_text())['runtime_image'], image)
            config = yaml.safe_load((ROOT / row['capture_config']).read_text())
            self.assertEqual(config['headkernel']['docker'], image)
            self.assertTrue((ROOT / row['environment_and_setup_guide']).is_file())
        # The existing materializer does not implement the Qwen task-local input route.
        with self.assertRaises(ValueError):
            prepare.selected_tasks(self.catalog, ['headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster'])
        with self.assertRaises(ValueError):
            prepare.selected_tasks(self.catalog, ['headkernel/unknown'])

    def test_unified_selection_contains_all_23_current_sources_without_promoting_history(self):
        rows = self.catalog['tasks']
        config = yaml.safe_load((ROOT / 'example_configs/headkernel_sg520_current_mi355x.yaml').read_text())
        self.assertEqual(len(rows), 23)
        self.assertEqual(len(config['tasks']), 23)
        self.assertEqual(set(config['tasks']), {row['task'] for row in rows})
        self.assertEqual(self.catalog['refresh_scope']['active_task_count'], 23)
        self.assertEqual(self.catalog['refreshed_task_count'], 23)
        self.assertEqual(self.catalog['refresh_scope']['excluded_models'], [])
        self.assertEqual(len(self.catalog['refresh_scope']['models']), 5)
        self.assertFalse(self.catalog['ready'])
        self.assertTrue(all(not row['ready'] and not row['framework_qualified']
                            and not row['trusted_retest_qualified'] for row in rows))
        historical = [row for row in rows if 'historical_ready_receipt' in row]
        self.assertEqual(len(historical), 5)
        self.assertEqual(self.catalog['unfinished_task_count'], len(rows) - len(historical))
        for row in historical:
            receipt = json.loads((ROOT / row['historical_ready_receipt']).read_text())
            self.assertIn(row['task'], receipt['ready_tasks'])
            self.assertTrue((ROOT / row['historical_ready_guide']).is_file())

    def test_prepared_tree_keeps_fixtures_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); arena=root/'arena'; stage=root/'stage'; originals=root/'originals'
            task='tasks/headkernel/example'
            (arena/task).mkdir(parents=True); (arena/task/'source.py').write_text('original')
            (stage/'task/fixtures').mkdir(parents=True)
            (stage/'task/source.py').write_text('original')
            (stage/'task/fixtures/captured.bin').write_bytes(b'real-data')
            (stage/'staging_receipt.json').write_text(json.dumps({'status':'staged_not_evaluated','task_path':task}))
            prepare.install_prepared_task(arena, task, stage, originals)
            self.assertEqual((arena/task/'fixtures/captured.bin').read_bytes(), b'real-data')
            self.assertEqual((originals/task/'source.py').read_text(), 'original')
            self.assertTrue((stage/'staging_receipt.json').exists())

    def test_mismatched_stage_receipt_cannot_replace_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); arena=root/'arena'; stage=root/'stage'; task='tasks/headkernel/example'
            (arena/task).mkdir(parents=True); (arena/task/'source.py').write_text('original')
            (stage/'task').mkdir(parents=True)
            (stage/'staging_receipt.json').write_text(json.dumps({'status':'staged_not_evaluated','task_path':'tasks/other'}))
            with self.assertRaises(ValueError):
                prepare.install_prepared_task(arena, task, stage, root/'originals')
            self.assertEqual((arena/task/'source.py').read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
