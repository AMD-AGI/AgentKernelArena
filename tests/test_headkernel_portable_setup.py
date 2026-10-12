import hashlib
import importlib.util
import json
from copy import deepcopy
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

    def test_all_23_staging_protocols_resolve_task_contracts(self):
        current = prepare.selected_tasks(self.catalog, None)
        self.assertEqual(len(current), 23)
        self.assertEqual(len(current), self.catalog['fixture_preparation_task_count'])
        for row in current:
            config = yaml.safe_load((ROOT/row['capture_config']).read_text())
            self.assertEqual(config.get('platform_support', {}).get('status', 'active'), 'active')
            self.assertEqual(row['runtime_image'], self.catalog['images']['sglang_v0520']['pull_reference'])
            if row['preparation_protocol'] == prepare.LEGACY_PROTOCOL:
                self.assertTrue(prepare.requires_legacy_protocol(ROOT, row))
                path = ROOT / row['legacy_artifact_manifest']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row['legacy_artifact_manifest_sha256'])
                manifest = prepare.legacy_artifacts.load_manifest(path)
                assets = prepare.legacy_artifacts.select_artifacts(manifest, [prepare.legacy_task_name(row['task'])])
                self.assertEqual(row['fixture_oci_prefix'], manifest.oci_remote_root)
                self.assertEqual(row['fixture_asset_count'], len(assets))
                self.assertEqual(row['fixture_declared_bytes'], sum(item.size_bytes for item in assets))
                self.assertIsNone(row['fixture_manifest'])
                continue
            self.assertEqual(row['preparation_protocol'], prepare.TRUSTED_PROTOCOL)
            self.assertFalse(prepare.requires_legacy_protocol(ROOT, row))
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
            self.assertEqual(row['preparation_protocol'], prepare.LEGACY_PROTOCOL)
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
            self.assertEqual(prepare.selected_tasks(self.catalog, [row['task']]), [row])
        self.assertEqual(sum(row['fixture_asset_count'] for row in qwen), 3)
        self.assertEqual(sum(row['fixture_declared_bytes'] for row in qwen), 3664105258)
        with self.assertRaises(ValueError):
            prepare.selected_tasks(self.catalog, ['headkernel/unknown'])

    def test_both_scopes_require_an_explicit_valid_legacy_protocol(self):
        qwen = next(row for row in self.catalog['tasks'] if row['scope'] == 'source_refreshed_sg520')
        changes = {
            'missing_protocol': lambda row: row.pop('preparation_protocol'),
            'unsupported_protocol': lambda row: row.update(preparation_protocol='guess_from_scope'),
            'missing_fixture_declaration': lambda row: row.pop('fixture_manifest'),
            'external_manifest': lambda row: row.update(fixture_manifest='fixtures/EXTERNAL-MANIFEST.json'),
            'wrong_registry': lambda row: row.update(legacy_artifact_manifest='../untrusted.json'),
            'missing_registry_pin': lambda row: row.pop('legacy_artifact_manifest_sha256'),
            'malformed_registry_pin': lambda row: row.update(legacy_artifact_manifest_sha256='not-a-digest'),
        }
        for scope in prepare.PREPARATION_SCOPES:
            valid = {**qwen, 'scope':scope}
            self.assertEqual(prepare.selected_tasks({'tasks':[valid]}, None), [valid])
            for name, change in changes.items():
                with self.subTest(scope=scope, mutation=name):
                    invalid = deepcopy(valid)
                    change(invalid)
                    with self.assertRaises(ValueError):
                        prepare.selected_tasks({'tasks':[invalid]}, None)
        invalid = {**qwen, 'preparation_protocol':prepare.TRUSTED_PROTOCOL}
        with self.assertRaises(ValueError):
            prepare.selected_tasks({'tasks':[invalid]}, None)

    def test_trusted_protocol_is_explicit_and_duplicate_catalog_tasks_are_rejected(self):
        row = deepcopy(next(row for row in self.catalog['tasks'] if row['preparation_protocol'] == prepare.TRUSTED_PROTOCOL))
        del row['preparation_protocol']
        with self.assertRaises(ValueError):
            prepare.selected_tasks({'tasks':[row]}, None)
        with self.assertRaises(ValueError):
            prepare.selected_tasks({'tasks':[self.catalog['tasks'][0]] * 2}, None)

    def test_unified_selection_keeps_history_and_binds_scoped_qualifications(self):
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
        qualified = [row for row in rows if row.get('scoped_qualification')]
        self.assertEqual({row['task'] for row in qualified}, {
            'headkernel/deepseek-v4-pro__unified_paged_attention_prefill',
            'headkernel/minimax-m3__gemm_afp4wfp4_kernel',
            'headkernel/kimi-k3__lean_attention_decode',
            'headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm',
            'headkernel/qwen3.8-2.4t__paged_attention_decode',
            'headkernel/qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode',
            'headkernel/qwen3.8-2.4t__fused_moe_2stage_mxfp4',
            'headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster',
        })
        for row in rows:
            approved = 'scoped_qualification' in row
            for flag in ('ready', 'framework_qualified', 'trusted_retest_qualified'):
                self.assertEqual(row[flag], approved)
            if not approved:
                continue
            evidence = row['scoped_qualification']
            receipt = json.loads((ROOT / evidence['receipt']).read_text())
            review = json.loads((ROOT / evidence['compatibility_review']).read_text())
            self.assertIn(row['task'], receipt['ready_tasks'])
            self.assertTrue((ROOT / evidence['guide']).is_file())
            self.assertEqual(evidence['framework_execution_commit'], receipt['qualified_framework_commit'])
            self.assertEqual(evidence['trusted_execution_commit'], receipt['qualified_trusted_commit'])
            reviewed_commit = review.get('compatible_checkout_commit', review.get('current_commit'))
            self.assertEqual(evidence['compatible_checkout_commit'], reviewed_commit)
            self.assertEqual(evidence['new_framework_run_at_compatible_checkout'],
                             evidence['framework_execution_commit'] == reviewed_commit)
            self.assertEqual(evidence['new_trusted_retest_at_compatible_checkout'],
                             evidence['trusted_execution_commit'] == reviewed_commit)
            self.assertFalse(evidence['accepted_gain'])
            self.assertFalse(evidence['model_serving_qualification_claimed'])
            self.assertFalse(evidence['strict_snapshot_verifier_acceptance_changed'])
        historical = [row for row in rows if 'historical_ready_receipt' in row]
        self.assertEqual(len(historical), 5)
        self.assertEqual(self.catalog['unfinished_task_count'], len(rows) - len(qualified))
        self.assertEqual(self.catalog['tasks_without_historical_ready_receipt_count'],
                         len(rows) - len(historical))
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
