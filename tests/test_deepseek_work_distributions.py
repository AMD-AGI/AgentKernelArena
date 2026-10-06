"""Portable CPU regressions; no hydrated tensor assets or GPU runtime required."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import unittest
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'tasks/headkernel'
MOE_NAMES = (
    'deepseek-v4-pro__moe_stage1_grouped_gemm_silu_flydsl',
    'deepseek-v4-pro__moe_stage1_grouped_gemm_silu_opus_a8w4',
    'deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4',
)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeepSeekDistributionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.moe = module_at('reviewed_moe_distribution', TASKS / MOE_NAMES[0] / 'ut/work_distribution.py')
        cls.contracts = []
        for name in MOE_NAMES:
            task = TASKS / name
            manifest = json.loads((task / 'cases.json').read_text())
            base, registry = cls.moe.load_contract(task, manifest)
            cls.contracts.append((task, manifest, base, registry))

    def test_full_histograms_targeted_default_and_optional_exhaustive(self):
        fixed = total = targeted = exhaustive = 0
        for _, manifest, base, registry in self.contracts:
            fixed += len(base['cases']); total += len(manifest['cases'])
            self.assertEqual(manifest['cases'][:len(base['cases'])], base['cases'])
            self.assertEqual(manifest['measurement'], base['measurement'])
            self.assertEqual(manifest['tolerance'], base['tolerance'])
            for group in registry['groups'].values():
                mode, selected = self.moe.correctness_variants(group)
                other_mode, all_rows = self.moe.correctness_variants(group, 'exhaustive_observed_values')
                self.assertEqual(mode, 'targeted_uncovered_behaviors')
                self.assertEqual(other_mode, 'exhaustive_observed_values')
                self.assertEqual(len(selected), 13)
                self.assertLess(len(selected), len(all_rows))
                self.assertIn(all_rows[0], selected); self.assertIn(all_rows[-1], selected)
                targeted += len(selected); exhaustive += len(all_rows)
        self.assertEqual((fixed, total, targeted, exhaustive), (24, 28, 52, 750))

    def test_original_case_or_validation_policy_changes_are_rejected(self):
        task, manifest, _, _ = self.contracts[0]
        for mutate in (lambda m: m['cases'].pop(0),
                       lambda m: m['measurement'].__setitem__('benchmark_iterations', 99),
                       lambda m: m.__setitem__('original_served_work_control_gate_passed', True)):
            changed = copy.deepcopy(manifest); mutate(changed)
            with self.assertRaises(ValueError): self.moe.load_contract(task, changed)

    def test_histogram_rare_value_omission_is_rejected(self):
        task, manifest, _, registry = self.contracts[0]
        changed = copy.deepcopy(registry)
        group = next(iter(changed['groups'].values()))
        group['histogram'].remove(min(group['histogram'], key=lambda row: row['occurrences']))
        encoded = json.dumps(changed)
        original_read = Path.read_text
        def replacement(path, *args, **kwargs):
            if path == task / manifest['observed_work_distributions']['path']: return encoded
            return original_read(path, *args, **kwargs)
        with patch.object(Path, 'read_text', replacement), self.assertRaises(ValueError):
            self.moe.load_contract(task, manifest)

    def test_every_integer_histogram_bucket_is_reachable(self):
        for _, _, _, registry in self.contracts:
            for group in registry['groups'].values():
                start = 0
                for row in group['histogram']:
                    for ticket in (start, start + row['occurrences'] - 1):
                        with patch.object(self.moe.random, 'Random') as constructor:
                            constructor.return_value.randrange.return_value = ticket
                            self.assertEqual(self.moe.weighted_variant(group, 42), row)
                            constructor.return_value.randrange.assert_called_once_with(group['observed_calls'])
                    start += row['occurrences']
                self.assertEqual(start, group['observed_calls'])

    def test_mla_policy_and_all_observed_CSR_widths(self):
        task = TASKS / 'deepseek-v4-pro__unified_paged_attention_decode'
        module = module_at('reviewed_mla_distribution', task / 'ut/mla_decode_distribution.py')
        manifest = json.loads((task / 'cases.json').read_text())
        policy = module.load_policy(task, manifest)
        self.assertEqual(policy['lengths'], list(range(192, 201)))
        indices = list(range(12800)) + [-1] * (16384 - 12800)
        indptr = [200 * row for row in range(65)]
        for width in policy['lengths']:
            result, pointer = module.controls_for_length(indices, indptr, width)
            self.assertEqual(pointer, [width * row for row in range(65)])
            for row in range(64):
                self.assertEqual(result[row * width:(row + 1) * width], list(range(row * 200, row * 200 + width)))
            self.assertEqual(result[64 * width:], [-1] * (16384 - 64 * width))
        with self.assertRaises(ValueError): module.controls_for_length(indices, indptr, 191)

    def test_opus2_progress_is_durable_and_outside_unchanged_callbacks(self):
        task = TASKS / MOE_NAMES[2]
        current = ast.parse((task / 'scripts/task_runner.py').read_text())
        original = ast.parse((task / 'provenance/BASE-RUNNER.py').read_text())
        current_functions = {node.name: node for node in current.body if isinstance(node, ast.FunctionDef)}
        for node in original.body:
            if isinstance(node, ast.FunctionDef) and node.name != 'main':
                self.assertEqual(ast.dump(node), ast.dump(current_functions[node.name]))
        main = current_functions['main']
        for callback in [node for node in ast.walk(main) if isinstance(node, ast.FunctionDef) and node is not main]:
            self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                                 and node.func.id == 'case_progress' for node in ast.walk(callback)))
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'case_progress']
        self.assertEqual(sorted(node.args[0].value for node in calls), ['completed', 'completed', 'start'])
        self.assertIn('performance_timeout: 7200', (task / 'config.yaml').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'build').mkdir()
            namespace = {'ROOT': root, 'canonical': json.dumps, 'os': os, 'time': time}
            isolated = ast.Module(body=[current_functions['case_progress']], type_ignores=[])
            exec(compile(isolated, '<progress CPU check>', 'exec'), namespace)
            with patch.object(os, 'fsync') as fsync:
                namespace['case_progress']('start', 14, 14, 'distribution-prefill', 'request-1')
                namespace['case_progress']('completed', 14, 14, 'distribution-prefill', 'request-1')
                self.assertEqual(fsync.call_count, 2)
            rows = [json.loads(line) for line in (root / 'build/performance_progress.jsonl').read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows], ['start', 'completed'])
            self.assertTrue(all(row['scoreable'] is False and row['case_count'] == 14 for row in rows))


if __name__ == '__main__': unittest.main()
