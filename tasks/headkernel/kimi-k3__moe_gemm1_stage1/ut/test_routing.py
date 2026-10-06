"""CPU-only routing regressions; no fixture payloads or device calls are needed."""
from collections import Counter
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fresh_runner import FreshCallbacks
import routing
import runtime_adapter

ROOT = Path(__file__).resolve().parents[1]


def geometry(case):
    tensors = case['tensors']
    tokens = tensors.get('inputs.a', tensors.get('inputs.out'))['shape'][0]
    controls = case['scalars']['arguments']
    tile = controls.get('tile_m', controls.get('block_m'))
    return tokens, tile, tensors['inputs.sorted_token_ids']['shape'][0], tensors['inputs.sorted_expert_ids']['shape'][0]


def assert_routes(test, routes, tokens, tile, rows, capacity, slots, expected=None, padding_token=None):
    """Check the realized arrays independently of the degree-construction code."""
    test.assertEqual(list(routes['num_valid_ids']), [rows, tokens])
    test.assertEqual(len(routes['sorted_token_ids']), capacity)
    test.assertEqual(len(routes['sorted_expert_ids']), slots)
    test.assertEqual(len(routes['topk_ids']), tokens * 16)
    sentinel = (16 << 24) | (tokens if padding_token is None else padding_token)
    seen = bytearray(tokens * 16)
    live_counts = Counter()
    for row, fused in enumerate(routes['sorted_token_ids'][:rows]):
        token, slot = fused & 0xffffff, fused >> 24
        if token >= tokens or not 0 <= slot < 16:
            test.assertEqual(fused, sentinel)
            continue
        expert = routes['sorted_expert_ids'][row // tile]
        test.assertTrue(0 <= expert < 896)
        index = token * 16 + slot
        test.assertEqual(seen[index], 0)
        seen[index] = 1
        test.assertEqual(routes['topk_ids'][index], expert)
        live_counts[expert] += 1
    test.assertEqual(sum(seen), tokens * 16)
    for token in range(tokens):
        test.assertEqual(len(set(routes['topk_ids'][token * 16:(token + 1) * 16])), 16)
    histogram = Counter(routes['sorted_expert_ids'][:rows // tile])
    test.assertEqual(histogram, Counter({e: (n + tile - 1) // tile for e, n in live_counts.items()}))
    test.assertTrue(all(n <= tokens for n in live_counts.values()))
    test.assertTrue(all(n == -1 for n in routes['sorted_expert_ids'][rows // tile:]))
    test.assertTrue(all(n == sentinel for n in routes['sorted_token_ids'][rows:]))
    if expected is not None:
        test.assertEqual(histogram, Counter(dict(expected)))


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'cases.json').read_text())

    def test_all_retained_expert_ids_and_block_counts_are_realized(self):
        count = 0
        for case in self.manifest['cases']:
            tokens, tile, capacity, slots = geometry(case)
            examples = routing.retained_populations(case)
            self.assertEqual(len(examples), 9)
            for example in examples:
                with self.subTest(case=case['case_id'], population=example['population_id']):
                    rows = example['num_valid_ids'][0]
                    routes = routing.make_routes(tokens, tile, rows, capacity, slots, 1901,
                                                  example['expert_sorted_block_histogram'])
                    assert_routes(self, routes, tokens, tile, rows, capacity, slots,
                                  example['expert_sorted_block_histogram'])
                    summary = routing.population_summary(routes, tile, example)
                    self.assertEqual(summary['population_id'], example['population_id'])
                    self.assertEqual(summary['routing_kind'], 'retained_population')
                    self.assertEqual(example['observed_on_ranks'], list(range(8)))
                    count += 1
        self.assertEqual(count, 27)

    def test_every_histogram_bin_has_feasible_skew_capable_degrees(self):
        count = 0
        for case in self.manifest['cases']:
            tokens, tile, capacity, slots = geometry(case)
            for rows, frequency in case['work_distribution']['valid_rows_histogram']:
                self.assertGreater(frequency, 0)
                self.assertLessEqual(rows, capacity)
                self.assertLessEqual(rows // tile, slots)
                for seed in self.manifest['measurement']['correctness_seeds']:
                    population, degrees = routing.route_population(tokens, tile, rows, random.Random(seed))
                    self.assertEqual(sum(degrees), tokens * 16)
                    self.assertEqual(sum(b for _, b in population), rows // tile)
                    self.assertEqual(len(set(e for e, _ in population)), len(population))
                    self.assertTrue(all(0 <= e < 896 and (n + tile - 1) // tile == b and 0 < n <= tokens
                                        for (e, b), n in zip(population, degrees)))
                count += 1
        self.assertEqual(count, 503)

    def test_fresh_realized_routes_at_small_middle_and_large_work_counts(self):
        for case in self.manifest['cases']:
            tokens, tile, capacity, slots = geometry(case)
            histogram = case['work_distribution']['valid_rows_histogram']
            for index, seed in zip((0, len(histogram) // 2, -1), (0, 1, 2)):
                rows = histogram[index][0]
                routes = routing.make_routes(tokens, tile, rows, capacity, slots, seed, padding_token=0)
                assert_routes(self, routes, tokens, tile, rows, capacity, slots, padding_token=0)
                self.assertEqual(routes, routing.make_routes(tokens, tile, rows, capacity, slots, seed, padding_token=0))
                other = routing.make_routes(tokens, tile, rows, capacity, slots, seed + 100)
                self.assertNotEqual(routes['topk_ids'], other['topk_ids'])

    def test_concrete_prefill_counterexample_is_preserved(self):
        found = []
        for case in self.manifest['cases']:
            for example in routing.retained_populations(case):
                if (example['num_valid_ids'] == [290304, 16384]
                        and len(example['expert_sorted_block_histogram']) == 690
                        and max(n for _, n in example['expert_sorted_block_histogram']) == 167):
                    found.append(example)
        self.assertEqual(len(found), 1)

    def test_sampling_law_and_timing_seeds_stay_separate_from_population_rng(self):
        for case in self.manifest['cases']:
            tokens, tile, _, _ = geometry(case)
            histogram = case['work_distribution']['valid_rows_histogram']
            seeds = range(1971, 1971 + 10 + 100)
            before = [runtime_adapter.weighted_work(histogram, random.Random(seed)) for seed in seeds]
            after = []
            populations = set()
            for seed in seeds:
                rows = runtime_adapter.weighted_work(histogram, random.Random(seed))
                population, _ = routing.route_population(tokens, tile, rows, random.Random(seed))
                populations.add(tuple(population))
                after.append(rows)
            self.assertEqual(before, after)
            self.assertEqual(len(after[10:]), 100)
            self.assertGreater(len(populations), 1)
            if tokens > 64:
                self.assertTrue(any(max(n for _, n in p) - min(n for _, n in p) > 1 for p in populations))

    def test_illegal_populations_and_allocations_fail(self):
        legal = [[e, 1] for e in range(32)]
        bad = [[], [[e, 1] for e in range(31)], legal[:-1] + [[0, 1]],
               legal[:-1] + [[896, 1]], legal[:-1] + [[31, 0]],
               legal[:-1] + [[31, 3]], legal[:-1] + [[True, 1]]]
        for population in bad:
            with self.subTest(population=population), self.assertRaises(ValueError):
                routing.make_routes(64, 32, 1024, 29680, 928, 0, population)
        for args in ((64, 32, 1024, 1023, 928), (64, 32, 1024, 29680, 31),
                     (64, 32, 320, 29680, 928), (64, 32, 1025, 29680, 928)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                routing.make_routes(*args, 0)

    def test_targeted_callback_visits_all_examples_and_clears_override_on_failure(self):
        case = self.manifest['cases'][0]
        calls = []
        prepared = SimpleNamespace(case=case, route_population=None, observe=lambda: case)
        def check(seed):
            calls.append((prepared.route_population['population_id'], seed))
            return True
        prepared.callbacks = SimpleNamespace(check_once=check)
        result = routing.check_retained_populations(prepared, 819)
        self.assertEqual(calls, [(e['population_id'], 819) for e in routing.retained_populations(case)])
        self.assertEqual(len(result), 9)
        self.assertIsNone(prepared.route_population)
        prepared.callbacks.check_once = lambda seed: False
        with self.assertRaisesRegex(AssertionError, 'oracle did not pass'):
            routing.check_retained_populations(prepared, 819)
        self.assertIsNone(prepared.route_population)

    def test_correctness_extension_inherits_performance_and_preserves_existing_checks(self):
        self.assertIs(routing.RoutingCallbacks.performance_row, FreshCallbacks.performance_row)
        self.assertIs(routing.RoutingCallbacks.verify, FreshCallbacks.verify)
        case = self.manifest['cases'][0]
        callbacks = object.__new__(routing.RoutingCallbacks)
        callbacks.prepared = SimpleNamespace(case=case)
        original = {'case': case, 'seeds': [0, 1, 2], 'negative_controls': {'no_op': True, 'wrong_output': True}}
        with patch.object(FreshCallbacks, 'correctness_row', return_value=copy.deepcopy(original)) as base:
            with patch.object(routing, 'check_retained_populations', return_value=['checked']) as extra:
                result = callbacks.correctness_row(case, self.manifest['measurement'],
                    observe=lambda: case, corrupt_outputs=lambda: None, challenge_seed=819)
        self.assertEqual({k: v for k, v in result.items() if k != 'retained_route_populations'}, original)
        base.assert_called_once()
        extra.assert_called_once_with(callbacks.prepared, 819)
        self.assertEqual(result['retained_route_populations'], ['checked'])


if __name__ == '__main__':
    unittest.main()
