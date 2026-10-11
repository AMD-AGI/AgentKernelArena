"""CPU checks using a supplied authentic six-phase proof directory."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

import check_timing as gate


class TimingTests(unittest.TestCase):
    def test_actual_stable_same_source_proof_is_control_without_gain(self):
        quality = gate.read_and_assess(PROOF)
        self.assertEqual(quality['status'], 'pass')
        self.assertEqual(quality['comparison_status'], 'unchanged_source_control')
        self.assertFalse(quality['accepted_gain'])
        self.assertIsNone(quality['accepted_arithmetic_mean_speedup'])
        self.assertAlmostEqual(quality['raw_arithmetic_mean_speedup'], 0.9955920201151162)

    def test_injected_68ms_outlier_rejects_without_trimming(self):
        measurement, reports = deepcopy(MEASUREMENT), deepcopy(REPORTS)
        report = reports['reference']['performance']; row = report['cases'][0]
        row['samples_ms'][0] = 68.0
        report.pop('test_cases')
        report['test_cases'] = gate.validate_report(report, gate.MANIFEST, report['request'])
        for item in measurement['cases']:
            item['reference_ms'] = next(x['execution_time_ms'] for x in report['test_cases'] if x['test_case_id'] == item['test_case_id'])
            item['speedup'] = item['reference_ms'] / item['candidate_ms']
        measurement['arithmetic_mean_speedup'] = math.fsum(x['speedup'] for x in measurement['cases']) / 2
        untouched = deepcopy((measurement, reports))
        quality = gate.validate_and_assess(measurement, reports)
        self.assertEqual(quality['status'], 'reject')
        self.assertFalse(quality['accepted_gain'])
        self.assertIsNone(quality['accepted_arithmetic_mean_speedup'])
        self.assertEqual(quality['cases'][0]['reference']['work_classes'][0]['reasons'], ['dominant_extreme_replay'])
        self.assertEqual(quality['cases'][0]['reference']['sample_count'], 100)
        self.assertEqual((measurement, reports), untouched)

    def test_missing_native_case_rejects_before_timing_admission(self):
        reports = deepcopy(REPORTS)
        reports['candidate']['performance']['cases'].pop()
        with self.assertRaises(ValueError):
            gate.validate_and_assess(MEASUREMENT, reports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--measurement-dir', required=True)
    args = parser.parse_args(); PROOF = Path(args.measurement_dir)
    MEASUREMENT = json.loads((PROOF / 'trusted_measurement.json').read_text())
    REPORTS = {leg: {phase: json.loads((PROOF / (leg + '_' + phase + '.json')).read_text())
                     for phase in ('compile', 'correctness', 'performance')} for leg in ('reference', 'candidate')}
    unittest.main(argv=['test_timing.py'])
