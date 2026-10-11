"""CPU-only regressions using a downloaded complete current proof directory."""
import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mla_current_quality',HERE/'check_timing.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

class Checks(unittest.TestCase):
    def test_current_complete_control_passes_without_gain(self):
        quality=gate.read_and_assess(PROOF)
        self.assertEqual(quality['status'],'pass')
        self.assertFalse(quality['accepted_gain'])
        self.assertEqual(quality['comparison_status'],'unchanged_source_control')
        self.assertAlmostEqual(quality['raw_arithmetic_mean_speedup'],0.9855112940583893)

    def test_missing_case_rejected(self):
        reports=deepcopy(REPORTS);reports['candidate']['performance']['cases'].pop()
        with self.assertRaises(ValueError):gate.validate_and_assess(MEASUREMENT,reports)

    def test_missing_phase_rejected(self):
        reports=deepcopy(REPORTS);del reports['reference']['compile']
        with self.assertRaises(ValueError):gate.validate_and_assess(MEASUREMENT,reports)

    def test_outlier_rejected_without_filtering(self):
        measurement,reports=deepcopy(MEASUREMENT),deepcopy(REPORTS)
        report=reports['reference']['performance'];report['cases'][0]['samples_ms'][0]=68.0
        report.pop('test_cases');report['test_cases']=gate.validate_report(report,gate.MANIFEST,report['request'])
        for row in measurement['cases']:
            row['reference_ms']=next(x['execution_time_ms'] for x in report['test_cases'] if x['test_case_id']==row['test_case_id'])
            row['raw_speedup']=row['reference_ms']/row['candidate_ms']
        measurement['raw_arithmetic_mean_speedup']=math.fsum(row['raw_speedup'] for row in measurement['cases'])/2
        before=deepcopy((measurement,reports));quality=gate.validate_and_assess(measurement,reports)
        self.assertEqual(quality['status'],'reject');self.assertFalse(quality['accepted_gain'])
        self.assertEqual(quality['cases'][0]['reference']['sample_count'],100)
        self.assertEqual((measurement,reports),before)

    def test_changed_postchecker_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ['verify.py','check_timing.py','READY.json','INPUT-PINS.json']:
                shutil.copy2(HERE/name,root/name)
            checker=root/'check_timing.py';checker.write_text(checker.read_text()+'\n# changed after review\n')
            spec=importlib.util.spec_from_file_location('changed_mla_verifier',root/'verify.py')
            verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
            with self.assertRaisesRegex(ValueError,'entry helper changed'):verifier.verify(REPO)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True);parser.add_argument('--measurement-dir',required=True)
    args=parser.parse_args();REPO=Path(args.repo);PROOF=Path(args.measurement_dir)
    gate.configure(REPO)
    MEASUREMENT=json.loads((PROOF/'trusted_measurement.json').read_text())
    REPORTS={leg:{phase:json.loads((PROOF/(leg+'_'+phase+'.json')).read_text())
                   for phase in ('compile','correctness','performance')} for leg in ('reference','candidate')}
    unittest.main(argv=['test_check_timing.py'])
