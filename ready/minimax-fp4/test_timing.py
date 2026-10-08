#!/usr/bin/env python3
"""CPU checks on an authentic completed comparison; original evidence is unchanged."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import check_timing as check

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--measurement-dir',required=True)
    args=parser.parse_args();directory=Path(args.measurement_dir)
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.json')}
    measured=json.loads((directory/'trusted_measurement.json').read_text())
    reports={leg:{phase:json.loads((directory/(leg+'_'+phase+'.json')).read_text())
                  for phase in ('compile','correctness','performance')} for leg in ('reference','candidate')}
    baseline=check.read_and_assess(directory)
    assert baseline['status']=='pass' and not baseline['gain_eligible'] and not baseline['accepted_gain']
    altered=deepcopy(reports);summary=deepcopy(measured)
    sample_row=min(altered['reference']['performance']['cases'],key=lambda row:math.fsum(row['samples_ms']))
    sample_row['samples_ms'][27]=max(68.0,max(sample_row['samples_ms'])*1000)
    case_id=sample_row['case']['case_id'];mean=math.fsum(sample_row['samples_ms'])/100
    for row in altered['reference']['performance']['test_cases']:
        if row['test_case_id']==case_id:row['execution_time_ms']=mean
    for row in summary['cases']:
        if row['test_case_id']==case_id:
            row['reference_ms']=mean;row['speedup']=mean/row['candidate_ms']
    summary['arithmetic_mean_speedup']=math.fsum(row['speedup'] for row in summary['cases'])/14
    rejected=check.validate_and_assess(summary,altered)
    assert rejected['status']=='reject' and not rejected['accepted_gain']
    incomplete=deepcopy(reports);incomplete['candidate']['correctness']['submitted_source_controls'].pop()
    try:check.validate_and_assess(measured,incomplete)
    except ValueError:pass
    else:raise AssertionError('Missing source control was accepted')
    false_negative=deepcopy(reports)
    false_negative['reference']['correctness']['submitted_source_controls'][0]['status']='candidate_accepted'
    try:check.validate_and_assess(measured,false_negative)
    except ValueError:pass
    else:raise AssertionError('Accepted no-op was treated as qualification')
    unrelated_failure=deepcopy(reports)
    unrelated_failure['reference']['correctness']['submitted_source_controls'][0]['error']='out of memory'
    try:check.validate_and_assess(measured,unrelated_failure)
    except ValueError:pass
    else:raise AssertionError('Non-numerical source-control failure was accepted')
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.json')}
    print(json.dumps({'tests_passed':5,'authentic_comparison_stable':True,'unchanged_source_gain_eligible':False,
        'injected_extreme_replay_rejected':True,'missing_or_accepted_source_control_rejected':True,
        'non_numerical_failure_rejected':True,'original_evidence_bytes_unchanged':True,'GPU_actions':False},indent=2))

if __name__=='__main__':main()
