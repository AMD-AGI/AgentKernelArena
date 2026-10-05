"""Apply independently verified compatible workload histograms without changing fixtures."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from evaluation_contract import fingerprint,strict_json,validate_manifest
from runtime_adapter import file_sha


def update(compatibility_path):
    path=Path(compatibility_path);compatibility=strict_json(path.read_text())
    if (compatibility.get('schema')!='staged-stage1-v4-compatibility-v1'
            or compatibility.get('structural_contract_covers_v4') is not True
            or compatibility.get('frequency_weights_unchanged') is not True):
        raise ValueError('Independent static-contract and frequency compatibility are required')
    audit_path=path.parent/'AUDIT.json'
    if file_sha(audit_path)!=compatibility['v4_audit_ref']['sha256']:raise ValueError('The independent v4 audit changed')
    audit=strict_json(audit_path.read_text())
    if (any(audit.get(key) is not True for key in ('passed','exact_workload_gate_passed',
            'image_and_runtime_source_pins_verified','prior_stage1_contract_covers_latest'))
            or audit.get('rank_count')!=8):raise ValueError('The v4 workload/source audit did not pass on all eight ranks')
    manifest_path=ROOT/'cases.json';manifest=strict_json(manifest_path.read_text())
    previous=manifest.get('workload_histogram_update',{})
    if file_sha(manifest_path)!=compatibility['staged_cases_ref']['sha256'] and previous.get('compatibility_sha256')!=file_sha(path):
        raise ValueError('Current cases differ from the independently reviewed prior contract')
    if len(compatibility['cases'])!=3:raise ValueError('Exactly three structural updates are required')
    rows={row['staged_case_id']:row for row in compatibility['cases']}
    if len(rows)!=3 or set(rows)!={case['case_id'] for case in manifest['cases']}:raise ValueError('All three structural classes must be updated exactly once')
    summary=[]
    for case in manifest['cases']:
        row=rows[case['case_id']]
        if (any(row.get(key) is not True for key in ('static_controls_and_tensor_ABI_equal',
                'aliases_and_none_arguments_equal','per_rank_counts_equal'))
                or row['all_rank_occurrences']!=case['occurrences']
                or row['input_a_shape']!=case['tensors']['inputs.a']['shape']):
            raise ValueError('The v4 update changes native ABI or structural frequency')
        histogram=row['v4_valid_rows_histogram'];tm=case['scalars']['arguments']['tile_m']
        capacity=case['tensors']['inputs.sorted_token_ids']['shape'][0]
        if (not histogram or any(type(work) is not int or type(count) is not int or work<=0
                or count<=0 or work%tm or work>capacity for work,count in histogram)
                or histogram!=sorted(histogram) or len({work for work,_ in histogram})!=len(histogram)
                or sum(count for _,count in histogram)!=case['occurrences']):
            raise ValueError('Invalid or incomplete empirical workload histogram')
        case['work_distribution']['valid_rows_histogram']=histogram
        case['work_distribution'].pop('exact_numeric_cases',None)
        case['work_distribution']['numeric_work_bins']=len(histogram)
        case['work_distribution']['source_run']=audit['run_id']
        case['work_distribution']['v4_case_id']=row['v4_case_id']
        summary.append({'case_id':case['case_id'],'v4_case_id':row['v4_case_id'],
            'observed_occurrences':case['occurrences'],'valid_row_range':[histogram[0][0],histogram[-1][0]],
            'numeric_work_bins':len(histogram)})
    proof={'schema':'kimi-stage1-workload-histogram-refresh-v1','run_id':audit['run_id'],
           'compatibility_sha256':file_sha(path),'audit_sha256':file_sha(audit_path),
           'prior_case_file_sha256':compatibility['staged_cases_ref']['sha256'],
           'fixture_bytes_and_native_ABI_unchanged':True,'frequency_weights_unchanged':True,'cases':summary}
    manifest['workload_histogram_update']={key:proof[key] for key in ('run_id','compatibility_sha256','audit_sha256')}
    validate_manifest(manifest)
    external_path=ROOT/'fixtures/EXTERNAL-MANIFEST.json';external=strict_json(external_path.read_text())
    external['case_manifest_fingerprint']=fingerprint(manifest)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    external_path.write_text(json.dumps(external,indent=2)+'\n')
    (ROOT/'provenance/STAGE1-WORKLOAD-V4.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--compatibility',required=True)
    args=parser.parse_args();print(json.dumps(update(args.compatibility),indent=2))
