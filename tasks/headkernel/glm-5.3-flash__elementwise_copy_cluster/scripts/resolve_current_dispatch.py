"""Resolve scale-copy dispatch only from a finalized full native GLM capture."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]


def read_pin(run,pin):
    original=Path(pin['path']);path=original if original.is_absolute() else run/original
    path=path.resolve()
    if not path.is_relative_to(run) or not path.is_file() or path.is_symlink():raise ValueError('Receipt escapes finalized capture')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=pin['sha256']:raise ValueError('Finalized capture receipt changed')
    return path,json.loads(data)


def resolve(run,ready,mapping):
    if not ready.get('fresh_profile_completed') or not ready.get('source_capture_complete') or ready.get('completed_requests')!=64 or ready.get('total_input_tokens')!=524288 or ready.get('total_output_tokens')!=65536:raise ValueError('Finalized exact64-request source capture required')
    if ready['image']!=mapping['runtime_image']:raise ValueError('Current capture image mismatch')
    _,verified=read_pin(run,ready['source_capture_verified_ref'])
    if not verified.get('complete') or not verified.get('source_capture_complete') or set(verified['ranks'])!={str(x) for x in range(8)}:raise ValueError('All8rank source verification required')
    _,flushed=read_pin(run,{'path':'CAPTURE-FLUSH.json','sha256':verified['flush_receipt_sha256']})
    if flushed['run_id']!=verified['run_id']:raise ValueError('Flush receipt belongs to another run')
    materialized={};alias_only={};views={};pins={}
    for rank,pin in verified['ranks'].items():
        path,record=read_pin(run,pin)
        if record['provenance']['run_id']!=verified['run_id'] or record['provenance']['image']!=ready['image'] or record['provenance']['tp_rank']!=int(rank):raise ValueError('Rank identity mismatch')
        if not record['sealed'] or record['checkpoint_reason'] not in ('profile_stop','native_stop_profile') or record['failures']:raise ValueError('Rank did not finish native source collection')
        summary_ref=flushed['ranks'][rank]['owner_summary']
        _,summary=read_pin(run,{'path':summary_ref['relative_path'],'sha256':summary_ref['sha256']})
        if summary['coverage']!=verified['owner_coverage_by_rank'][rank]:raise ValueError('Owner coverage changed after finalization')
        if not summary.get('complete') or not summary['coverage']['complete']:raise ValueError('Owner required coverage is incomplete')
        for seam in ('scale_copy','scale_view'):
            if summary['installed_seams'][seam]['source_sha256']!=mapping['source_sha256']:raise ValueError('Scale probe source is not pinned current dispatch')
        actual=[];aliases=[]
        for key,schema in record['case_schemas'].items():
            if schema['family']!='scale_copy':continue
            count=record['runtime_case_counts'][key]
            if type(count) is not int or count<=0:raise ValueError('Invalid actual scale-copy count')
            row={'case_key':key,'occurrences':count,'stage':schema['stage'],'input':schema['inputs']['scale'],'output':schema['outputs']['result']}
            (aliases if row['input']['alias']==row['output']['alias'] else actual).append(row)
        materialized[rank]=actual;alias_only[rank]=aliases;views[rank]=summary['view_call_counts'].get('scale_view',0)
        if not actual and not aliases and views[rank]<=0:raise ValueError('No runtime evidence for scale dispatch on rank'+rank)
        pins[rank]={'path':str(path.relative_to(run)),'sha256':pin['sha256']}
    status='CURRENT_COPY_OBSERVED_REBUILD_REQUIRED' if any(materialized.values()) else 'NOT_OBSERVED_CURRENT'
    return {'status':status,'capture_run':verified['run_id'],'runtime_image':ready['image'],'materialized_cases_by_rank':materialized,'alias_only_calls_by_rank':alias_only,'view_call_counts_by_rank':views,'native_stop_manifest_pins':pins,'historical_task_retired':status=='NOT_OBSERVED_CURRENT','timing_or_speedup_claim':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture-ready',required=True);args=parser.parse_args()
    path=Path(args.capture_ready).resolve();run=path.parent;mapping=json.loads((ROOT/'CURRENT-MAPPING.json').read_text());ready=json.loads(path.read_text())
    disposition=resolve(run,ready,mapping)
    mapping.update(disposition);mapping['source_run']=disposition['capture_run'];mapping['capture_ready_ref']={'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (ROOT/'CURRENT-MAPPING.json').write_text(json.dumps(mapping,indent=2)+'\n')
    config=yaml.safe_load((ROOT/'config.yaml').read_text());config['headkernel']['current_mapping_status']=disposition['status'];config['headkernel']['source_run']=disposition['capture_run']
    if disposition['historical_task_retired']:
        config['platform_support']={'required_arch':'gfx950','status':'skip','skip_reason':'Retired historical scale-copy head: completed current native capture shows only producer-layout views/alias-only materialize calls and zero actual materialized copy cases on all8ranks.'}
        config['headkernel']['validation_status']='retired_not_observed_current_no_score'
    else:config['headkernel']['validation_status']='current_copy_fixtures_observed_rebuild_required_no_score'
    (ROOT/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,width=100))
    print(json.dumps(disposition,indent=2))
if __name__=='__main__':main()
