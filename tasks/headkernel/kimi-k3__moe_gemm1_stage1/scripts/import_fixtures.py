"""Seal three verified stage-1 classes and their trusted external fixture inventory."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from evaluation_contract import canonical, fingerprint, strict_json, validate_manifest
from runtime_adapter import IMAGE, NATIVE_SOURCE, FAMILY, DTYPES, NONE_INPUTS, file_sha, validate_fixture

EXPECTED={('decode',64):94208,('prefill',8192):184,('prefill',16384):2852}


def relative_capture(path,root):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Capture artifact must remain within the verified frozen mirror')
    return path.relative_to(root).as_posix()


def structural_projection(value):
    if isinstance(value,dict):
        return {key:structural_projection(item) for key,item in value.items() if key!='device'}
    if isinstance(value,list):return [structural_projection(item) for item in value]
    return value


def seal(handoff_path,capture_root,oci_prefix,materialize=False):
    handoff_path=Path(handoff_path);capture_root=Path(capture_root).resolve()
    handoff=strict_json(handoff_path.read_text())
    if (handoff.get('run_id')!='kimi-served-194550' or handoff.get('exact_workload_and_source_image_passed') is not True
            or handoff.get('all_rank_explicit_post_workload_seals_passed') is not True):
        raise ValueError('Independent full-workload owner audit is required')
    audit_path=Path(handoff['audit_ref']['path'])
    if file_sha(audit_path)!=handoff['audit_ref']['sha256']:raise ValueError('Independent audit changed')
    audit=strict_json(audit_path.read_text())
    if any(audit.get(key) is not True for key in ('full_token_gate_passed','runtime_source_identity_verified',
            'checkpoint_contract_files_verified','explicit_full_workload_stop_receipts_valid','payload_integrity_verified',
            'observed_family_structural_coverage_complete')):
        raise ValueError('Current capture failed independent source/workload/payload checks')
    classes=audit['structural_cases']
    if len(classes)!=3 or any(row['family']!=FAMILY for row in classes):raise ValueError('Exactly three observed stage-1 classes are required')
    rank_manifests={};rank_refs={}
    for rank in range(8):
        path=capture_root/f'tp-{rank}/workload/manifest.json'
        value=strict_json(path.read_text())
        if (value.get('sealed') is not True or value.get('metadata_counts_include_all_notified_served_calls') is not True
                or value.get('checkpoint_reason')!='profile_stop' or value.get('failures')
                or value.get('additional_failure_count',0) or value['provenance']['run_id']!=handoff['run_id']
                or value['provenance']['image']!=IMAGE or value['provenance']['tp_rank']!=rank):
            raise ValueError('Missing complete same-run rank manifest: '+str(rank))
        rank_manifests[str(rank)]=value;rank_refs[str(rank)]={'object_key':relative_capture(path,capture_root),'sha256':file_sha(path)}
    cases=[];assets={};seen_geometry=set();keys_by_rank={str(rank):set() for rank in range(8)}
    for group in classes:
        shape=group['schema']['inputs']['a']['shape'];geometry=(group['stage'],shape[0])
        if geometry not in EXPECTED or geometry in seen_geometry:raise ValueError('Missing or duplicate required structural geometry')
        seen_geometry.add(geometry)
        expected_count=EXPECTED[geometry]
        if group['per_rank_counts']!={str(rank):expected_count for rank in range(8)}:
            raise ValueError('Structural workload frequencies differ from the independently verified run')
        histogram=Counter()
        for rank,keys in group['exact_case_keys_by_rank'].items():
            if set(keys)&keys_by_rank[rank]:raise ValueError('Numerical case frequency assigned to multiple structural classes')
            keys_by_rank[rank].update(keys);rank_count=0
            for key in keys:
                native=rank_manifests[rank];schema=native['case_schemas'][key];count=native['runtime_case_counts'][key]
                if type(count) is not int or count<=0:raise ValueError('Actual positive occurrence count required')
                if (schema['family']!=FAMILY or schema['source']!=NATIVE_SOURCE or schema['stage']!=group['stage']
                        or schema['inputs']['a']['shape']!=shape or canonical(schema['controls'])!=canonical(group['schema']['controls'])):
                    raise ValueError('Exact numerical observation belongs to a different native class')
                work=schema['tensor_controls']['inputs.num_valid_ids']
                if len(work)!=2 or work[1]!=shape[0] or type(work[0]) is not int:raise ValueError('Work-control ABI differs')
                histogram[work[0]]+=count;rank_count+=count
            if rank_count!=expected_count:raise ValueError('Projected frequencies do not sum to the observed class')
        representative=sorted(group['fixture_representatives'],key=lambda row:(row['rank'],row['case_key']))[0]
        path=capture_root/f"tp-{representative['rank']}/profile-fixtures/{representative['case_key']}.json"
        if file_sha(path)!=representative['sha256']:raise ValueError('Selected current fixture differs from independent audit')
        record=strict_json(path.read_text());tokens=validate_fixture(record)
        if tokens!=shape[0] or canonical(record['controls'])!=canonical(group['schema']['controls']):raise ValueError('Fixture does not represent its structural class')
        tensors={}
        for phase,role in (('inputs','input'),('outputs','output')):
            for name,meta in record[phase].items():
                if meta is None:continue
                tensors[phase+'.'+name]={'role':role,'shape':meta['shape'],'strides':meta['stride'],
                    'storage_offset':meta['storage_offset'],'dtype':meta['dtype'].removeprefix('torch.'),'device_type':'cuda'}
        relative='fixtures/'+path.name
        assets[relative]={'path':relative,'object_key':relative_capture(path,capture_root),'bytes':path.stat().st_size,
            'sha256':file_sha(path),'codec':'served-tensor-fixture-v1','roles':['case_metadata']}
        for phase,groups in record['payload'].items():
            for alias,payload in groups.items():
                names=[name for name,meta in record[phase].items() if meta is not None and meta['alias']==alias]
                roles=set()
                for name in names:
                    roles.add('oracle_output' if phase=='outputs' else 'kernel_weight' if name=='w1'
                              else 'kernel_weight_scale' if name=='w1_scale' else 'runtime_control'
                              if name in ('num_valid_ids','sorted_expert_ids','sorted_token_ids','topk_ids') else 'activation')
                for segment in payload['segments']:
                    blob=path.parent/segment['blob'];blob_relative='fixtures/'+segment['blob']
                    if blob.stat().st_size!=segment['bytes'] or file_sha(blob)!=segment['sha256']:raise ValueError('Fixture storage hash/size differs')
                    row={'path':blob_relative,'object_key':relative_capture(blob,capture_root),'bytes':segment['bytes'],
                         'sha256':segment['sha256'],'codec':'raw-storage-segment-v1','roles':sorted(roles)}
                    if blob_relative in assets:
                        previous=assets[blob_relative]
                        if any(previous[key]!=row[key] for key in ('object_key','bytes','sha256','codec')):raise ValueError('Conflicting shared blob identity')
                        row['roles']=sorted(set(previous['roles'])|roles)
                    assets[blob_relative]=row
        cases.append({'case_id':group['structural_key'],'occurrences':sum(histogram.values()),'calls_per_sample':1,
            'tensors':tensors,'scalars':{'arguments':record['controls'],'none_inputs':NONE_INPUTS,'none_outputs':['out']},
            'fixture':{'path':relative,'sha256':file_sha(path)},'stage':group['stage'],
            'per_rank_counts':group['per_rank_counts'],'frequency_scope':'actual complete TP0-7 workload; each structural frequency counted once',
            'work_distribution':{'valid_rows_histogram':[[rows,count] for rows,count in sorted(histogram.items())],
                'exact_numeric_cases':group['observed_exact_control_variant_count'],
                'sampling':'seeded empirical draw of observed valid-row amounts, followed by legal fresh top-16 routing'},
            'physical_aliases':{phase+'.'+name:meta['alias'] for phase in ('inputs','outputs') for name,meta in record[phase].items() if meta is not None}})
    for rank,native in rank_manifests.items():
        required={key for key,schema in native['case_schemas'].items() if schema['family']==FAMILY}
        if keys_by_rank[rank]!=required:raise ValueError('A complete-workload stage-1 observation was omitted')
    manifest=validate_manifest({'schema_version':1,'status':'FROZEN_CURRENT_CAPTURE','runtime_image':IMAGE,
        'run_id':handoff['run_id'],'native_source_sha256':NATIVE_SOURCE,'family':FAMILY,'cases':cases,
        'measurement':{'method':'cuda_graph','warmup_iterations':10,'benchmark_iterations':100,'correctness_seeds':[0,1,2],
            'refresh_inputs':'each_replay','initialize_outputs':'each_replay','validate_outputs':'each_replay','negative_controls':['no_op','wrong_output']},
        'tolerance':0.02,'scale_comparison':'exact live tiled E8M0 bytes',
        'refresh_semantics':'Fresh FP8 activation signs/token permutation, legal top-16 routes at empirically observed padded work, and packed FP4 weight sign changes; original per-group magnitudes/scales and native physical layouts remain valid.',
        'frequency_policy':'Three structural score cases only; numerical representatives and generated validation draws do not duplicate workload counts.',
        'native_captured_golden_parity':'mandatory CPU fixture parity for candidate and independent private reference before generated trials',
        'qualification':'capture sealed; GPU and framework task qualification remain required'})
    external={'schema':'trusted-external-fixtures-v1','case_manifest_fingerprint':fingerprint(manifest),
        'runtime_image':IMAGE,'oci_prefix':oci_prefix,'assets':sorted(assets.values(),key=lambda row:row['path'])}
    if not oci_prefix.startswith('oci:') or not oci_prefix.endswith('/'):raise ValueError('An explicit planned/published OCI prefix ending in / is required')
    (ROOT/'fixtures').mkdir(exist_ok=True);(ROOT/'provenance').mkdir(exist_ok=True)
    (ROOT/'cases.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'fixtures/EXTERNAL-MANIFEST.json').write_text(json.dumps(external,indent=2)+'\n')
    provenance={'schema':'kimi-stage1-capture-seal-v1','run_id':handoff['run_id'],'owner_handoff_sha256':file_sha(handoff_path),
        'independent_audit_sha256':file_sha(audit_path),'rank_manifests':rank_refs,'structural_classes':len(cases),
        'all_rank_total_calls':sum(row['occurrences'] for row in cases),'asset_count':len(assets),
        'asset_bytes':sum(row['bytes'] for row in assets.values()),'gpu_qualified':False}
    (ROOT/'provenance/STAGE1-CAPTURE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if materialize:
        for asset in assets.values():
            source=capture_root/asset['object_key'];target=ROOT/asset['path'];target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if file_sha(target)!=asset['sha256']:raise ValueError('Existing staged fixture differs; refusing overwrite')
                continue
            subprocess.run(['rclone','copyto',str(source),str(target),'--transfers','64000','--progress','--buffer-size','0','--config','/dev/null'],check=True,timeout=300)
            if file_sha(target)!=asset['sha256']:raise ValueError('Materialized fixture hash differs')
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff',required=True);parser.add_argument('--capture-root',required=True)
    parser.add_argument('--oci-prefix',required=True);parser.add_argument('--materialize-local',action='store_true')
    args=parser.parse_args();print(json.dumps(seal(args.handoff,args.capture_root,args.oci_prefix,args.materialize_local),indent=2))
if __name__=='__main__':main()
