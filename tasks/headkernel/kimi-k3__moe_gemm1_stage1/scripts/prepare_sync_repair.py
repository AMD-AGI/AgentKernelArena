"""Prepare a separate experimental task with a conservative async-LDS wait repair."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
RELATIVE='source/flydsl/kernels/mixed_moe_gemm_2stage_common.py'
BEFORE='rocdl.s_waitcnt(body_vmcnt_before_barrier)'
AFTER='rocdl.s_waitcnt(0)'


def prepare(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):raise ValueError('A new destination outside the source task is required')
    cases=json.loads((source/'cases.json').read_text())
    if cases.get('family')!='flydsl_moe_stage1':raise ValueError('This repair experiment is specific to Kimi stage1')
    original=(source/RELATIVE).read_text()
    if original.count(BEFORE)!=1:raise ValueError('Pinned stage1 partial-wait site changed; review required')
    for path in source.rglob('*'):
        if path.is_symlink():raise ValueError('The source task must contain only regular files/directories')
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns('build','__pycache__','.git','.validator*','validation_report.yaml'))
    changed=original.replace(BEFORE,AFTER)
    original_hash=hashlib.sha256(original.encode()).hexdigest();repaired_hash=hashlib.sha256(changed.encode()).hexdigest()
    policy=json.loads((source/'ut/source_guard_policy.json').read_text())
    runtime_reference=(Path(policy.get('native_reference_root','ut/baseline_src/flydsl'))
                       /Path(RELATIVE).relative_to('source/flydsl')).as_posix()
    paths=list(dict.fromkeys([RELATIVE,runtime_reference,policy['sources'][RELATIVE]['reference']]))
    for relative in paths:
        path=destination/relative
        if path.read_text()!=original:raise ValueError('Candidate/reference source copies differ before repair')
        path.write_text(changed)
    provenance_path=destination/'SOURCE-PROVENANCE.json'
    provenance=json.loads(provenance_path.read_text())
    entries=[entry for entry in provenance['files'] if entry['file']==RELATIVE]
    if len(entries)!=1 or entries[0]['sha256']!=original_hash:raise ValueError('Source provenance does not bind the partial-wait source')
    entries[0]['sha256']=repaired_hash;entries[0]['only_import_scope_changed']=False
    entries[0]['correctness_repair']='Drain VMEM/LDS completion before cross-wave ping-pong buffer reuse'
    provenance['math_AST_unchanged_except_importfrom_scope']=False
    provenance['synchronization_repair_pending_gpu_validation']=True
    provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')
    repair={'schema':'kimi-stage1-async-lds-repair-experiment-v1','status':'GPU validation pending',
        'qualified':False,'unmodified_production_baseline':False,
        'source_file':RELATIVE,'source_sha256_before':original_hash,'source_sha256_after':repaired_hash,
        'change':BEFORE+' -> '+AFTER,
        'reason':'Identical-input native outputs vary in aligned 4-row by 16-column fragments and violate the unchanged 0.02 tolerance after dequantization. The partial VMEM wait before an inline workgroup barrier is the targeted async-LDS race candidate.',
        'math_tolerances_shapes_counts_and_fixtures_changed':False,
        'candidate_and_frozen_reference_repaired_equally':True,
        'required_checks':['repeated failing-seed diagnostic','all three cases with original exact-scale and 0.02 checks','submitted no-op/wrong-output rejection','fresh framework qualification'],
        'cache_requirement':'Use a new empty writable FLYDSL_RUNTIME_CACHE_DIR for the repair experiment; leave native AITER module lookup intact.'}
    (destination/'provenance/STAGE1-SYNC-REPAIR.json').write_text(json.dumps(repair,indent=2)+'\n')
    config=destination/'config.yaml';text=config.read_text().replace('  source_seed: stock\n','  source_seed: correctness_repair_experiment\n  native_baseline_variant: conservative_async_lds_wait_repair_pending_gpu_validation\n')
    config.write_text(text)
    readme=destination/'README.md';readme.write_text('**Experimental synchronization repair: GPU validation pending.** This copy changes the candidate and frozen reference equally; it is not an unmodified production baseline. See `provenance/STAGE1-SYNC-REPAIR.json`.\n\n'+readme.read_text())
    sys.path.insert(0,str(source/'ut'))
    from source_guard import validate_sources
    validate_sources(destination,source)
    validate_sources(destination,destination)
    return repair


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-task',default=str(ROOT));parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.source_task,args.output),indent=2))
