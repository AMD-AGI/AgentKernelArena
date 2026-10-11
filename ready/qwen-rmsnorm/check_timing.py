#!/usr/bin/env python3
"""Validate complete task reports and apply the canonical raw-series gate.

This is a single-run quality check, not source attribution or speedup approval.
Run only after the protected correctness/performance commands have completed.
"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TASK=ROOT/'tasks/headkernel/qwen3.8-2.4t__gemma_fused_add_rmsnorm'
sys.path.insert(0,str(ROOT))
from src.benchmark_quality import assess_series,POLICY
spec=importlib.util.spec_from_file_location('protected_rmsnorm_runner',TASK/'scripts/task_runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--report',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
report=json.loads(a.report.read_text());raw=json.loads(a.raw.read_text())
assert report['status']=='ok' and report['timer']=='cuda_graph'
assert report['warmup_iterations']==10 and report['benchmark_iterations']==100
assert report['test_cases']==runner._benchmark_cases(raw)
rows=[{'case_id':r['test_case_id'],**assess_series(r['samples_ms'])} for r in report['test_cases']]
out={'status':'pass' if all(r['status']=='pass' for r in rows) else 'reject','policy':POLICY,'cases':rows,
     'report_sha256':hashlib.sha256(a.report.read_bytes()).hexdigest(),'raw_sha256':hashlib.sha256(a.raw.read_bytes()).hexdigest(),
     'raw_samples_changed':False,'accepted_gain':False,'scope':'single-run timing quality only; no source attribution or speedup approval'}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'cases':len(rows)},indent=2))
raise SystemExit(0 if out['status']=='pass' else 1)
