"""Prepare source-only no-op/wrong-output candidates for the trusted GPU evaluator."""
import argparse
import ast
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
from source_guard import validate_sources


def create(destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    policy=json.loads((ROOT/'ut/source_guard_policy.json').read_text())
    relative='source/flydsl/kernels/mixed_moe_gemm_2stage_common.py'
    original=(ROOT/relative).read_text()
    function=next(node for node in ast.walk(ast.parse(original)) if isinstance(node,ast.FunctionDef) and node.name=='_emit_moe_gemm1')
    lines=original.splitlines(keepends=True);first=function.body[0].lineno-1;last=function.end_lineno
    body=''.join(lines[first:last]);needle='scaled_vals.append(frag_vals[i] * quant_scale)'
    if body.count(needle)!=1:raise ValueError('Pinned stage-1 FP8 store expression changed; review mutation')
    no_op=''.join(lines[:first])+(' '*function.body[0].col_offset+'return\n')+''.join(lines[last:])
    wrong=''.join(lines[:first])+body.replace(needle,'scaled_vals.append(frag_vals[i] * quant_scale * 0.0)')+''.join(lines[last:])
    for name,source in [('no_op',no_op),('wrong_output',wrong)]:
        candidate=destination/name
        for path in policy['sources']:
            target=candidate/path;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/path).read_bytes())
        (candidate/relative).write_text(source)
        validate_sources(candidate,ROOT)
    record={'controls':['no_op','wrong_output'],'source_boundary_passed':True,
            'expected_gpu_result':'correctness rejection; no scoreable performance',
            'gpu_execution_performed':False,'wrong_output_mutation':'multiply FP8 payload values by zero; leave inputs and scales unchanged'}
    (destination/'CONTROL-MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(create(args.output),indent=2))
