"""Decode saved FP8/E8M0 arrays on CPU; never change the task oracle or run a GPU."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path


def decode_fp8(byte):
    exponent=(byte>>3)&15;mantissa=byte&7;sign=-1 if byte&128 else 1
    if exponent==15 and mantissa==7:return float('nan')
    return sign*(mantissa/512 if exponent==0 else (1+mantissa/8)*2**(exponent-7))


def analyze(directory):
    root=Path(directory);report=json.loads((root/'DIAGNOSTIC.json').read_text())
    def read(item):
        path=root/item['blob']
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe diagnostic blob')
        data=path.read_bytes()
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Diagnostic blob changed')
        return data
    def indices(item):
        if item['dtype']!='torch.int64':raise ValueError('Expected int64 physical indices')
        values=array('q');values.frombytes(read(item));return values
    table=[decode_fp8(byte) for byte in range(256)];results=[]
    for trial in report['trials']:
        actual=read(trial['actual']['return']);reference=read(trial['reference']['return'])
        actual_scale=read(trial['actual']['return_scale']);reference_scale=read(trial['reference']['return_scale'])
        rows=indices(trial['live_sorted_rows']);payload=indices(trial['live_payload_rows']);offsets=indices(trial['live_scale_offsets'])
        if len(offsets)!=len(rows)*12 or len(rows)!=len(payload):raise ValueError('Live mask dimensions differ')
        groups=[];differences=[];reference_square_sum=0
        for index,offset in enumerate(offsets):
            route,column=divmod(index,12);start=payload[route]*384+column*32
            sa,sb=actual_scale[offset],reference_scale[offset]
            if sa==255 or sb==255:raise ValueError('Nonfinite E8M0 scale in saved live outputs')
            a=[table[value]*2**(sa-127) for value in actual[start:start+32]]
            b=[table[value]*2**(sb-127) for value in reference[start:start+32]]
            if not all(math.isfinite(value) for value in a+b):raise ValueError('Nonfinite live FP8 output')
            deltas=[abs(x-y) for x,y in zip(a,b)]
            reference_square_sum+=sum(value*value for value in b)
            differences.extend(zip(deltas,b))
            if any(deltas):
                changed=[i for i,value in enumerate(deltas) if value]
                groups.append({'sorted_row':rows[route],'payload_row':payload[route],'scale_column':column,
                    'scale_actual':sa,'scale_reference':sb,'max_abs_dequantized_difference':max(deltas),
                    'changed_columns_within_group':changed,
                    'amax_actual':max(map(abs,a)),'amax_reference':max(map(abs,b))})
        rms=math.sqrt(reference_square_sum/len(differences));atol=.02*max(rms,1e-6)
        violations=sum(delta>atol+.02*abs(value) for delta,value in differences)
        ratio=max(delta/(atol+.02*abs(value)) for delta,value in differences)
        results.append({'trial':trial['trial'],'same_input_truth':trial['identical_input_truth_to_first_trial'],
            'candidate_inputs_unchanged':trial['candidate_input_bytes_match_truth_after_call'],
            'reference_inputs_unchanged':trial['reference_input_bytes_match_truth_after_call'],
            'attributes_identical':trial['tensor_attributes_equal'],
            'scale_mismatches':trial['live_scale_mismatches'],'different_semantic_groups':len(groups),
            'dequantized_reference_rms':rms,'original_0_02_tolerance_violations':violations,
            'maximum_error_over_original_tolerance':ratio,
            'max_abs_dequantized_difference':max(delta for delta,_ in differences),'groups':groups})
    return {'schema':'kimi-stage1-independent-semantic-diagnostic-v1','diagnostic_sha256':hashlib.sha256((root/'DIAGNOSTIC.json').read_bytes()).hexdigest(),
        'seed':report['seed'],'valid_rows':report['forced_valid_rows'],
        'method':'exact CPU E4M3FN decoding times E8M0 powers of two; unchanged0.02 mixed RMS threshold',
        'conclusion':'not representation-only rounding' if any(row['original_0_02_tolerance_violations'] for row in results) else 'requires further boundary analysis',
        'GPU_actions':False,'task_oracle_changed':False,'trials':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--diagnostic',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=analyze(args.diagnostic)
    with Path(args.output).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'conclusion':result['conclusion'],'violations':[row['original_0_02_tolerance_violations'] for row in result['trials']]},indent=2))
