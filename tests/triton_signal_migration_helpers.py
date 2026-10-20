"""Exact normalization of retained FLA/SSD timed-input revisions for old pins."""
import ast
from copy import deepcopy
import json


SCALES = {
    'triton_fla_fused_recurrent': {'q': 8, 'v': 8, 'h0': 8},
    'triton_ssd_chunk_scan': {'cb': 32},
}

FLA_GEOMETRY_CONTROLS = [
    {'test_case_id': 'control_tail_no_initial_state', 'case_index': 30000,
     'shape': [2, 3, 3, 17, 35], 'initial_state': False, 'scale': 0.5},
    {'test_case_id': 'control_tail_initial_state', 'case_index': 30001,
     'shape': [2, 3, 3, 17, 35], 'initial_state': True, 'scale': 0.5},
    {'test_case_id': 'control_decode_multitile', 'case_index': 30002,
     'shape': [2, 1, 4, 64, 64], 'initial_state': True, 'scale': 0.5},
]


def without_fla_geometry(data):
    """Remove only the three exact reviewed additions before old hash checks."""
    data = deepcopy(data)
    assert json.dumps(data.pop('geometry_controls'), sort_keys=True) == json.dumps(FLA_GEOMETRY_CONTROLS, sort_keys=True)
    rows = [{'test_case_id': case['test_case_id'], 'dtype': 'float32',
             'params': {'case_index': case['case_index'], 'configuration': case['shape'],
                        'configuration_names': '(B,T,H,K,V)',
                        'initial_state': case['initial_state'], 'scale': case['scale']},
             'checks': ['correctness'], 'status': 'PASS'} for case in FLA_GEOMETRY_CONTROLS]
    assert json.dumps(data['cases'][-3:], sort_keys=True) == json.dumps(rows, sort_keys=True)
    data['cases'] = data['cases'][:-3]
    return data


def historical_manifest(data, task_name):
    if task_name not in SCALES:
        return deepcopy(data)
    data = deepcopy(data)
    if task_name == 'triton_fla_fused_recurrent':
        data = without_fla_geometry(data)
    assert data.pop('input_distribution') == {'name': 'strong_signal_v2', 'operand_scales': SCALES[task_name]}
    expected = []
    extra_rows = []
    for index, configuration in enumerate(data['input_table']):
        old = data['cases'][index]
        assert old['params'].pop('input_distribution') == 'strong_signal_v2'
        variants = [(None, configuration)] if task_name != 'triton_ssd_chunk_scan' else [('correctness', 42 + index), ('performance', 0)]
        for kind, seed in variants:
            case = {'test_case_id': f'historical{index+1}' if kind is None else f'historical-{kind}{index+1}',
                    'case_index': 20000 + len(expected), 'seed': seed}
            if kind is not None:
                case['shape_index'] = index
            expected.append(case)
            extra_rows.append({'test_case_id': case['test_case_id'], 'dtype': old['dtype'],
                'params': {**{key: value for key, value in case.items() if key != 'test_case_id'},
                           'configuration': configuration,
                           'configuration_names': old['params']['configuration_names'],
                           'input_distribution': 'historical_v1'},
                'checks': ['correctness'], 'status': 'PASS'})
    assert data.pop('historical_cases') == expected
    assert data['cases'][-len(extra_rows):] == extra_rows
    data['cases'] = data['cases'][:-len(extra_rows)]
    return data


def historical_generator(node, scales):
    """Remove only the exact terminal in-place scaling; retain every RNG byte in AST."""
    node = deepcopy(node)
    assert [arg.arg for arg in node.args.kwonlyargs] == ['historical']
    assert [value.value for value in node.args.kw_defaults] == [False]
    node.args.kwonlyargs = []
    node.args.kw_defaults = []
    block = node.body[-2]
    expected = ast.parse('if not historical:\n' + ''.join(f'    {name}.mul_({factor})\n' for name, factor in scales.items())).body[0]
    assert ast.dump(block) == ast.dump(expected)
    node.body.pop(-2)
    return node


def original_ssd_draw(source):
    """Extract the original performance draw without importing a GPU module."""
    tree = ast.parse(source)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_performance')
    loop = next(node for node in fn.body if isinstance(node, ast.For))
    block = next(node for node in loop.body if isinstance(node, ast.Try)).body
    statements = []
    for statement in block:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'readonly' for target in statement.targets):
            break
        statements.append(deepcopy(statement))
    assert len(statements) == 11
    seed = next(node for statement in statements for node in ast.walk(statement)
                if isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch.manual_seed')
    assert ast.literal_eval(seed.args[0]) == 0
    seed.args = [ast.Name(id='seed', ctx=ast.Load())]
    header = ast.parse('def draw(shape, seed, device):\n    import torch\n    seqlen,nheads,headdim,ngroups,dstate,chunk_size=shape\n').body[0]
    header.body.extend(statements)
    header.body.extend(ast.parse('return dict(cb=cb,x=x,C=C,dt=dt,dA_cumsum=dA_cumsum,states=states,seq_idx=seq_idx,cu_chunk_seqlens=cu,out=out)').body)
    module = ast.fix_missing_locations(ast.Module(body=[header], type_ignores=[]))
    namespace = {}
    exec(compile(module, '<original SSD performance inputs>', 'exec'), namespace)
    return namespace['draw']
