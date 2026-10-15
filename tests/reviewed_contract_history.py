"""Exact inverse identities for specifically reviewed task-contract revisions."""
from copy import deepcopy
import hashlib,json
from pathlib import Path

SSD_CURRENT_SHA = '1c8fa00216fc922d55de8802ee8a134a49b024394d82298c7daf15f51c8e54c3'
SSD_BASELINE_SHA = 'b1a4a426c11f3e7ccc7b7238572f70b9afdc444540dbb47c365d168b249a6058'
SSD_REVERSE_EDITS = [(22,
  29,
  ['INPUT_DISTRIBUTION = {"name": "strong_signal_v2", "operand_scales": {"cb": 32}}\n',
   'HISTORICAL_CASES = [\n',
   '    {"test_case_id": f"historical-{kind}{i + 1}", "case_index": 20000 + 2 * i + j,\n',
   '     "shape_index": i, "seed": seed}\n',
   '    for i in range(len(TEST_SHAPES))\n',
   '    for j, (kind, seed) in enumerate((("correctness", 42 + i), ("performance", 0)))\n',
   ']\n'],
  []),
 (97,
  122,
  ['def gen_inputs(shape, seed, device, *, historical=False):\n',
   '    return semantic_controls.scored_shape_inputs(shape, seed, device, historical=historical)\n',
   '\n',
   '\n',
   'def run_historical_correctness(index):\n',
   '    import torch\n',
   '    try:\n',
   '        if not 0 <= index < len(HISTORICAL_CASES):\n',
   '            raise IndexError("Invalid historical case index")\n',
   '        case = HISTORICAL_CASES[index]\n',
   "        shape = TEST_SHAPES[case['shape_index']]\n",
   "        inputs = gen_inputs(shape, case['seed'], 'cuda', historical=True)\n",
   '        readonly = InputSnapshot({key: value for key, value in inputs.items() if key != '
   "'out'})\n",
   '        load_module().chunk_scan_fwd(**inputs)\n',
   '        readonly.check()\n',
   '        torch.cuda.synchronize()\n',
   "        names = ('cb', 'x', 'dt', 'dA_cumsum', 'C', 'states', 'seq_idx')\n",
   '        expected = reference_chunk_scan(*(inputs[key] for key in names), '
   "shape[-1]).to('cuda')\n",
   "        check_outputs(inputs['out'], expected, atol=5e-2, rtol=5e-2,\n",
   '                      inputs=[entry[1] for entry in readonly.entries])\n',
   '        return True, None\n',
   '    except Exception as exc:\n',
   '        return False, f"Historical case {index}: {type(exc).__name__}: {exc}"\n',
   '\n',
   '\n'],
  []),
 (136,
  138,
  ['    if case_index is not None and case_index >= 20000:\n',
   '        return run_historical_correctness(case_index - 20000)\n'],
  []),
 (154, 155, ['            cb.mul_(32)\n'], []),
 (168,
  170,
  ['            semantic_controls.require_intra_signal(ref, '
   'semantic_controls.inter_chunk_reference(\n',
   '                x, dA_cumsum, C, states, seq_idx, chunk_size))\n'],
  []),
 (193, 194, ['            cb.mul_(32)\n'], [])]
FEEDFORWARD_KNOWN_SOURCE_GEOMETRY = {'1a0a54eff24dcfbb9717ec37515e70a65bed614661a95e4836c1fdbd6ab4237d': [(1, 1, 4, 4),
                                                                      (2, 2, 4, 4),
                                                                      (4, 4, 4, 4),
                                                                      (8, 8, 4, 4),
                                                                      (16, 16, 4, 4)],
 'aa73aead8c6d947f937e7d68a381eecc706f749a28d76383dd7c5170b5b80fe5': [(1, 3, 128, 128),
                                                                      (7, 9, 128, 128),
                                                                      (31, 33, 128, 128),
                                                                      (128, 129, 128, 128),
                                                                      (257, 255, 128, 128)]}
KDA_CURRENT_MANIFEST_SHA = {'1c26e49c792ea5a95ea4cf750c6d1b49d036270bbd93312beda8569167917da9', 'deaff28f9dd78535f090696faf6cde47f79bcd0892735bc3375b497cfc7a2175'}
KDA_DECLARATIONS = [
    {'name':'batch_heads_feature_tail_chunk64','B':2,'T':67,'H':3,'K':37,'chunk_size':64},
    {'name':'batch_heads_feature_tail_chunk32','B':2,'T':35,'H':1,'K':17,'chunk_size':32},
]

def historical_ssd_runner(text):
    """Every current byte is pinned; reverse only the reviewed exact line spans.

    The older whole-file comparison still runs after this inverse. Timed cb
    amplitudes intentionally changed in this revision; historical values remain
    unscored and the new measured-output/readonly/replay guards remain required.
    """
    assert hashlib.sha256(text.encode()).hexdigest()==SSD_CURRENT_SHA
    lines=text.splitlines(keepends=True)
    for start,end,expected,replacement in reversed(SSD_REVERSE_EDITS):
        assert lines[start:end]==expected
        lines[start:end]=replacement
    restored=''.join(lines)
    assert hashlib.sha256(restored.encode()).hexdigest()==SSD_BASELINE_SHA
    return restored

def historical_kda_manifest(text):
    assert hashlib.sha256(text.encode()).hexdigest() in KDA_CURRENT_MANIFEST_SHA
    data=json.loads(text)
    assert data.pop('semantic_control_cases')==KDA_DECLARATIONS
    controls=[{'test_case_id':'control_'+c['name'],'dtype':'float32',
       'params':{'semantic_control':c['name'],**{k:v for k,v in c.items() if k!='name'}},
       'checks':['correctness'],'status':'PASS'} for c in KDA_DECLARATIONS]
    assert data['cases'][-2:]==controls
    assert len(data['cases'])==7
    data['cases']=data['cases'][:-2]
    assert [c['test_case_id'] for c in data['cases']]==['perf1','perf2','perf3','perf4','perf5']
    assert all(c['checks']==['correctness','performance'] for c in data['cases'])
    return json.dumps(data,indent=2)+'\n'

def known_feedforward_geometry(source):
    digest=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    assert digest in FEEDFORWARD_KNOWN_SOURCE_GEOMETRY, 'Unreviewed Feedforward source identity'
    return FEEDFORWARD_KNOWN_SOURCE_GEOMETRY[digest]


# Scope-20 device boundaries are approved contract changes, not permission to
# alter task math or guards. Full current/before bytes bound every inverse.
TRITON_DEVICE_BOUNDARY_FILES = {'tasks/triton2triton/vllm/triton_apply_grammar_bitmask/README.md': {'after_sha256': '9f3170437d3dbd923e46a35aa16a893ccb411b279ed48304d2c7d65f175f1391',
                                                                     'before_sha256': '38dfc5fb94d122c45228aa5803cb1953e7b1dff899a21c1cf94d4df8cb638218',
                                                                     'reverse_edits': [(56,
                                                                                        65,
                                                                                        ['\n',
                                                                                         '\n',
                                                                                         '### Editable '
                                                                                         'device '
                                                                                         'implementation '
                                                                                         'boundary\n',
                                                                                         '\n',
                                                                                         'The editable '
                                                                                         'device functions '
                                                                                         'are '
                                                                                         '`_apply_grammar_bitmask_kernel`.\n',
                                                                                         'Their bodies and '
                                                                                         'additional '
                                                                                         'implementation '
                                                                                         'helpers may be '
                                                                                         'optimized. '
                                                                                         'Module\n',
                                                                                         'imports, '
                                                                                         'initialization, '
                                                                                         'host dispatch and '
                                                                                         'pointer '
                                                                                         'preparation remain '
                                                                                         'protected,\n',
                                                                                         'along with the '
                                                                                         'existing numerical '
                                                                                         'checks and '
                                                                                         'benchmark '
                                                                                         'protocol. The '
                                                                                         'structural\n',
                                                                                         'guard enforces '
                                                                                         'this boundary; it '
                                                                                         'does not isolate '
                                                                                         'arbitrary Python '
                                                                                         'execution.\n'],
                                                                                        [])]},
 'tasks/triton2triton/vllm/triton_apply_grammar_bitmask/config.yaml': {'after_sha256': 'fb54d74399b7e25da2e50e4ad8a7f6a95d0cbc8f0ea76d61b7b6b10567d5028e',
                                                                       'before_sha256': '6b72732ff602cbe75aeb064f4979a3fc7f0c13fa0bdbe7acb8fd04d9d2a85a26',
                                                                       'reverse_edits': [(6,
                                                                                          11,
                                                                                          ['  - path: '
                                                                                           'source/triton_apply_grammar_bitmask.py\n',
                                                                                           '    scope: '
                                                                                           'symbols\n',
                                                                                           '    symbols:\n',
                                                                                           '    - '
                                                                                           '_apply_grammar_bitmask_kernel\n',
                                                                                           '    '
                                                                                           'allow_new_helpers: '
                                                                                           'true\n'],
                                                                                          ['  - '
                                                                                           'source/triton_apply_grammar_bitmask.py\n'])]},
 'tasks/triton2triton/vllm/triton_awq_gemm/README.md': {'after_sha256': '32f98d3128642eec65ccb01422054f8f64d04de733d7f5500dc0fa35fbf17ae4',
                                                        'before_sha256': '9522b039a0b59837f313888ef7470015eef03c6aca0a5b8ddf2ad94b86d1cc7e',
                                                        'reverse_edits': [(51,
                                                                           60,
                                                                           ['\n',
                                                                            '\n',
                                                                            '### Editable device '
                                                                            'implementation boundary\n',
                                                                            '\n',
                                                                            'The editable device functions '
                                                                            'are `awq_gemm_kernel`.\n',
                                                                            'Their bodies and additional '
                                                                            'implementation helpers may be '
                                                                            'optimized. Module\n',
                                                                            'imports, initialization, host '
                                                                            'dispatch and pointer '
                                                                            'preparation remain protected,\n',
                                                                            'along with the existing '
                                                                            'numerical checks and benchmark '
                                                                            'protocol. The structural\n',
                                                                            'guard enforces this boundary; '
                                                                            'it does not isolate arbitrary '
                                                                            'Python execution.\n'],
                                                                           [])]},
 'tasks/triton2triton/vllm/triton_awq_gemm/config.yaml': {'after_sha256': '0de7eb1b20549c759c057d3cf3d34b664afd9337036cf3224dc0f4f58ad1819f',
                                                          'before_sha256': 'cdba9bac6e646921796eb59af68d9668af0447d58d0f8dc76a9c133331098fce',
                                                          'reverse_edits': [(6,
                                                                             11,
                                                                             ['  - path: '
                                                                              'source/triton_awq_gemm.py\n',
                                                                              '    scope: symbols\n',
                                                                              '    symbols:\n',
                                                                              '    - awq_gemm_kernel\n',
                                                                              '    allow_new_helpers: '
                                                                              'true\n'],
                                                                             ['  - '
                                                                              'source/triton_awq_gemm.py\n'])]},
 'tasks/triton2triton/vllm/triton_bmm/README.md': {'after_sha256': '37b3cf8062b43ad958556ab4ed039ac8c679271b3830365922e72a3d81383178',
                                                   'before_sha256': '688c9a2650e4e2ec1b3cf9a2e1a72c79c65ed8fe870e5e1cea5de9d4b5dceaa2',
                                                   'reverse_edits': [(41,
                                                                      50,
                                                                      ['\n',
                                                                       '\n',
                                                                       '### Editable device implementation '
                                                                       'boundary\n',
                                                                       '\n',
                                                                       'The editable device functions are '
                                                                       '`bmm_kernel`.\n',
                                                                       'Their bodies and additional '
                                                                       'implementation helpers may be '
                                                                       'optimized. Module\n',
                                                                       'imports, initialization, host '
                                                                       'dispatch and pointer preparation '
                                                                       'remain protected,\n',
                                                                       'along with the existing numerical '
                                                                       'checks and benchmark protocol. The '
                                                                       'structural\n',
                                                                       'guard enforces this boundary; it '
                                                                       'does not isolate arbitrary Python '
                                                                       'execution.\n'],
                                                                      [])]},
 'tasks/triton2triton/vllm/triton_bmm/config.yaml': {'after_sha256': '6e04236bbc34f35a60c1dcaaa47f218ac2e42d75a4cf85778240feaa8088462e',
                                                     'before_sha256': 'c7315e3dbb45c3e4e40f2d713e67b8071a14fd1c77d1f600bc047174b21d48c2',
                                                     'reverse_edits': [(6,
                                                                        11,
                                                                        ['  - path: source/triton_bmm.py\n',
                                                                         '    scope: symbols\n',
                                                                         '    symbols:\n',
                                                                         '    - bmm_kernel\n',
                                                                         '    allow_new_helpers: true\n'],
                                                                        ['  - source/triton_bmm.py\n'])]},
 'tasks/triton2triton/vllm/triton_causal_conv1d_update/README.md': {'after_sha256': 'e01416edaebd6602a48872b9fbfdaea252a393896daed2890ed5953ba1f802f1',
                                                                    'before_sha256': 'c9bfde244528d690541c3e7027e283eaeafcc480f911cb6ed815cc88f1b96c42',
                                                                    'reverse_edits': [(54,
                                                                                       63,
                                                                                       ['\n',
                                                                                        '\n',
                                                                                        '### Editable device '
                                                                                        'implementation '
                                                                                        'boundary\n',
                                                                                        '\n',
                                                                                        'The editable device '
                                                                                        'functions are '
                                                                                        '`_causal_conv1d_update_kernel`.\n',
                                                                                        'Their bodies and '
                                                                                        'additional '
                                                                                        'implementation '
                                                                                        'helpers may be '
                                                                                        'optimized. Module\n',
                                                                                        'imports, '
                                                                                        'initialization, '
                                                                                        'host dispatch and '
                                                                                        'pointer preparation '
                                                                                        'remain protected,\n',
                                                                                        'along with the '
                                                                                        'existing numerical '
                                                                                        'checks and '
                                                                                        'benchmark protocol. '
                                                                                        'The structural\n',
                                                                                        'guard enforces this '
                                                                                        'boundary; it does '
                                                                                        'not isolate '
                                                                                        'arbitrary Python '
                                                                                        'execution.\n'],
                                                                                       [])]},
 'tasks/triton2triton/vllm/triton_causal_conv1d_update/config.yaml': {'after_sha256': '7baf927cf33928b34680f96aed96049cb846b5af01d35a60f1bc691a577eddb5',
                                                                      'before_sha256': '104ccd038ca4820dfa7cffc67f4f24875c8e3bfd8a6d2ebc882baf6ac1022b1d',
                                                                      'reverse_edits': [(6,
                                                                                         11,
                                                                                         ['  - path: '
                                                                                          'source/triton_causal_conv1d_update.py\n',
                                                                                          '    scope: '
                                                                                          'symbols\n',
                                                                                          '    symbols:\n',
                                                                                          '    - '
                                                                                          '_causal_conv1d_update_kernel\n',
                                                                                          '    '
                                                                                          'allow_new_helpers: '
                                                                                          'true\n'],
                                                                                         ['  - '
                                                                                          'source/triton_causal_conv1d_update.py\n'])]},
 'tasks/triton2triton/vllm/triton_chunked_prefill_paged_decode/README.md': {'after_sha256': 'f9680f7f601eba320655fd3874d3e340b1ac8ee45c057f8b8e6430f14a4fc071',
                                                                            'before_sha256': 'ee4c03b2a1cf9116985bb9f14bc12e978bf1c09cf841cd07be68e82fa3d33b0f',
                                                                            'reverse_edits': [(82,
                                                                                               91,
                                                                                               ['\n',
                                                                                                '\n',
                                                                                                '### '
                                                                                                'Editable '
                                                                                                'device '
                                                                                                'implementation '
                                                                                                'boundary\n',
                                                                                                '\n',
                                                                                                'The '
                                                                                                'editable '
                                                                                                'device '
                                                                                                'functions '
                                                                                                'are '
                                                                                                '`cdiv_fn`, '
                                                                                                '`kernel_paged_attention_2d`.\n',
                                                                                                'Their '
                                                                                                'bodies and '
                                                                                                'additional '
                                                                                                'implementation '
                                                                                                'helpers may '
                                                                                                'be '
                                                                                                'optimized. '
                                                                                                'Module\n',
                                                                                                'imports, '
                                                                                                'initialization, '
                                                                                                'host '
                                                                                                'dispatch '
                                                                                                'and pointer '
                                                                                                'preparation '
                                                                                                'remain '
                                                                                                'protected,\n',
                                                                                                'along with '
                                                                                                'the '
                                                                                                'existing '
                                                                                                'numerical '
                                                                                                'checks and '
                                                                                                'benchmark '
                                                                                                'protocol. '
                                                                                                'The '
                                                                                                'structural\n',
                                                                                                'guard '
                                                                                                'enforces '
                                                                                                'this '
                                                                                                'boundary; '
                                                                                                'it does not '
                                                                                                'isolate '
                                                                                                'arbitrary '
                                                                                                'Python '
                                                                                                'execution.\n'],
                                                                                               [])]},
 'tasks/triton2triton/vllm/triton_chunked_prefill_paged_decode/config.yaml': {'after_sha256': 'acba87fb1d4ca5620a8560a0751a8dd4650df4dc353b4b6f5798c66d049f9fa7',
                                                                              'before_sha256': '9c2299ba86847d816573c626c13821ac00d26ed527a83001ef2491b04a5e8865',
                                                                              'reverse_edits': [(6,
                                                                                                 12,
                                                                                                 ['  - path: '
                                                                                                  'source/triton_chunked_prefill_paged_decode.py\n',
                                                                                                  '    '
                                                                                                  'scope: '
                                                                                                  'symbols\n',
                                                                                                  '    '
                                                                                                  'symbols:\n',
                                                                                                  '    - '
                                                                                                  'cdiv_fn\n',
                                                                                                  '    - '
                                                                                                  'kernel_paged_attention_2d\n',
                                                                                                  '    '
                                                                                                  'allow_new_helpers: '
                                                                                                  'true\n'],
                                                                                                 ['  - '
                                                                                                  'source/triton_chunked_prefill_paged_decode.py\n'])]},
 'tasks/triton2triton/vllm/triton_correct_attn_cp_out/README.md': {'after_sha256': 'dc3283ad5b4a43e8c46c26a7a20f6123854ed2f68cc8dcec88ab0fb97537757f',
                                                                   'before_sha256': 'b620b666d73be6b5346a22849ebf56639eb411b8153fff009b4c607832354232',
                                                                   'reverse_edits': [(54,
                                                                                      63,
                                                                                      ['\n',
                                                                                       '\n',
                                                                                       '### Editable device '
                                                                                       'implementation '
                                                                                       'boundary\n',
                                                                                       '\n',
                                                                                       'The editable device '
                                                                                       'functions are '
                                                                                       '`_correct_attn_cp_out_kernel`.\n',
                                                                                       'Their bodies and '
                                                                                       'additional '
                                                                                       'implementation '
                                                                                       'helpers may be '
                                                                                       'optimized. Module\n',
                                                                                       'imports, '
                                                                                       'initialization, host '
                                                                                       'dispatch and pointer '
                                                                                       'preparation remain '
                                                                                       'protected,\n',
                                                                                       'along with the '
                                                                                       'existing numerical '
                                                                                       'checks and benchmark '
                                                                                       'protocol. The '
                                                                                       'structural\n',
                                                                                       'guard enforces this '
                                                                                       'boundary; it does '
                                                                                       'not isolate '
                                                                                       'arbitrary Python '
                                                                                       'execution.\n'],
                                                                                      [])]},
 'tasks/triton2triton/vllm/triton_correct_attn_cp_out/config.yaml': {'after_sha256': '66ac82fce41b23f5d0aef546168b6523996140bf820c7b3812424b01873e4994',
                                                                     'before_sha256': 'd734d0a7b8f9ddf17111dec1ee435e81fc68865079a8d524099b7486e9b1914f',
                                                                     'reverse_edits': [(6,
                                                                                        11,
                                                                                        ['  - path: '
                                                                                         'source/triton_correct_attn_cp_out.py\n',
                                                                                         '    scope: '
                                                                                         'symbols\n',
                                                                                         '    symbols:\n',
                                                                                         '    - '
                                                                                         '_correct_attn_cp_out_kernel\n',
                                                                                         '    '
                                                                                         'allow_new_helpers: '
                                                                                         'true\n'],
                                                                                        ['  - '
                                                                                         'source/triton_correct_attn_cp_out.py\n'])]},
 'tasks/triton2triton/vllm/triton_decode_attn_grouped_stage1/README.md': {'after_sha256': '0f2a551ce51340a3709600bc78d91b0acc2d2205c10756e4dfe45d5b6d380d7d',
                                                                          'before_sha256': 'ada3cf20f1eda4171fc77a6249f9aedd11890a2741fb8967134f0ca52dd9afc6',
                                                                          'reverse_edits': [(89,
                                                                                             98,
                                                                                             ['\n',
                                                                                              '\n',
                                                                                              '### Editable '
                                                                                              'device '
                                                                                              'implementation '
                                                                                              'boundary\n',
                                                                                              '\n',
                                                                                              'The editable '
                                                                                              'device '
                                                                                              'functions are '
                                                                                              '`tanh`, '
                                                                                              '`_fwd_grouped_kernel_stage1`.\n',
                                                                                              'Their bodies '
                                                                                              'and '
                                                                                              'additional '
                                                                                              'implementation '
                                                                                              'helpers may '
                                                                                              'be optimized. '
                                                                                              'Module\n',
                                                                                              'imports, '
                                                                                              'initialization, '
                                                                                              'host dispatch '
                                                                                              'and pointer '
                                                                                              'preparation '
                                                                                              'remain '
                                                                                              'protected,\n',
                                                                                              'along with '
                                                                                              'the existing '
                                                                                              'numerical '
                                                                                              'checks and '
                                                                                              'benchmark '
                                                                                              'protocol. The '
                                                                                              'structural\n',
                                                                                              'guard '
                                                                                              'enforces this '
                                                                                              'boundary; it '
                                                                                              'does not '
                                                                                              'isolate '
                                                                                              'arbitrary '
                                                                                              'Python '
                                                                                              'execution.\n'],
                                                                                             [])]},
 'tasks/triton2triton/vllm/triton_decode_attn_grouped_stage1/config.yaml': {'after_sha256': 'd5abee78e0d739495acc39027a22d30462c42f37c1d86332403476af0c83fdc6',
                                                                            'before_sha256': '24c98bbcaff663b18709845a25e03d1608cf50e6068e65caacace55e37e47dbd',
                                                                            'reverse_edits': [(6,
                                                                                               12,
                                                                                               ['  - path: '
                                                                                                'source/_fwd_grouped_kernel_stage1.py\n',
                                                                                                '    scope: '
                                                                                                'symbols\n',
                                                                                                '    '
                                                                                                'symbols:\n',
                                                                                                '    - '
                                                                                                'tanh\n',
                                                                                                '    - '
                                                                                                '_fwd_grouped_kernel_stage1\n',
                                                                                                '    '
                                                                                                'allow_new_helpers: '
                                                                                                'true\n'],
                                                                                               ['  - '
                                                                                                'source/_fwd_grouped_kernel_stage1.py\n'])]},
 'tasks/triton2triton/vllm/triton_decode_attn_stage1/README.md': {'after_sha256': 'd9f166fa5bb3c0b8373ded99c03069135faacd3410b89ccfa9de91f0dd69c792',
                                                                  'before_sha256': '9e164b707e474647d6d64bd32612a5961b9cd4a405982c671cb320e6195f7578',
                                                                  'reverse_edits': [(88,
                                                                                     97,
                                                                                     ['\n',
                                                                                      '\n',
                                                                                      '### Editable device '
                                                                                      'implementation '
                                                                                      'boundary\n',
                                                                                      '\n',
                                                                                      'The editable device '
                                                                                      'functions are `tanh`, '
                                                                                      '`_fwd_kernel_stage1`.\n',
                                                                                      'Their bodies and '
                                                                                      'additional '
                                                                                      'implementation '
                                                                                      'helpers may be '
                                                                                      'optimized. Module\n',
                                                                                      'imports, '
                                                                                      'initialization, host '
                                                                                      'dispatch and pointer '
                                                                                      'preparation remain '
                                                                                      'protected,\n',
                                                                                      'along with the '
                                                                                      'existing numerical '
                                                                                      'checks and benchmark '
                                                                                      'protocol. The '
                                                                                      'structural\n',
                                                                                      'guard enforces this '
                                                                                      'boundary; it does not '
                                                                                      'isolate arbitrary '
                                                                                      'Python execution.\n'],
                                                                                     [])]},
 'tasks/triton2triton/vllm/triton_decode_attn_stage1/config.yaml': {'after_sha256': 'c69bd6c3c573230c1358f6fb000a3ea82a56510fd921d5b5209e454fee63f684',
                                                                    'before_sha256': '793e6e10b6227e67ce8d50ca18f88ca87deb888631af904c5a4998bfea6768ba',
                                                                    'reverse_edits': [(6,
                                                                                       12,
                                                                                       ['  - path: '
                                                                                        'source/_fwd_kernel_stage1.py\n',
                                                                                        '    scope: '
                                                                                        'symbols\n',
                                                                                        '    symbols:\n',
                                                                                        '    - tanh\n',
                                                                                        '    - '
                                                                                        '_fwd_kernel_stage1\n',
                                                                                        '    '
                                                                                        'allow_new_helpers: '
                                                                                        'true\n'],
                                                                                       ['  - '
                                                                                        'source/_fwd_kernel_stage1.py\n'])]},
 'tasks/triton2triton/vllm/triton_decode_attn_stage2/README.md': {'after_sha256': 'ec94272c221580401c332d01eefd616789bc972d7cb42e0be7dc72c1aa13309e',
                                                                  'before_sha256': '02b7883d55eaffb47865a326df31a2420408a8e395ad35cd4c43da20ea9ef6fc',
                                                                  'reverse_edits': [(79,
                                                                                     88,
                                                                                     ['\n',
                                                                                      '\n',
                                                                                      '### Editable device '
                                                                                      'implementation '
                                                                                      'boundary\n',
                                                                                      '\n',
                                                                                      'The editable device '
                                                                                      'functions are '
                                                                                      '`_fwd_kernel_stage2`.\n',
                                                                                      'Their bodies and '
                                                                                      'additional '
                                                                                      'implementation '
                                                                                      'helpers may be '
                                                                                      'optimized. Module\n',
                                                                                      'imports, '
                                                                                      'initialization, host '
                                                                                      'dispatch and pointer '
                                                                                      'preparation remain '
                                                                                      'protected,\n',
                                                                                      'along with the '
                                                                                      'existing numerical '
                                                                                      'checks and benchmark '
                                                                                      'protocol. The '
                                                                                      'structural\n',
                                                                                      'guard enforces this '
                                                                                      'boundary; it does not '
                                                                                      'isolate arbitrary '
                                                                                      'Python execution.\n'],
                                                                                     [])]},
 'tasks/triton2triton/vllm/triton_decode_attn_stage2/config.yaml': {'after_sha256': 'b3810ddb5da0e7632f526c54399db2023ac6ae54a25801ff3d13dfa7fd5047c9',
                                                                    'before_sha256': '7e1e7018cd00382729b75e71be8f2c211c146e957381472ddcae0d61ae7e6866',
                                                                    'reverse_edits': [(6,
                                                                                       11,
                                                                                       ['  - path: '
                                                                                        'source/_fwd_kernel_stage2.py\n',
                                                                                        '    scope: '
                                                                                        'symbols\n',
                                                                                        '    symbols:\n',
                                                                                        '    - '
                                                                                        '_fwd_kernel_stage2\n',
                                                                                        '    '
                                                                                        'allow_new_helpers: '
                                                                                        'true\n'],
                                                                                       ['  - '
                                                                                        'source/_fwd_kernel_stage2.py\n'])]},
 'tasks/triton2triton/vllm/triton_ep_gather/README.md': {'after_sha256': '95e35778905788a75c2f1fc286138133de24a4fb7c9daf764bf3fcbc4cce721d',
                                                         'before_sha256': '7fa777a1203ec1836570002443f93c14973a2362ee0f16652dc24dbf466b6cf7',
                                                         'reverse_edits': [(44,
                                                                            53,
                                                                            ['\n',
                                                                             '\n',
                                                                             '### Editable device '
                                                                             'implementation boundary\n',
                                                                             '\n',
                                                                             'The editable device functions '
                                                                             'are `_fwd_kernel_ep_gather`.\n',
                                                                             'Their bodies and additional '
                                                                             'implementation helpers may be '
                                                                             'optimized. Module\n',
                                                                             'imports, initialization, host '
                                                                             'dispatch and pointer '
                                                                             'preparation remain '
                                                                             'protected,\n',
                                                                             'along with the existing '
                                                                             'numerical checks and benchmark '
                                                                             'protocol. The structural\n',
                                                                             'guard enforces this boundary; '
                                                                             'it does not isolate arbitrary '
                                                                             'Python execution.\n'],
                                                                            [])]},
 'tasks/triton2triton/vllm/triton_ep_gather/config.yaml': {'after_sha256': 'a0d64d529fd939f9367280f7b714b1f7710286a3042b7e5aec6ab55f98c72bab',
                                                           'before_sha256': '1c3a174ddf8be7fe280178935745ad80ae0eb4b97c1cc8fe62ceadce6738e771',
                                                           'reverse_edits': [(6,
                                                                              11,
                                                                              ['  - path: '
                                                                               'source/triton_ep_gather.py\n',
                                                                               '    scope: symbols\n',
                                                                               '    symbols:\n',
                                                                               '    - '
                                                                               '_fwd_kernel_ep_gather\n',
                                                                               '    allow_new_helpers: '
                                                                               'true\n'],
                                                                              ['  - '
                                                                               'source/triton_ep_gather.py\n'])]},
 'tasks/triton2triton/vllm/triton_ep_scatter_1/README.md': {'after_sha256': '1f3f587569cec719ec6df7df8a1a276b9815ddcb7ade33accf5aa38326ea2996',
                                                            'before_sha256': 'b92081f80227ebd2be5086adea4e1da6794e7350ce3028eed4b92a5ff2c25693',
                                                            'reverse_edits': [(44,
                                                                               53,
                                                                               ['\n',
                                                                                '\n',
                                                                                '### Editable device '
                                                                                'implementation boundary\n',
                                                                                '\n',
                                                                                'The editable device '
                                                                                'functions are '
                                                                                '`round_up_128`, '
                                                                                '`_fwd_kernel_ep_scatter_1`.\n',
                                                                                'Their bodies and additional '
                                                                                'implementation helpers may '
                                                                                'be optimized. Module\n',
                                                                                'imports, initialization, '
                                                                                'host dispatch and pointer '
                                                                                'preparation remain '
                                                                                'protected,\n',
                                                                                'along with the existing '
                                                                                'numerical checks and '
                                                                                'benchmark protocol. The '
                                                                                'structural\n',
                                                                                'guard enforces this '
                                                                                'boundary; it does not '
                                                                                'isolate arbitrary Python '
                                                                                'execution.\n'],
                                                                               [])]},
 'tasks/triton2triton/vllm/triton_ep_scatter_1/config.yaml': {'after_sha256': 'b938b81659606fdf16fc26341d225ebebf74fa4ca06ad11e0e8560312772f805',
                                                              'before_sha256': '0b8c3d2c6db69fa55132fd26e1647845c6fe12669e5ccde2bfecdb6b2f2009f6',
                                                              'reverse_edits': [(6,
                                                                                 12,
                                                                                 ['  - path: '
                                                                                  'source/triton_ep_scatter_1.py\n',
                                                                                  '    scope: symbols\n',
                                                                                  '    symbols:\n',
                                                                                  '    - round_up_128\n',
                                                                                  '    - '
                                                                                  '_fwd_kernel_ep_scatter_1\n',
                                                                                  '    allow_new_helpers: '
                                                                                  'true\n'],
                                                                                 ['  - '
                                                                                  'source/triton_ep_scatter_1.py\n'])]},
 'tasks/triton2triton/vllm/triton_expert_kernel/README.md': {'after_sha256': '363d3769876fa84773b205e3f5ff7bc7f1f0d823fcb6c3a894f72a99c623536a',
                                                             'before_sha256': '14d610a41846b32393589aa59acd137fa83854c884c626e99af86802736ef7ea',
                                                             'reverse_edits': [(47,
                                                                                56,
                                                                                ['\n',
                                                                                 '\n',
                                                                                 '### Editable device '
                                                                                 'implementation boundary\n',
                                                                                 '\n',
                                                                                 'The editable device '
                                                                                 'functions are '
                                                                                 '`expert_triton_kernel`.\n',
                                                                                 'Their bodies and '
                                                                                 'additional implementation '
                                                                                 'helpers may be optimized. '
                                                                                 'Module\n',
                                                                                 'imports, initialization, '
                                                                                 'host dispatch and pointer '
                                                                                 'preparation remain '
                                                                                 'protected,\n',
                                                                                 'along with the existing '
                                                                                 'numerical checks and '
                                                                                 'benchmark protocol. The '
                                                                                 'structural\n',
                                                                                 'guard enforces this '
                                                                                 'boundary; it does not '
                                                                                 'isolate arbitrary Python '
                                                                                 'execution.\n'],
                                                                                [])]},
 'tasks/triton2triton/vllm/triton_expert_kernel/config.yaml': {'after_sha256': 'c65abb1bc437b26f4a62e68d6aa19b430f4bea874db35ecfc9f2e7aa84ac4ea7',
                                                               'before_sha256': '713b77fabbe6bd2775f79b76fc497f06d20093988ddc5438d52aa452c8efee4f',
                                                               'reverse_edits': [(6,
                                                                                  11,
                                                                                  ['  - path: '
                                                                                   'source/triton_expert_kernel.py\n',
                                                                                   '    scope: symbols\n',
                                                                                   '    symbols:\n',
                                                                                   '    - '
                                                                                   'expert_triton_kernel\n',
                                                                                   '    allow_new_helpers: '
                                                                                   'true\n'],
                                                                                  ['  - '
                                                                                   'source/triton_expert_kernel.py\n'])]},
 'tasks/triton2triton/vllm/triton_fla_cumsum_scalar/README.md': {'after_sha256': '6d4aa1f9463abe19a5de42e29635ebf10def1284e5b6992e753965daa6e65055',
                                                                 'before_sha256': '768a3276cc87535c61847e5c788ac4aa294e3ed38e2944a93d090ed91f79d12e',
                                                                 'reverse_edits': [(45,
                                                                                    54,
                                                                                    ['\n',
                                                                                     '\n',
                                                                                     '### Editable device '
                                                                                     'implementation '
                                                                                     'boundary\n',
                                                                                     '\n',
                                                                                     'The editable device '
                                                                                     'functions are '
                                                                                     '`chunk_local_cumsum_scalar_kernel`.\n',
                                                                                     'Their bodies and '
                                                                                     'additional '
                                                                                     'implementation helpers '
                                                                                     'may be optimized. '
                                                                                     'Module\n',
                                                                                     'imports, '
                                                                                     'initialization, host '
                                                                                     'dispatch and pointer '
                                                                                     'preparation remain '
                                                                                     'protected,\n',
                                                                                     'along with the '
                                                                                     'existing numerical '
                                                                                     'checks and benchmark '
                                                                                     'protocol. The '
                                                                                     'structural\n',
                                                                                     'guard enforces this '
                                                                                     'boundary; it does not '
                                                                                     'isolate arbitrary '
                                                                                     'Python execution.\n'],
                                                                                    [])]},
 'tasks/triton2triton/vllm/triton_fla_cumsum_scalar/config.yaml': {'after_sha256': 'dd352194470d6bb09e72ba256bdd8ddda5b2ff2337f1a63049824b5d189bea52',
                                                                   'before_sha256': '70d159272b991575cfebbe5dd0c9691153523882aa3d81295282e5c37b1e9db8',
                                                                   'reverse_edits': [(6,
                                                                                      11,
                                                                                      ['  - path: '
                                                                                       'source/triton_fla_cumsum_scalar.py\n',
                                                                                       '    scope: symbols\n',
                                                                                       '    symbols:\n',
                                                                                       '    - '
                                                                                       'chunk_local_cumsum_scalar_kernel\n',
                                                                                       '    '
                                                                                       'allow_new_helpers: '
                                                                                       'true\n'],
                                                                                      ['  - '
                                                                                       'source/triton_fla_cumsum_scalar.py\n'])]},
 'tasks/triton2triton/vllm/triton_fla_cumsum_vector/README.md': {'after_sha256': 'c4f4a3d533d49e0d32abbdd3fd4f9c1ecc8641371da10c4fe4fdf4b0c351c5fd',
                                                                 'before_sha256': 'c0389d8bf0a67ef033edf35a528ff04d7aee3c2a26765fd13a6ea6af94de158e',
                                                                 'reverse_edits': [(45,
                                                                                    54,
                                                                                    ['\n',
                                                                                     '\n',
                                                                                     '### Editable device '
                                                                                     'implementation '
                                                                                     'boundary\n',
                                                                                     '\n',
                                                                                     'The editable device '
                                                                                     'functions are '
                                                                                     '`chunk_local_cumsum_vector_kernel`.\n',
                                                                                     'Their bodies and '
                                                                                     'additional '
                                                                                     'implementation helpers '
                                                                                     'may be optimized. '
                                                                                     'Module\n',
                                                                                     'imports, '
                                                                                     'initialization, host '
                                                                                     'dispatch and pointer '
                                                                                     'preparation remain '
                                                                                     'protected,\n',
                                                                                     'along with the '
                                                                                     'existing numerical '
                                                                                     'checks and benchmark '
                                                                                     'protocol. The '
                                                                                     'structural\n',
                                                                                     'guard enforces this '
                                                                                     'boundary; it does not '
                                                                                     'isolate arbitrary '
                                                                                     'Python execution.\n'],
                                                                                    [])]},
 'tasks/triton2triton/vllm/triton_fla_cumsum_vector/config.yaml': {'after_sha256': 'd1f55ec961bbe5bda4d98131e20fcf95f387ad4650ad4d191f95895fbfa4789b',
                                                                   'before_sha256': '125e25621fe54e1b123e0cc3fdfcb12ebcf0e16cd262b187d197c2d36849ec2b',
                                                                   'reverse_edits': [(6,
                                                                                      11,
                                                                                      ['  - path: '
                                                                                       'source/triton_fla_cumsum_vector.py\n',
                                                                                       '    scope: symbols\n',
                                                                                       '    symbols:\n',
                                                                                       '    - '
                                                                                       'chunk_local_cumsum_vector_kernel\n',
                                                                                       '    '
                                                                                       'allow_new_helpers: '
                                                                                       'true\n'],
                                                                                      ['  - '
                                                                                       'source/triton_fla_cumsum_vector.py\n'])]},
 'tasks/triton2triton/vllm/triton_fla_layernorm/README.md': {'after_sha256': 'd7d057a38a4390a58817653f635aa44a1c7acca0376d2aaa181c4b0fadea3132',
                                                             'before_sha256': 'ee1f1ae727e1a49a156e4043618ad2ae7a3deac2f4cc4c6c663ad4e33b59f8f3',
                                                             'reverse_edits': [(41,
                                                                                50,
                                                                                ['\n',
                                                                                 '\n',
                                                                                 '### Editable device '
                                                                                 'implementation boundary\n',
                                                                                 '\n',
                                                                                 'The editable device '
                                                                                 'functions are '
                                                                                 '`layer_norm_fwd_kernel`.\n',
                                                                                 'Their bodies and '
                                                                                 'additional implementation '
                                                                                 'helpers may be optimized. '
                                                                                 'Module\n',
                                                                                 'imports, initialization, '
                                                                                 'host dispatch and pointer '
                                                                                 'preparation remain '
                                                                                 'protected,\n',
                                                                                 'along with the existing '
                                                                                 'numerical checks and '
                                                                                 'benchmark protocol. The '
                                                                                 'structural\n',
                                                                                 'guard enforces this '
                                                                                 'boundary; it does not '
                                                                                 'isolate arbitrary Python '
                                                                                 'execution.\n'],
                                                                                [])]},
 'tasks/triton2triton/vllm/triton_fla_layernorm/config.yaml': {'after_sha256': '0d27a2c8ca2b5d1edd587c771a1e93721fe0f3c199d87e9314e45262db373146',
                                                               'before_sha256': 'c86e2519f055c0271a9201c3818be96fb11bbd8163cd039e34b1836fdf348871',
                                                               'reverse_edits': [(6,
                                                                                  11,
                                                                                  ['  - path: '
                                                                                   'source/triton_fla_layernorm.py\n',
                                                                                   '    scope: symbols\n',
                                                                                   '    symbols:\n',
                                                                                   '    - '
                                                                                   'layer_norm_fwd_kernel\n',
                                                                                   '    allow_new_helpers: '
                                                                                   'true\n'],
                                                                                  ['  - '
                                                                                   'source/triton_fla_layernorm.py\n'])]},
 'tasks/triton2triton/vllm/triton_flash_prefill_attention/README.md': {'after_sha256': '5ddab47c3790ee699a4f8f03660702dde34826123a0cfec769398b44f0a5dd06',
                                                                       'before_sha256': 'cca8c62515eb874e93a9e50f5371a5a50c5edf78f8aff2f14e37bf1b5490ca32',
                                                                       'reverse_edits': [(76,
                                                                                          85,
                                                                                          ['\n',
                                                                                           '\n',
                                                                                           '### Editable '
                                                                                           'device '
                                                                                           'implementation '
                                                                                           'boundary\n',
                                                                                           '\n',
                                                                                           'The editable '
                                                                                           'device functions '
                                                                                           'are '
                                                                                           '`_fwd_kernel`.\n',
                                                                                           'Their bodies and '
                                                                                           'additional '
                                                                                           'implementation '
                                                                                           'helpers may be '
                                                                                           'optimized. '
                                                                                           'Module\n',
                                                                                           'imports, '
                                                                                           'initialization, '
                                                                                           'host dispatch '
                                                                                           'and pointer '
                                                                                           'preparation '
                                                                                           'remain '
                                                                                           'protected,\n',
                                                                                           'along with the '
                                                                                           'existing '
                                                                                           'numerical checks '
                                                                                           'and benchmark '
                                                                                           'protocol. The '
                                                                                           'structural\n',
                                                                                           'guard enforces '
                                                                                           'this boundary; '
                                                                                           'it does not '
                                                                                           'isolate '
                                                                                           'arbitrary Python '
                                                                                           'execution.\n'],
                                                                                          [])]},
 'tasks/triton2triton/vllm/triton_flash_prefill_attention/config.yaml': {'after_sha256': 'dca4c1e0037ce58f17d85d6fd5159ef850130e2c74b79afc42d57d88803995b6',
                                                                         'before_sha256': '2fbb4c96d9705a85dd42753c9a1638b67212d87639dbc53e36a072b9bee90fff',
                                                                         'reverse_edits': [(6,
                                                                                            11,
                                                                                            ['  - path: '
                                                                                             'source/triton_flash_prefill_attention.py\n',
                                                                                             '    scope: '
                                                                                             'symbols\n',
                                                                                             '    symbols:\n',
                                                                                             '    - '
                                                                                             '_fwd_kernel\n',
                                                                                             '    '
                                                                                             'allow_new_helpers: '
                                                                                             'true\n'],
                                                                                            ['  - '
                                                                                             'source/triton_flash_prefill_attention.py\n'])]},
 'tasks/triton2triton/vllm/triton_fused_gdn_gating/README.md': {'after_sha256': 'b5eefca1312024bcfa4a97c533b6db43ca3a4c1ebb0d76b6b5705cce6a67147d',
                                                                'before_sha256': '224db382310fe35792d0afcbb60d2c37e32888f7bdb1af02799693a61ed4d4b3',
                                                                'reverse_edits': [(62,
                                                                                   71,
                                                                                   ['\n',
                                                                                    '\n',
                                                                                    '### Editable device '
                                                                                    'implementation '
                                                                                    'boundary\n',
                                                                                    '\n',
                                                                                    'The editable device '
                                                                                    'functions are '
                                                                                    '`fused_gdn_gating_kernel`.\n',
                                                                                    'Their bodies and '
                                                                                    'additional '
                                                                                    'implementation helpers '
                                                                                    'may be optimized. '
                                                                                    'Module\n',
                                                                                    'imports, '
                                                                                    'initialization, host '
                                                                                    'dispatch and pointer '
                                                                                    'preparation remain '
                                                                                    'protected,\n',
                                                                                    'along with the existing '
                                                                                    'numerical checks and '
                                                                                    'benchmark protocol. The '
                                                                                    'structural\n',
                                                                                    'guard enforces this '
                                                                                    'boundary; it does not '
                                                                                    'isolate arbitrary '
                                                                                    'Python execution.\n'],
                                                                                   [])]},
 'tasks/triton2triton/vllm/triton_fused_gdn_gating/config.yaml': {'after_sha256': '00e08420c025b289d4059a9fb1cd65920dbd0122b017c3f8e9b490d5feb10245',
                                                                  'before_sha256': '171a150edf22833e18a3e6956e9e403743bff25fb2762cba3d067a244f7641db',
                                                                  'reverse_edits': [(6,
                                                                                     11,
                                                                                     ['  - path: '
                                                                                      'source/triton_fused_gdn_gating.py\n',
                                                                                      '    scope: symbols\n',
                                                                                      '    symbols:\n',
                                                                                      '    - '
                                                                                      'fused_gdn_gating_kernel\n',
                                                                                      '    '
                                                                                      'allow_new_helpers: '
                                                                                      'true\n'],
                                                                                     ['  - '
                                                                                      'source/triton_fused_gdn_gating.py\n'])]},
 'tasks/triton2triton/vllm/triton_fused_moe_lora/README.md': {'after_sha256': '783e1749a0c1f394b87b311b94789f655a9251199431fdd31e45175355a9070d',
                                                              'before_sha256': '8fec6fffc0da84e702baaa4e258f746a7b796d60749c18a7b19d8515189d1ed2',
                                                              'reverse_edits': [(84,
                                                                                 93,
                                                                                 ['\n',
                                                                                  '\n',
                                                                                  '### Editable device '
                                                                                  'implementation boundary\n',
                                                                                  '\n',
                                                                                  'The editable device '
                                                                                  'functions are '
                                                                                  '`_get_lora_id`, '
                                                                                  '`_get_expert_id`, '
                                                                                  '`_get_token_offs`, '
                                                                                  '`fused_moe_lora_kernel`.\n',
                                                                                  'Their bodies and '
                                                                                  'additional implementation '
                                                                                  'helpers may be optimized. '
                                                                                  'Module\n',
                                                                                  'imports, initialization, '
                                                                                  'host dispatch and pointer '
                                                                                  'preparation remain '
                                                                                  'protected,\n',
                                                                                  'along with the existing '
                                                                                  'numerical checks and '
                                                                                  'benchmark protocol. The '
                                                                                  'structural\n',
                                                                                  'guard enforces this '
                                                                                  'boundary; it does not '
                                                                                  'isolate arbitrary Python '
                                                                                  'execution.\n'],
                                                                                 [])]},
 'tasks/triton2triton/vllm/triton_fused_moe_lora/config.yaml': {'after_sha256': '579e981824b47adb3e515709798ec2f419dd79f543f477e15001969911ddbc48',
                                                                'before_sha256': 'd55709edbab279c8bfa2e8d2b1de6d34df4ab10d1a717c1b3b23e8d87314349b',
                                                                'reverse_edits': [(6,
                                                                                   14,
                                                                                   ['  - path: '
                                                                                    'source/triton_fused_moe_lora.py\n',
                                                                                    '    scope: symbols\n',
                                                                                    '    symbols:\n',
                                                                                    '    - _get_lora_id\n',
                                                                                    '    - _get_expert_id\n',
                                                                                    '    - _get_token_offs\n',
                                                                                    '    - '
                                                                                    'fused_moe_lora_kernel\n',
                                                                                    '    allow_new_helpers: '
                                                                                    'true\n'],
                                                                                   ['  - '
                                                                                    'source/triton_fused_moe_lora.py\n'])]},
 'tasks/triton2triton/vllm/triton_gumbel_sample/README.md': {'after_sha256': '0829c043a8b293dec4049a63a72f5e6f867001e6a09a05d20ae6fb7d7100a9cd',
                                                             'before_sha256': 'd90372720a66a694251a764f8c1ec4d66acf68eee25cf1c325f75e84e3147799',
                                                             'reverse_edits': [(46,
                                                                                55,
                                                                                ['\n',
                                                                                 '\n',
                                                                                 '### Editable device '
                                                                                 'implementation boundary\n',
                                                                                 '\n',
                                                                                 'The editable device '
                                                                                 'functions are '
                                                                                 '`_gumbel_sample_kernel`.\n',
                                                                                 'Their bodies and '
                                                                                 'additional implementation '
                                                                                 'helpers may be optimized. '
                                                                                 'Module\n',
                                                                                 'imports, initialization, '
                                                                                 'host dispatch and pointer '
                                                                                 'preparation remain '
                                                                                 'protected,\n',
                                                                                 'along with the existing '
                                                                                 'numerical checks and '
                                                                                 'benchmark protocol. The '
                                                                                 'structural\n',
                                                                                 'guard enforces this '
                                                                                 'boundary; it does not '
                                                                                 'isolate arbitrary Python '
                                                                                 'execution.\n'],
                                                                                [])]},
 'tasks/triton2triton/vllm/triton_gumbel_sample/config.yaml': {'after_sha256': '3fac26ea5bb59a212e9d1b19202ee9dadb828e8115e2ae4f85e38f3bba5c6551',
                                                               'before_sha256': '719857b33e405fc18855ed590f16daca552bd9ce9ecbfa0d1f75417a5e664c94',
                                                               'reverse_edits': [(6,
                                                                                  11,
                                                                                  ['  - path: '
                                                                                   'source/triton_gumbel_sample.py\n',
                                                                                   '    scope: symbols\n',
                                                                                   '    symbols:\n',
                                                                                   '    - '
                                                                                   '_gumbel_sample_kernel\n',
                                                                                   '    allow_new_helpers: '
                                                                                   'true\n'],
                                                                                  ['  - '
                                                                                   'source/triton_gumbel_sample.py\n'])]},
 'tasks/triton2triton/vllm/triton_lightning_attn_kv_reduce/README.md': {'after_sha256': '33a07c73e464adb2c03e3fbc57050f6182f945da06ba9f56ed520dcec42dedae',
                                                                        'before_sha256': 'e8d3dd2ff72794c0797698f506dba6ff3b0112acf0cbcb7bfe98b9378b057839',
                                                                        'reverse_edits': [(53,
                                                                                           62,
                                                                                           ['\n',
                                                                                            '\n',
                                                                                            '### Editable '
                                                                                            'device '
                                                                                            'implementation '
                                                                                            'boundary\n',
                                                                                            '\n',
                                                                                            'The editable '
                                                                                            'device '
                                                                                            'functions are '
                                                                                            '`_fwd_kv_reduce`.\n',
                                                                                            'Their bodies '
                                                                                            'and additional '
                                                                                            'implementation '
                                                                                            'helpers may be '
                                                                                            'optimized. '
                                                                                            'Module\n',
                                                                                            'imports, '
                                                                                            'initialization, '
                                                                                            'host dispatch '
                                                                                            'and pointer '
                                                                                            'preparation '
                                                                                            'remain '
                                                                                            'protected,\n',
                                                                                            'along with the '
                                                                                            'existing '
                                                                                            'numerical '
                                                                                            'checks and '
                                                                                            'benchmark '
                                                                                            'protocol. The '
                                                                                            'structural\n',
                                                                                            'guard enforces '
                                                                                            'this boundary; '
                                                                                            'it does not '
                                                                                            'isolate '
                                                                                            'arbitrary '
                                                                                            'Python '
                                                                                            'execution.\n'],
                                                                                           [])]},
 'tasks/triton2triton/vllm/triton_lightning_attn_kv_reduce/config.yaml': {'after_sha256': '56cdeefe603318820ff265adc4276a0f7a52b508322a5c9cdcdd78c523ac1cf0',
                                                                          'before_sha256': '170bdeb7d769ba61c2b3c053bab56dbb66f65d234ea109cf0e6c107fd25ec4ef',
                                                                          'reverse_edits': [(6,
                                                                                             11,
                                                                                             ['  - path: '
                                                                                              'source/triton_lightning_attn_kv_reduce.py\n',
                                                                                              '    scope: '
                                                                                              'symbols\n',
                                                                                              '    '
                                                                                              'symbols:\n',
                                                                                              '    - '
                                                                                              '_fwd_kv_reduce\n',
                                                                                              '    '
                                                                                              'allow_new_helpers: '
                                                                                              'true\n'],
                                                                                             ['  - '
                                                                                              'source/triton_lightning_attn_kv_reduce.py\n'])]},
 'tasks/triton2triton/vllm/triton_lora_expand/README.md': {'after_sha256': 'c6ea829ed8c704eedef85db31ccf6bdd153d080dfdfbc792c06d018412c4c8f1',
                                                           'before_sha256': 'e1de456de9737753bf5059cdbe746b06bce0fd10672d628e1113792d4e258f3a',
                                                           'reverse_edits': [(57,
                                                                              66,
                                                                              ['\n',
                                                                               '\n',
                                                                               '### Editable device '
                                                                               'implementation boundary\n',
                                                                               '\n',
                                                                               'The editable device '
                                                                               'functions are `mm_k`, '
                                                                               '`do_expand_kernel`, '
                                                                               '`_lora_expand_kernel`.\n',
                                                                               'Their bodies and additional '
                                                                               'implementation helpers may '
                                                                               'be optimized. Module\n',
                                                                               'imports, initialization, '
                                                                               'host dispatch and pointer '
                                                                               'preparation remain '
                                                                               'protected,\n',
                                                                               'along with the existing '
                                                                               'numerical checks and '
                                                                               'benchmark protocol. The '
                                                                               'structural\n',
                                                                               'guard enforces this '
                                                                               'boundary; it does not '
                                                                               'isolate arbitrary Python '
                                                                               'execution.\n'],
                                                                              [])]},
 'tasks/triton2triton/vllm/triton_lora_expand/config.yaml': {'after_sha256': '5599d6b8efcdc61ac58b8738c781bc8692527b25fa26ff62f6cb67d104d14ec1',
                                                             'before_sha256': '440ee2bc0389a8f18b1b9082fb00fe104a869619819309b0431e126178602131',
                                                             'reverse_edits': [(6,
                                                                                13,
                                                                                ['  - path: '
                                                                                 'source/triton_lora_expand.py\n',
                                                                                 '    scope: symbols\n',
                                                                                 '    symbols:\n',
                                                                                 '    - mm_k\n',
                                                                                 '    - do_expand_kernel\n',
                                                                                 '    - '
                                                                                 '_lora_expand_kernel\n',
                                                                                 '    allow_new_helpers: '
                                                                                 'true\n'],
                                                                                ['  - '
                                                                                 'source/triton_lora_expand.py\n'])]},
 'tasks/triton2triton/vllm/triton_lora_shrink/README.md': {'after_sha256': '337dfc337a34cacf1fa6427dd6a0388b08ef53841a3b0df72cf0dff5a4993a19',
                                                           'before_sha256': 'a11e56f3f3f7fc809aa99f322b9d1afb4a6d00252f568fd24c37ade7d02f0dd1',
                                                           'reverse_edits': [(79,
                                                                              88,
                                                                              ['\n',
                                                                               '\n',
                                                                               '### Editable device '
                                                                               'implementation boundary\n',
                                                                               '\n',
                                                                               'The editable device '
                                                                               'functions are `mm_k`, '
                                                                               '`do_shrink_kernel`, '
                                                                               '`_lora_shrink_kernel`.\n',
                                                                               'Their bodies and additional '
                                                                               'implementation helpers may '
                                                                               'be optimized. Module\n',
                                                                               'imports, initialization, '
                                                                               'host dispatch and pointer '
                                                                               'preparation remain '
                                                                               'protected,\n',
                                                                               'along with the existing '
                                                                               'numerical checks and '
                                                                               'benchmark protocol. The '
                                                                               'structural\n',
                                                                               'guard enforces this '
                                                                               'boundary; it does not '
                                                                               'isolate arbitrary Python '
                                                                               'execution.\n'],
                                                                              [])]},
 'tasks/triton2triton/vllm/triton_lora_shrink/config.yaml': {'after_sha256': 'bfdb6bb0a29cb205f22647f5176caf55c31ed3db818248f0868448d6c8a8fdf0',
                                                             'before_sha256': '313c1c2ccb404c3f9ca3045735eaa78d7224d5189ad221a89633e1830c29dbb5',
                                                             'reverse_edits': [(6,
                                                                                13,
                                                                                ['  - path: '
                                                                                 'source/triton_lora_shrink.py\n',
                                                                                 '    scope: symbols\n',
                                                                                 '    symbols:\n',
                                                                                 '    - mm_k\n',
                                                                                 '    - do_shrink_kernel\n',
                                                                                 '    - '
                                                                                 '_lora_shrink_kernel\n',
                                                                                 '    allow_new_helpers: '
                                                                                 'true\n'],
                                                                                ['  - '
                                                                                 'source/triton_lora_shrink.py\n'])]},
 'tasks/triton2triton/vllm/triton_matmul_persistent/README.md': {'after_sha256': '785b8ccd17474e0d4a077b2a5b10883e5799280cdaa46f36acc7adaee1abe316',
                                                                 'before_sha256': '0e46ab7d08ae6b77aa256fa89b5fcbad1a8659af32cf5f344fd76911b638995e',
                                                                 'reverse_edits': [(63,
                                                                                    72,
                                                                                    ['\n',
                                                                                     '\n',
                                                                                     '### Editable device '
                                                                                     'implementation '
                                                                                     'boundary\n',
                                                                                     '\n',
                                                                                     'The editable device '
                                                                                     'functions are '
                                                                                     '`_compute_pid`, '
                                                                                     '`matmul_kernel_persistent`.\n',
                                                                                     'Their bodies and '
                                                                                     'additional '
                                                                                     'implementation helpers '
                                                                                     'may be optimized. '
                                                                                     'Module\n',
                                                                                     'imports, '
                                                                                     'initialization, host '
                                                                                     'dispatch and pointer '
                                                                                     'preparation remain '
                                                                                     'protected,\n',
                                                                                     'along with the '
                                                                                     'existing numerical '
                                                                                     'checks and benchmark '
                                                                                     'protocol. The '
                                                                                     'structural\n',
                                                                                     'guard enforces this '
                                                                                     'boundary; it does not '
                                                                                     'isolate arbitrary '
                                                                                     'Python execution.\n'],
                                                                                    [])]},
 'tasks/triton2triton/vllm/triton_matmul_persistent/config.yaml': {'after_sha256': '173346b0733047ebc9be30b5b2af7c6f20dda5127fc9051890c5c270767821cf',
                                                                   'before_sha256': '902b575005ab4bb8a563ea1ed7bf68e4f3239473fbf64eb8c282dec59aaa8d72',
                                                                   'reverse_edits': [(6,
                                                                                      12,
                                                                                      ['  - path: '
                                                                                       'source/triton_matmul_persistent.py\n',
                                                                                       '    scope: symbols\n',
                                                                                       '    symbols:\n',
                                                                                       '    - _compute_pid\n',
                                                                                       '    - '
                                                                                       'matmul_kernel_persistent\n',
                                                                                       '    '
                                                                                       'allow_new_helpers: '
                                                                                       'true\n'],
                                                                                      ['  - '
                                                                                       'source/triton_matmul_persistent.py\n'])]},
 'tasks/triton2triton/vllm/triton_mean/README.md': {'after_sha256': 'b5d0cd154e6eff6928a423005ac45881a9bb8cde32a95afd6f312c4b62d9c7f0',
                                                    'before_sha256': '8bd9e87ca515a8e2caadbd1828d2ded335f531f22308ed16f5a22c9e9f99672e',
                                                    'reverse_edits': [(60,
                                                                       69,
                                                                       ['\n',
                                                                        '\n',
                                                                        '### Editable device implementation '
                                                                        'boundary\n',
                                                                        '\n',
                                                                        'The editable device functions are '
                                                                        '`mean_kernel`.\n',
                                                                        'Their bodies and additional '
                                                                        'implementation helpers may be '
                                                                        'optimized. Module\n',
                                                                        'imports, initialization, host '
                                                                        'dispatch and pointer preparation '
                                                                        'remain protected,\n',
                                                                        'along with the existing numerical '
                                                                        'checks and benchmark protocol. The '
                                                                        'structural\n',
                                                                        'guard enforces this boundary; it '
                                                                        'does not isolate arbitrary Python '
                                                                        'execution.\n'],
                                                                       [])]},
 'tasks/triton2triton/vllm/triton_mean/config.yaml': {'after_sha256': 'ee104296178d1ccf897e4e50770c1348e5c63b9979b91a02ce1691335429299b',
                                                      'before_sha256': 'c3f57fb29709e2fb9c7f5c6dce970f8ac0d474ac143be383dc21b74e5dbb26a4',
                                                      'reverse_edits': [(6,
                                                                         11,
                                                                         ['  - path: source/triton_mean.py\n',
                                                                          '    scope: symbols\n',
                                                                          '    symbols:\n',
                                                                          '    - mean_kernel\n',
                                                                          '    allow_new_helpers: true\n'],
                                                                         ['  - source/triton_mean.py\n'])]},
 'tasks/triton2triton/vllm/triton_merge_16x16_to_32x32/README.md': {'after_sha256': 'e5d4f924c5a65adec6d0373037b8fe8150c08f29584b619fedcbd2c4f981aabd',
                                                                    'before_sha256': '935b705e42500ef092e764087f293f71806a789001ed6b5e8e739f9cf38fae54',
                                                                    'reverse_edits': [(55,
                                                                                       64,
                                                                                       ['\n',
                                                                                        '\n',
                                                                                        '### Editable device '
                                                                                        'implementation '
                                                                                        'boundary\n',
                                                                                        '\n',
                                                                                        'The editable device '
                                                                                        'functions are '
                                                                                        '`merge_16x16_to_32x32_inverse_kernel`.\n',
                                                                                        'Their bodies and '
                                                                                        'additional '
                                                                                        'implementation '
                                                                                        'helpers may be '
                                                                                        'optimized. Module\n',
                                                                                        'imports, '
                                                                                        'initialization, '
                                                                                        'host dispatch and '
                                                                                        'pointer preparation '
                                                                                        'remain protected,\n',
                                                                                        'along with the '
                                                                                        'existing numerical '
                                                                                        'checks and '
                                                                                        'benchmark protocol. '
                                                                                        'The structural\n',
                                                                                        'guard enforces this '
                                                                                        'boundary; it does '
                                                                                        'not isolate '
                                                                                        'arbitrary Python '
                                                                                        'execution.\n'],
                                                                                       [])]},
 'tasks/triton2triton/vllm/triton_merge_16x16_to_32x32/config.yaml': {'after_sha256': '216eb7997a713f039541f31b30e506a4acf96640dcedf247d3e0d49c80dfd878',
                                                                      'before_sha256': 'e366a210f4501524d3ea499824bb46c82760c105cd1a895ae36f59ba9caba953',
                                                                      'reverse_edits': [(6,
                                                                                         11,
                                                                                         ['  - path: '
                                                                                          'source/triton_merge_16x16_to_32x32.py\n',
                                                                                          '    scope: '
                                                                                          'symbols\n',
                                                                                          '    symbols:\n',
                                                                                          '    - '
                                                                                          'merge_16x16_to_32x32_inverse_kernel\n',
                                                                                          '    '
                                                                                          'allow_new_helpers: '
                                                                                          'true\n'],
                                                                                         ['  - '
                                                                                          'source/triton_merge_16x16_to_32x32.py\n'])]},
 'tasks/triton2triton/vllm/triton_merge_attn_states/README.md': {'after_sha256': '9bdf3cd1221a83f3a9a61580a3c630f96141756321fe28c425afafcd5304ede8',
                                                                 'before_sha256': '6f11a6d78252446ffe1f4bebe1d235d29e8bfce8155601bbbe71535e6e13259f',
                                                                 'reverse_edits': [(44,
                                                                                    53,
                                                                                    ['\n',
                                                                                     '\n',
                                                                                     '### Editable device '
                                                                                     'implementation '
                                                                                     'boundary\n',
                                                                                     '\n',
                                                                                     'The editable device '
                                                                                     'functions are '
                                                                                     '`merge_attn_states_kernel`.\n',
                                                                                     'Their bodies and '
                                                                                     'additional '
                                                                                     'implementation helpers '
                                                                                     'may be optimized. '
                                                                                     'Module\n',
                                                                                     'imports, '
                                                                                     'initialization, host '
                                                                                     'dispatch and pointer '
                                                                                     'preparation remain '
                                                                                     'protected,\n',
                                                                                     'along with the '
                                                                                     'existing numerical '
                                                                                     'checks and benchmark '
                                                                                     'protocol. The '
                                                                                     'structural\n',
                                                                                     'guard enforces this '
                                                                                     'boundary; it does not '
                                                                                     'isolate arbitrary '
                                                                                     'Python execution.\n'],
                                                                                    [])]},
 'tasks/triton2triton/vllm/triton_merge_attn_states/config.yaml': {'after_sha256': 'b81487fe1dcc58b1a79a6f71c1dd8d78f86adf98b09824541fccd0d204a07982',
                                                                   'before_sha256': 'e7d6291904f5a87e3b14a561e60a781b7074391c863c3568e74dca4709e43d9b',
                                                                   'reverse_edits': [(6,
                                                                                      11,
                                                                                      ['  - path: '
                                                                                       'source/triton_merge_attn_states.py\n',
                                                                                       '    scope: symbols\n',
                                                                                       '    symbols:\n',
                                                                                       '    - '
                                                                                       'merge_attn_states_kernel\n',
                                                                                       '    '
                                                                                       'allow_new_helpers: '
                                                                                       'true\n'],
                                                                                      ['  - '
                                                                                       'source/triton_merge_attn_states.py\n'])]},
 'tasks/triton2triton/vllm/triton_min_p/README.md': {'after_sha256': '6e2a8bdb32719268b795cb177a25a01367faca40d45b5d6bda31a01bfacf13ec',
                                                     'before_sha256': '704b6b17285d21fd3d609df5884b8ba9bfc0799aad869604829250a52c88aa5c',
                                                     'reverse_edits': [(30,
                                                                        39,
                                                                        ['\n',
                                                                         '\n',
                                                                         '### Editable device implementation '
                                                                         'boundary\n',
                                                                         '\n',
                                                                         'The editable device functions are '
                                                                         '`_min_p_kernel`.\n',
                                                                         'Their bodies and additional '
                                                                         'implementation helpers may be '
                                                                         'optimized. Module\n',
                                                                         'imports, initialization, host '
                                                                         'dispatch and pointer preparation '
                                                                         'remain protected,\n',
                                                                         'along with the existing numerical '
                                                                         'checks and benchmark protocol. The '
                                                                         'structural\n',
                                                                         'guard enforces this boundary; it '
                                                                         'does not isolate arbitrary Python '
                                                                         'execution.\n'],
                                                                        [])]},
 'tasks/triton2triton/vllm/triton_min_p/config.yaml': {'after_sha256': '451ecdc34ee5ce1c1269bb617dfaa5835d8845a1f8c4d68843b5d1035018d163',
                                                       'before_sha256': 'cbfcd4e4f1e7e24da8ed20e8ee8a1264d3035115642ba2577cec70e4274545b3',
                                                       'reverse_edits': [(6,
                                                                          11,
                                                                          ['  - path: '
                                                                           'source/triton_min_p.py\n',
                                                                           '    scope: symbols\n',
                                                                           '    symbols:\n',
                                                                           '    - _min_p_kernel\n',
                                                                           '    allow_new_helpers: true\n'],
                                                                          ['  - source/triton_min_p.py\n'])]},
 'tasks/triton2triton/vllm/triton_mrope/README.md': {'after_sha256': 'bd741fe955f9323dbd4699deccd52d9cb3356d4b5097eb9034609f54fc605a68',
                                                     'before_sha256': '6c6a3a34e733c8a6c9fedfa1acb8cd88068ef63fd9647b71f2898018c354f9f2',
                                                     'reverse_edits': [(59,
                                                                        68,
                                                                        ['\n',
                                                                         '\n',
                                                                         '### Editable device implementation '
                                                                         'boundary\n',
                                                                         '\n',
                                                                         'The editable device functions are '
                                                                         '`_triton_mrope_forward`.\n',
                                                                         'Their bodies and additional '
                                                                         'implementation helpers may be '
                                                                         'optimized. Module\n',
                                                                         'imports, initialization, host '
                                                                         'dispatch and pointer preparation '
                                                                         'remain protected,\n',
                                                                         'along with the existing numerical '
                                                                         'checks and benchmark protocol. The '
                                                                         'structural\n',
                                                                         'guard enforces this boundary; it '
                                                                         'does not isolate arbitrary Python '
                                                                         'execution.\n'],
                                                                        [])]},
 'tasks/triton2triton/vllm/triton_mrope/config.yaml': {'after_sha256': '578f4fed58c2c6bb8cadf907064c765a627a03112024921cc5e4aa435f07c0fc',
                                                       'before_sha256': 'bd290946f28272245199bd7977af5f94c31fd8f29f9ede48deadac81d86403be',
                                                       'reverse_edits': [(6,
                                                                          11,
                                                                          ['  - path: '
                                                                           'source/triton_mrope.py\n',
                                                                           '    scope: symbols\n',
                                                                           '    symbols:\n',
                                                                           '    - _triton_mrope_forward\n',
                                                                           '    allow_new_helpers: true\n'],
                                                                          ['  - source/triton_mrope.py\n'])]},
 'tasks/triton2triton/vllm/triton_paged_prefix_prefill/README.md': {'after_sha256': '22b77cd2cb40f5ffcc1e8a103237adf667bcb27d0f0952c0286fcdea96198143',
                                                                    'before_sha256': '9b9f0668b7673c7acf3d85d6cb9724a850e6da0da039499e88a45960b62e1adb',
                                                                    'reverse_edits': [(92,
                                                                                       101,
                                                                                       ['\n',
                                                                                        '\n',
                                                                                        '### Editable device '
                                                                                        'implementation '
                                                                                        'boundary\n',
                                                                                        '\n',
                                                                                        'The editable device '
                                                                                        'functions are '
                                                                                        '`_fwd_kernel`.\n',
                                                                                        'Their bodies and '
                                                                                        'additional '
                                                                                        'implementation '
                                                                                        'helpers may be '
                                                                                        'optimized. Module\n',
                                                                                        'imports, '
                                                                                        'initialization, '
                                                                                        'host dispatch and '
                                                                                        'pointer preparation '
                                                                                        'remain protected,\n',
                                                                                        'along with the '
                                                                                        'existing numerical '
                                                                                        'checks and '
                                                                                        'benchmark protocol. '
                                                                                        'The structural\n',
                                                                                        'guard enforces this '
                                                                                        'boundary; it does '
                                                                                        'not isolate '
                                                                                        'arbitrary Python '
                                                                                        'execution.\n'],
                                                                                       [])]},
 'tasks/triton2triton/vllm/triton_paged_prefix_prefill/config.yaml': {'after_sha256': '7cc272bea549b0eca3fdf675b9a0654e6870167a0bfedc966fd518518d8b097a',
                                                                      'before_sha256': '6140359155f3f66e00bb28378ae9675f04b48a3c51a1ab21ff546d5b963a8269',
                                                                      'reverse_edits': [(6,
                                                                                         11,
                                                                                         ['  - path: '
                                                                                          'source/triton_paged_prefix_prefill.py\n',
                                                                                          '    scope: '
                                                                                          'symbols\n',
                                                                                          '    symbols:\n',
                                                                                          '    - '
                                                                                          '_fwd_kernel\n',
                                                                                          '    '
                                                                                          'allow_new_helpers: '
                                                                                          'true\n'],
                                                                                         ['  - '
                                                                                          'source/triton_paged_prefix_prefill.py\n'])]},
 'tasks/triton2triton/vllm/triton_paged_prefix_prefill_alibi/README.md': {'after_sha256': 'af9f0bead07356e26418cbffaa96fe8faf708987a742f8520320225684f43825',
                                                                          'before_sha256': '765fb8a7564c72024f9595a0aedc69bcc82bf96bb6cf59db0709003dc29699a2',
                                                                          'reverse_edits': [(112,
                                                                                             121,
                                                                                             ['\n',
                                                                                              '\n',
                                                                                              '### Editable '
                                                                                              'device '
                                                                                              'implementation '
                                                                                              'boundary\n',
                                                                                              '\n',
                                                                                              'The editable '
                                                                                              'device '
                                                                                              'functions are '
                                                                                              '`_fwd_kernel_alibi`.\n',
                                                                                              'Their bodies '
                                                                                              'and '
                                                                                              'additional '
                                                                                              'implementation '
                                                                                              'helpers may '
                                                                                              'be optimized. '
                                                                                              'Module\n',
                                                                                              'imports, '
                                                                                              'initialization, '
                                                                                              'host dispatch '
                                                                                              'and pointer '
                                                                                              'preparation '
                                                                                              'remain '
                                                                                              'protected,\n',
                                                                                              'along with '
                                                                                              'the existing '
                                                                                              'numerical '
                                                                                              'checks and '
                                                                                              'benchmark '
                                                                                              'protocol. The '
                                                                                              'structural\n',
                                                                                              'guard '
                                                                                              'enforces this '
                                                                                              'boundary; it '
                                                                                              'does not '
                                                                                              'isolate '
                                                                                              'arbitrary '
                                                                                              'Python '
                                                                                              'execution.\n'],
                                                                                             [])]},
 'tasks/triton2triton/vllm/triton_paged_prefix_prefill_alibi/config.yaml': {'after_sha256': '1459ef42022d411b6c358cd1ce4697b2c12c1c53d400c74b94f7e20d71f855cb',
                                                                            'before_sha256': 'b6b06a0041f97d27c626a5c477ae7751cdaa29683130ebe9be687c91e888eb40',
                                                                            'reverse_edits': [(6,
                                                                                               11,
                                                                                               ['  - path: '
                                                                                                'source/triton_paged_prefix_prefill_alibi.py\n',
                                                                                                '    scope: '
                                                                                                'symbols\n',
                                                                                                '    '
                                                                                                'symbols:\n',
                                                                                                '    - '
                                                                                                '_fwd_kernel_alibi\n',
                                                                                                '    '
                                                                                                'allow_new_helpers: '
                                                                                                'true\n'],
                                                                                               ['  - '
                                                                                                'source/triton_paged_prefix_prefill_alibi.py\n'])]},
 'tasks/triton2triton/vllm/triton_penalties/README.md': {'after_sha256': '2a8327217f233f1b239ad448d97c27c504ae9785cfe0109f0c747ee514380d51',
                                                         'before_sha256': 'e5582672f76bfd4ef57a396c38b6859d99747eb8105fb0fdad5a98fcfa60c3f2',
                                                         'reverse_edits': [(43,
                                                                            52,
                                                                            ['\n',
                                                                             '\n',
                                                                             '### Editable device '
                                                                             'implementation boundary\n',
                                                                             '\n',
                                                                             'The editable device functions '
                                                                             'are `_penalties_kernel`.\n',
                                                                             'Their bodies and additional '
                                                                             'implementation helpers may be '
                                                                             'optimized. Module\n',
                                                                             'imports, initialization, host '
                                                                             'dispatch and pointer '
                                                                             'preparation remain '
                                                                             'protected,\n',
                                                                             'along with the existing '
                                                                             'numerical checks and benchmark '
                                                                             'protocol. The structural\n',
                                                                             'guard enforces this boundary; '
                                                                             'it does not isolate arbitrary '
                                                                             'Python execution.\n'],
                                                                            [])]},
 'tasks/triton2triton/vllm/triton_penalties/config.yaml': {'after_sha256': '4a21f1dd179fbe50b9c9e97a7478f3c4e6b53e4af0c57c2db7807fe20a38f4b5',
                                                           'before_sha256': 'c87835714a24223c95940ed03d26116831111a96fb7198b2a3c746ecb60a6ba1',
                                                           'reverse_edits': [(6,
                                                                              11,
                                                                              ['  - path: '
                                                                               'source/triton_penalties.py\n',
                                                                               '    scope: symbols\n',
                                                                               '    symbols:\n',
                                                                               '    - _penalties_kernel\n',
                                                                               '    allow_new_helpers: '
                                                                               'true\n'],
                                                                              ['  - '
                                                                               'source/triton_penalties.py\n'])]},
 'tasks/triton2triton/vllm/triton_per_token_group_quant_fp8_colmajor/README.md': {'after_sha256': '7558ab9bf1409a7c61f0759f7e8d623a87b9260d070b137f458005023699320f',
                                                                                  'before_sha256': 'f8f2975deae4f6a891bc876ab93cf5e2dfde09273a206377c105a333122b6382',
                                                                                  'reverse_edits': [(38,
                                                                                                     47,
                                                                                                     ['\n',
                                                                                                      '\n',
                                                                                                      '### '
                                                                                                      'Editable '
                                                                                                      'device '
                                                                                                      'implementation '
                                                                                                      'boundary\n',
                                                                                                      '\n',
                                                                                                      'The '
                                                                                                      'editable '
                                                                                                      'device '
                                                                                                      'functions '
                                                                                                      'are '
                                                                                                      '`_per_token_group_quant_fp8_colmajor`.\n',
                                                                                                      'Their '
                                                                                                      'bodies '
                                                                                                      'and '
                                                                                                      'additional '
                                                                                                      'implementation '
                                                                                                      'helpers '
                                                                                                      'may '
                                                                                                      'be '
                                                                                                      'optimized. '
                                                                                                      'Module\n',
                                                                                                      'imports, '
                                                                                                      'initialization, '
                                                                                                      'host '
                                                                                                      'dispatch '
                                                                                                      'and '
                                                                                                      'pointer '
                                                                                                      'preparation '
                                                                                                      'remain '
                                                                                                      'protected,\n',
                                                                                                      'along '
                                                                                                      'with '
                                                                                                      'the '
                                                                                                      'existing '
                                                                                                      'numerical '
                                                                                                      'checks '
                                                                                                      'and '
                                                                                                      'benchmark '
                                                                                                      'protocol. '
                                                                                                      'The '
                                                                                                      'structural\n',
                                                                                                      'guard '
                                                                                                      'enforces '
                                                                                                      'this '
                                                                                                      'boundary; '
                                                                                                      'it '
                                                                                                      'does '
                                                                                                      'not '
                                                                                                      'isolate '
                                                                                                      'arbitrary '
                                                                                                      'Python '
                                                                                                      'execution.\n'],
                                                                                                     [])]},
 'tasks/triton2triton/vllm/triton_per_token_group_quant_fp8_colmajor/config.yaml': {'after_sha256': 'b95c4b7d2c45a36517fd45eef363e46665fb6de21f968cd41d6be2adf970d9c4',
                                                                                    'before_sha256': '15a3725e42c8657785196afd604fee95f3b9c5bb96df4483b6ba5f7271842746',
                                                                                    'reverse_edits': [(6,
                                                                                                       11,
                                                                                                       ['  - '
                                                                                                        'path: '
                                                                                                        'source/triton_per_token_group_quant_fp8_colmajor.py\n',
                                                                                                        '    '
                                                                                                        'scope: '
                                                                                                        'symbols\n',
                                                                                                        '    '
                                                                                                        'symbols:\n',
                                                                                                        '    '
                                                                                                        '- '
                                                                                                        '_per_token_group_quant_fp8_colmajor\n',
                                                                                                        '    '
                                                                                                        'allow_new_helpers: '
                                                                                                        'true\n'],
                                                                                                       ['  - '
                                                                                                        'source/triton_per_token_group_quant_fp8_colmajor.py\n'])]},
 'tasks/triton2triton/vllm/triton_per_token_group_quant_int8/README.md': {'after_sha256': '8f8665c9d509a8aa46c73cc767124a37121e3523f58cd99e7153d37763198fb7',
                                                                          'before_sha256': '42e1099f5b25787703190f5fd026cdea54fecd105891f767928b481f86ead9be',
                                                                          'reverse_edits': [(42,
                                                                                             51,
                                                                                             ['\n',
                                                                                              '\n',
                                                                                              '### Editable '
                                                                                              'device '
                                                                                              'implementation '
                                                                                              'boundary\n',
                                                                                              '\n',
                                                                                              'The editable '
                                                                                              'device '
                                                                                              'functions are '
                                                                                              '`_per_token_group_quant_int8`.\n',
                                                                                              'Their bodies '
                                                                                              'and '
                                                                                              'additional '
                                                                                              'implementation '
                                                                                              'helpers may '
                                                                                              'be optimized. '
                                                                                              'Module\n',
                                                                                              'imports, '
                                                                                              'initialization, '
                                                                                              'host dispatch '
                                                                                              'and pointer '
                                                                                              'preparation '
                                                                                              'remain '
                                                                                              'protected,\n',
                                                                                              'along with '
                                                                                              'the existing '
                                                                                              'numerical '
                                                                                              'checks and '
                                                                                              'benchmark '
                                                                                              'protocol. The '
                                                                                              'structural\n',
                                                                                              'guard '
                                                                                              'enforces this '
                                                                                              'boundary; it '
                                                                                              'does not '
                                                                                              'isolate '
                                                                                              'arbitrary '
                                                                                              'Python '
                                                                                              'execution.\n'],
                                                                                             [])]},
 'tasks/triton2triton/vllm/triton_per_token_group_quant_int8/config.yaml': {'after_sha256': 'ff25dceeed235597a061a14e39b34a9e8c1a76a902f13617d676e865ddedd334',
                                                                            'before_sha256': '5263a42ab1e4694b592059b8a745f1b2a3d48b403b70ac40805e4ac6772fdc25',
                                                                            'reverse_edits': [(6,
                                                                                               11,
                                                                                               ['  - path: '
                                                                                                'source/triton_per_token_group_quant_int8.py\n',
                                                                                                '    scope: '
                                                                                                'symbols\n',
                                                                                                '    '
                                                                                                'symbols:\n',
                                                                                                '    - '
                                                                                                '_per_token_group_quant_int8\n',
                                                                                                '    '
                                                                                                'allow_new_helpers: '
                                                                                                'true\n'],
                                                                                               ['  - '
                                                                                                'source/triton_per_token_group_quant_int8.py\n'])]},
 'tasks/triton2triton/vllm/triton_per_token_quant_int8/README.md': {'after_sha256': '5b1cdd96658e25336816f3dec8ce1b0699cef0d4931c0b7e68d16c35c3551cb0',
                                                                    'before_sha256': '52052c7e60fc57229a2b86e7736f01f768cf68b4556f8377595f7ef7680b473e',
                                                                    'reverse_edits': [(38,
                                                                                       47,
                                                                                       ['\n',
                                                                                        '\n',
                                                                                        '### Editable device '
                                                                                        'implementation '
                                                                                        'boundary\n',
                                                                                        '\n',
                                                                                        'The editable device '
                                                                                        'functions are '
                                                                                        '`_per_token_quant_int8`.\n',
                                                                                        'Their bodies and '
                                                                                        'additional '
                                                                                        'implementation '
                                                                                        'helpers may be '
                                                                                        'optimized. Module\n',
                                                                                        'imports, '
                                                                                        'initialization, '
                                                                                        'host dispatch and '
                                                                                        'pointer preparation '
                                                                                        'remain protected,\n',
                                                                                        'along with the '
                                                                                        'existing numerical '
                                                                                        'checks and '
                                                                                        'benchmark protocol. '
                                                                                        'The structural\n',
                                                                                        'guard enforces this '
                                                                                        'boundary; it does '
                                                                                        'not isolate '
                                                                                        'arbitrary Python '
                                                                                        'execution.\n'],
                                                                                       [])]},
 'tasks/triton2triton/vllm/triton_per_token_quant_int8/config.yaml': {'after_sha256': '39ae15d077be4a29d58b60f1c308019484616112070674622150145badc60035',
                                                                      'before_sha256': '441a8106d7efc9f0c0dc7f6c36c90a28efeb621c87451d0692e16a9e363ee831',
                                                                      'reverse_edits': [(6,
                                                                                         11,
                                                                                         ['  - path: '
                                                                                          'source/triton_per_token_quant_int8.py\n',
                                                                                          '    scope: '
                                                                                          'symbols\n',
                                                                                          '    symbols:\n',
                                                                                          '    - '
                                                                                          '_per_token_quant_int8\n',
                                                                                          '    '
                                                                                          'allow_new_helpers: '
                                                                                          'true\n'],
                                                                                         ['  - '
                                                                                          'source/triton_per_token_quant_int8.py\n'])]},
 'tasks/triton2triton/vllm/triton_rejection_sample/README.md': {'after_sha256': '28eb4b42d001e80476767edbf85efc79768418095ed8181e4dde26576a8e8bb7',
                                                                'before_sha256': '4cfc7d6f346342032ac3ab04e77da9d3fd2f2ed07e3d9cb85d9024526771a96f',
                                                                'reverse_edits': [(39,
                                                                                   48,
                                                                                   ['\n',
                                                                                    '\n',
                                                                                    '### Editable device '
                                                                                    'implementation '
                                                                                    'boundary\n',
                                                                                    '\n',
                                                                                    'The editable device '
                                                                                    'functions are '
                                                                                    '`_rejection_sample_kernel`.\n',
                                                                                    'Their bodies and '
                                                                                    'additional '
                                                                                    'implementation helpers '
                                                                                    'may be optimized. '
                                                                                    'Module\n',
                                                                                    'imports, '
                                                                                    'initialization, host '
                                                                                    'dispatch and pointer '
                                                                                    'preparation remain '
                                                                                    'protected,\n',
                                                                                    'along with the existing '
                                                                                    'numerical checks and '
                                                                                    'benchmark protocol. The '
                                                                                    'structural\n',
                                                                                    'guard enforces this '
                                                                                    'boundary; it does not '
                                                                                    'isolate arbitrary '
                                                                                    'Python execution.\n'],
                                                                                   [])]},
 'tasks/triton2triton/vllm/triton_rejection_sample/config.yaml': {'after_sha256': '09ad9ce0ff1ba90775985212fbecba18233307a30f3bfc479e5b293a0ff72bdc',
                                                                  'before_sha256': 'ed7bee1536ec258197e7e2123dae1b5efb1e2b4189f42589f09d226373300679',
                                                                  'reverse_edits': [(6,
                                                                                     11,
                                                                                     ['  - path: '
                                                                                      'source/triton_rejection_sample.py\n',
                                                                                      '    scope: symbols\n',
                                                                                      '    symbols:\n',
                                                                                      '    - '
                                                                                      '_rejection_sample_kernel\n',
                                                                                      '    '
                                                                                      'allow_new_helpers: '
                                                                                      'true\n'],
                                                                                     ['  - '
                                                                                      'source/triton_rejection_sample.py\n'])]},
 'tasks/triton2triton/vllm/triton_sample_recovered_tokens/README.md': {'after_sha256': 'a513a667069f51bfe2d50c930e3288e7beb72b19fd1119ba359531b5f8878629',
                                                                       'before_sha256': '00a8f44d6cc7f9d73e67baef45da8ea7d3788e082429a53cbdbaa43791455539',
                                                                       'reverse_edits': [(44,
                                                                                          53,
                                                                                          ['\n',
                                                                                           '\n',
                                                                                           '### Editable '
                                                                                           'device '
                                                                                           'implementation '
                                                                                           'boundary\n',
                                                                                           '\n',
                                                                                           'The editable '
                                                                                           'device functions '
                                                                                           'are '
                                                                                           '`sample_recovered_tokens_kernel`.\n',
                                                                                           'Their bodies and '
                                                                                           'additional '
                                                                                           'implementation '
                                                                                           'helpers may be '
                                                                                           'optimized. '
                                                                                           'Module\n',
                                                                                           'imports, '
                                                                                           'initialization, '
                                                                                           'host dispatch '
                                                                                           'and pointer '
                                                                                           'preparation '
                                                                                           'remain '
                                                                                           'protected,\n',
                                                                                           'along with the '
                                                                                           'existing '
                                                                                           'numerical checks '
                                                                                           'and benchmark '
                                                                                           'protocol. The '
                                                                                           'structural\n',
                                                                                           'guard enforces '
                                                                                           'this boundary; '
                                                                                           'it does not '
                                                                                           'isolate '
                                                                                           'arbitrary Python '
                                                                                           'execution.\n'],
                                                                                          [])]},
 'tasks/triton2triton/vllm/triton_sample_recovered_tokens/config.yaml': {'after_sha256': 'beabe1b62489b63d88942bf44dbe2b76d1249f56ed40abb890fc8a55c946c79f',
                                                                         'before_sha256': '254909e7c398b0621191ef88f04695334251954f1541ae5eace0accc2b4fadcf',
                                                                         'reverse_edits': [(6,
                                                                                            11,
                                                                                            ['  - path: '
                                                                                             'source/triton_sample_recovered_tokens.py\n',
                                                                                             '    scope: '
                                                                                             'symbols\n',
                                                                                             '    symbols:\n',
                                                                                             '    - '
                                                                                             'sample_recovered_tokens_kernel\n',
                                                                                             '    '
                                                                                             'allow_new_helpers: '
                                                                                             'true\n'],
                                                                                            ['  - '
                                                                                             'source/triton_sample_recovered_tokens.py\n'])]},
 'tasks/triton2triton/vllm/triton_scaled_mm/README.md': {'after_sha256': '64c385dcfa5b2c81dc49073f205cdf84efdd920dfe561335cac66a0e5548293a',
                                                         'before_sha256': '34668c3f22b9ef5977f6211907001f507699d04557808afd8fcf16d0e7e35c75',
                                                         'reverse_edits': [(46,
                                                                            55,
                                                                            ['\n',
                                                                             '\n',
                                                                             '### Editable device '
                                                                             'implementation boundary\n',
                                                                             '\n',
                                                                             'The editable device functions '
                                                                             'are `scaled_mm_kernel`.\n',
                                                                             'Their bodies and additional '
                                                                             'implementation helpers may be '
                                                                             'optimized. Module\n',
                                                                             'imports, initialization, host '
                                                                             'dispatch and pointer '
                                                                             'preparation remain '
                                                                             'protected,\n',
                                                                             'along with the existing '
                                                                             'numerical checks and benchmark '
                                                                             'protocol. The structural\n',
                                                                             'guard enforces this boundary; '
                                                                             'it does not isolate arbitrary '
                                                                             'Python execution.\n'],
                                                                            [])]},
 'tasks/triton2triton/vllm/triton_scaled_mm/config.yaml': {'after_sha256': 'ac99a1b5b0cea20678b3fed86404b5fd0943e80cd6b9fed86562068156840f51',
                                                           'before_sha256': '0bd6449f5a1ca2e4f282e97a453f9ff7cf1f2f9137a621991943a66590a8e23f',
                                                           'reverse_edits': [(6,
                                                                              11,
                                                                              ['  - path: '
                                                                               'source/triton_scaled_mm.py\n',
                                                                               '    scope: symbols\n',
                                                                               '    symbols:\n',
                                                                               '    - scaled_mm_kernel\n',
                                                                               '    allow_new_helpers: '
                                                                               'true\n'],
                                                                              ['  - '
                                                                               'source/triton_scaled_mm.py\n'])]},
 'tasks/triton2triton/vllm/triton_silu_mul_fp8_quant_dg/README.md': {'after_sha256': '2813243834774a3de6dfcffd2924902615491efce579dfdf05d660314ce5ce69',
                                                                     'before_sha256': '4470271985b81190278a4304fc468d91d32a177b3e5a3f8d208f60717325fa48',
                                                                     'reverse_edits': [(59,
                                                                                        68,
                                                                                        ['\n',
                                                                                         '\n',
                                                                                         '### Editable '
                                                                                         'device '
                                                                                         'implementation '
                                                                                         'boundary\n',
                                                                                         '\n',
                                                                                         'The editable '
                                                                                         'device functions '
                                                                                         'are '
                                                                                         '`_silu_mul_fp8_quant_deep_gemm`.\n',
                                                                                         'Their bodies and '
                                                                                         'additional '
                                                                                         'implementation '
                                                                                         'helpers may be '
                                                                                         'optimized. '
                                                                                         'Module\n',
                                                                                         'imports, '
                                                                                         'initialization, '
                                                                                         'host dispatch and '
                                                                                         'pointer '
                                                                                         'preparation remain '
                                                                                         'protected,\n',
                                                                                         'along with the '
                                                                                         'existing numerical '
                                                                                         'checks and '
                                                                                         'benchmark '
                                                                                         'protocol. The '
                                                                                         'structural\n',
                                                                                         'guard enforces '
                                                                                         'this boundary; it '
                                                                                         'does not isolate '
                                                                                         'arbitrary Python '
                                                                                         'execution.\n'],
                                                                                        [])]},
 'tasks/triton2triton/vllm/triton_silu_mul_fp8_quant_dg/config.yaml': {'after_sha256': '09006b9821ed8345abebb2b2cc60b7de1398039e3033b437292504e760e0c722',
                                                                       'before_sha256': '6708ba2b1970ed5c44736eadc153b95914ff030f4302de32b7b2e4971b66a9ed',
                                                                       'reverse_edits': [(6,
                                                                                          11,
                                                                                          ['  - path: '
                                                                                           'source/triton_silu_mul_fp8_quant_dg.py\n',
                                                                                           '    scope: '
                                                                                           'symbols\n',
                                                                                           '    symbols:\n',
                                                                                           '    - '
                                                                                           '_silu_mul_fp8_quant_deep_gemm\n',
                                                                                           '    '
                                                                                           'allow_new_helpers: '
                                                                                           'true\n'],
                                                                                          ['  - '
                                                                                           'source/triton_silu_mul_fp8_quant_dg.py\n'])]},
 'tasks/triton2triton/vllm/triton_silu_mul_quant_fp8/README.md': {'after_sha256': '5d288e485ea72c598a18fd805b5ec9be986a02d533767dde9ddf6239116d2426',
                                                                  'before_sha256': 'f0c2b1acdb30f1c5f831e6fa26ff184220991352bd7e09381f8f31deaa09953b',
                                                                  'reverse_edits': [(43,
                                                                                     52,
                                                                                     ['\n',
                                                                                      '\n',
                                                                                      '### Editable device '
                                                                                      'implementation '
                                                                                      'boundary\n',
                                                                                      '\n',
                                                                                      'The editable device '
                                                                                      'functions are '
                                                                                      '`_silu_mul_per_token_group_quant_fp8_colmajor`.\n',
                                                                                      'Their bodies and '
                                                                                      'additional '
                                                                                      'implementation '
                                                                                      'helpers may be '
                                                                                      'optimized. Module\n',
                                                                                      'imports, '
                                                                                      'initialization, host '
                                                                                      'dispatch and pointer '
                                                                                      'preparation remain '
                                                                                      'protected,\n',
                                                                                      'along with the '
                                                                                      'existing numerical '
                                                                                      'checks and benchmark '
                                                                                      'protocol. The '
                                                                                      'structural\n',
                                                                                      'guard enforces this '
                                                                                      'boundary; it does not '
                                                                                      'isolate arbitrary '
                                                                                      'Python execution.\n'],
                                                                                     [])]},
 'tasks/triton2triton/vllm/triton_silu_mul_quant_fp8/config.yaml': {'after_sha256': '1031f26d34c3e6a88429cc747fb79089c78fc27517f4a1a6caca34e0fb10a809',
                                                                    'before_sha256': '53a3f188d15a7127ca07fa6b8ca21710077c77ba4a93c4fef9745594dbdb6fae',
                                                                    'reverse_edits': [(6,
                                                                                       11,
                                                                                       ['  - path: '
                                                                                        'source/triton_silu_mul_quant_fp8.py\n',
                                                                                        '    scope: '
                                                                                        'symbols\n',
                                                                                        '    symbols:\n',
                                                                                        '    - '
                                                                                        '_silu_mul_per_token_group_quant_fp8_colmajor\n',
                                                                                        '    '
                                                                                        'allow_new_helpers: '
                                                                                        'true\n'],
                                                                                       ['  - '
                                                                                        'source/triton_silu_mul_quant_fp8.py\n'])]},
 'tasks/triton2triton/vllm/triton_solve_tril_16x16/README.md': {'after_sha256': '215ae38f167f9dc1a87980996aa2cc89eb630f86d98a9e6e18565405714688a5',
                                                                'before_sha256': 'f849b191e367faa0973525ef2b7b96944ed865520fd9eb48c35a6c12db4f9c80',
                                                                'reverse_edits': [(47,
                                                                                   56,
                                                                                   ['\n',
                                                                                    '\n',
                                                                                    '### Editable device '
                                                                                    'implementation '
                                                                                    'boundary\n',
                                                                                    '\n',
                                                                                    'The editable device '
                                                                                    'functions are '
                                                                                    '`solve_tril_16x16_kernel`.\n',
                                                                                    'Their bodies and '
                                                                                    'additional '
                                                                                    'implementation helpers '
                                                                                    'may be optimized. '
                                                                                    'Module\n',
                                                                                    'imports, '
                                                                                    'initialization, host '
                                                                                    'dispatch and pointer '
                                                                                    'preparation remain '
                                                                                    'protected,\n',
                                                                                    'along with the existing '
                                                                                    'numerical checks and '
                                                                                    'benchmark protocol. The '
                                                                                    'structural\n',
                                                                                    'guard enforces this '
                                                                                    'boundary; it does not '
                                                                                    'isolate arbitrary '
                                                                                    'Python execution.\n'],
                                                                                   [])]},
 'tasks/triton2triton/vllm/triton_solve_tril_16x16/config.yaml': {'after_sha256': '5ea41f7f632b542dbc3b8fb117dfd7091147fa52d5329231b8bba07bcb75170e',
                                                                  'before_sha256': '6569a133d70f6ca9587a903dcbe56f27dac0f53ada1f01da3c4f6cc7e99e755c',
                                                                  'reverse_edits': [(6,
                                                                                     11,
                                                                                     ['  - path: '
                                                                                      'source/triton_solve_tril_16x16.py\n',
                                                                                      '    scope: symbols\n',
                                                                                      '    symbols:\n',
                                                                                      '    - '
                                                                                      'solve_tril_16x16_kernel\n',
                                                                                      '    '
                                                                                      'allow_new_helpers: '
                                                                                      'true\n'],
                                                                                     ['  - '
                                                                                      'source/triton_solve_tril_16x16.py\n'])]},
 'tasks/triton2triton/vllm/triton_topk_topp/README.md': {'after_sha256': '17a0d182ddc31aa3456bf55f70e69a6986845e295cadb1081072c0aae4a7562a',
                                                         'before_sha256': '54030c630f2211cece8808956895ec5056937c455d116974d94b93296e4824be',
                                                         'reverse_edits': [(41,
                                                                            50,
                                                                            ['\n',
                                                                             '\n',
                                                                             '### Editable device '
                                                                             'implementation boundary\n',
                                                                             '\n',
                                                                             'The editable device functions '
                                                                             'are `_topk_topp_kernel`.\n',
                                                                             'Their bodies and additional '
                                                                             'implementation helpers may be '
                                                                             'optimized. Module\n',
                                                                             'imports, initialization, host '
                                                                             'dispatch and pointer '
                                                                             'preparation remain '
                                                                             'protected,\n',
                                                                             'along with the existing '
                                                                             'numerical checks and benchmark '
                                                                             'protocol. The structural\n',
                                                                             'guard enforces this boundary; '
                                                                             'it does not isolate arbitrary '
                                                                             'Python execution.\n'],
                                                                            [])]},
 'tasks/triton2triton/vllm/triton_topk_topp/config.yaml': {'after_sha256': 'e969d682b8029f4aa47f34eeaec705a32f79ae69857d7e5914f6c0bda5a45434',
                                                           'before_sha256': '9411f9918be12ff15202181b56a0c4c75e05fdd381411551ae460ab56872f39a',
                                                           'reverse_edits': [(6,
                                                                              11,
                                                                              ['  - path: '
                                                                               'source/triton_topk_topp.py\n',
                                                                               '    scope: symbols\n',
                                                                               '    symbols:\n',
                                                                               '    - _topk_topp_kernel\n',
                                                                               '    allow_new_helpers: '
                                                                               'true\n'],
                                                                              ['  - '
                                                                               'source/triton_topk_topp.py\n'])]},
 'tasks/triton2triton/vllm/triton_unified_attention_2d/README.md': {'after_sha256': '03f52cab38e8b23ce442033ce9e08eedd84819c178e81e80a7e3a23cd88e9c4d',
                                                                    'before_sha256': '59e557a89fb25f195699afbae737407f7d2325351e4ef62c056a2dc104bbd335',
                                                                    'reverse_edits': [(75,
                                                                                       84,
                                                                                       ['\n',
                                                                                        '\n',
                                                                                        '### Editable device '
                                                                                        'implementation '
                                                                                        'boundary\n',
                                                                                        '\n',
                                                                                        'The editable device '
                                                                                        'functions are '
                                                                                        '`cdiv_fn`, '
                                                                                        '`apply_softcap`, '
                                                                                        '`find_seq_idx`, '
                                                                                        '`kernel_unified_attention_2d`.\n',
                                                                                        'Their bodies and '
                                                                                        'additional '
                                                                                        'implementation '
                                                                                        'helpers may be '
                                                                                        'optimized. Module\n',
                                                                                        'imports, '
                                                                                        'initialization, '
                                                                                        'host dispatch and '
                                                                                        'pointer preparation '
                                                                                        'remain protected,\n',
                                                                                        'along with the '
                                                                                        'existing numerical '
                                                                                        'checks and '
                                                                                        'benchmark protocol. '
                                                                                        'The structural\n',
                                                                                        'guard enforces this '
                                                                                        'boundary; it does '
                                                                                        'not isolate '
                                                                                        'arbitrary Python '
                                                                                        'execution.\n'],
                                                                                       [])]},
 'tasks/triton2triton/vllm/triton_unified_attention_2d/config.yaml': {'after_sha256': '3b645127e8b8b9f929e6d102a9f043099935413cb174267c9a4436bf92c4481b',
                                                                      'before_sha256': '31b6b7d72a8c6703ad5eb6f956febd0b5f4f64c682f11b643aed2328dcc2a294',
                                                                      'reverse_edits': [(6,
                                                                                         14,
                                                                                         ['  - path: '
                                                                                          'source/triton_unified_attention_2d.py\n',
                                                                                          '    scope: '
                                                                                          'symbols\n',
                                                                                          '    symbols:\n',
                                                                                          '    - cdiv_fn\n',
                                                                                          '    - '
                                                                                          'apply_softcap\n',
                                                                                          '    - '
                                                                                          'find_seq_idx\n',
                                                                                          '    - '
                                                                                          'kernel_unified_attention_2d\n',
                                                                                          '    '
                                                                                          'allow_new_helpers: '
                                                                                          'true\n'],
                                                                                         ['  - '
                                                                                          'source/triton_unified_attention_2d.py\n'])]},
 'tasks/triton2triton/vllm/triton_unified_attention_3d/README.md': {'after_sha256': '07be2b136d75106b315575bba812a31b20431a60cd943fdb6e486725bf44632c',
                                                                    'before_sha256': 'e56fa3e3752b2fa440a65b5b1ff5b222a798fc961ef9c5d874fbdf1daf8d50bc',
                                                                    'reverse_edits': [(75,
                                                                                       84,
                                                                                       ['\n',
                                                                                        '\n',
                                                                                        '### Editable device '
                                                                                        'implementation '
                                                                                        'boundary\n',
                                                                                        '\n',
                                                                                        'The editable device '
                                                                                        'functions are '
                                                                                        '`cdiv_fn`, '
                                                                                        '`apply_softcap`, '
                                                                                        '`find_seq_idx`, '
                                                                                        '`kernel_unified_attention_3d`.\n',
                                                                                        'Their bodies and '
                                                                                        'additional '
                                                                                        'implementation '
                                                                                        'helpers may be '
                                                                                        'optimized. Module\n',
                                                                                        'imports, '
                                                                                        'initialization, '
                                                                                        'host dispatch and '
                                                                                        'pointer preparation '
                                                                                        'remain protected,\n',
                                                                                        'along with the '
                                                                                        'existing numerical '
                                                                                        'checks and '
                                                                                        'benchmark protocol. '
                                                                                        'The structural\n',
                                                                                        'guard enforces this '
                                                                                        'boundary; it does '
                                                                                        'not isolate '
                                                                                        'arbitrary Python '
                                                                                        'execution.\n'],
                                                                                       [])]},
 'tasks/triton2triton/vllm/triton_unified_attention_3d/config.yaml': {'after_sha256': 'c30d7d6826e2eeaf0a1a80c0b553848367d05b123e87e31f43867ac85f5502df',
                                                                      'before_sha256': 'b20d3a7328cf9dfdb3a6c404c86fd8d1431c23e55def172ba5e372bdf3016156',
                                                                      'reverse_edits': [(6,
                                                                                         14,
                                                                                         ['  - path: '
                                                                                          'source/triton_unified_attention_3d.py\n',
                                                                                          '    scope: '
                                                                                          'symbols\n',
                                                                                          '    symbols:\n',
                                                                                          '    - cdiv_fn\n',
                                                                                          '    - '
                                                                                          'apply_softcap\n',
                                                                                          '    - '
                                                                                          'find_seq_idx\n',
                                                                                          '    - '
                                                                                          'kernel_unified_attention_3d\n',
                                                                                          '    '
                                                                                          'allow_new_helpers: '
                                                                                          'true\n'],
                                                                                         ['  - '
                                                                                          'source/triton_unified_attention_3d.py\n'])]},
 'tasks/triton2triton/vllm/triton_w8a8_block_int8_matmul/README.md': {'after_sha256': 'fb32276a001f986fee3ef7b961471657661a613564943e172555ee2d2d1b41bf',
                                                                      'before_sha256': '5109f1f44ae579b00a5086e5fefa48a195995015217dc749e29dc2c100627a6b',
                                                                      'reverse_edits': [(63,
                                                                                         72,
                                                                                         ['\n',
                                                                                          '\n',
                                                                                          '### Editable '
                                                                                          'device '
                                                                                          'implementation '
                                                                                          'boundary\n',
                                                                                          '\n',
                                                                                          'The editable '
                                                                                          'device functions '
                                                                                          'are '
                                                                                          '`_w8a8_block_int8_matmul`.\n',
                                                                                          'Their bodies and '
                                                                                          'additional '
                                                                                          'implementation '
                                                                                          'helpers may be '
                                                                                          'optimized. '
                                                                                          'Module\n',
                                                                                          'imports, '
                                                                                          'initialization, '
                                                                                          'host dispatch and '
                                                                                          'pointer '
                                                                                          'preparation '
                                                                                          'remain '
                                                                                          'protected,\n',
                                                                                          'along with the '
                                                                                          'existing '
                                                                                          'numerical checks '
                                                                                          'and benchmark '
                                                                                          'protocol. The '
                                                                                          'structural\n',
                                                                                          'guard enforces '
                                                                                          'this boundary; it '
                                                                                          'does not isolate '
                                                                                          'arbitrary Python '
                                                                                          'execution.\n'],
                                                                                         [])]},
 'tasks/triton2triton/vllm/triton_w8a8_block_int8_matmul/config.yaml': {'after_sha256': '5f483f9292f468cd97adc7024c0453f7e40686dac93042210cd1fed56ea4e9e2',
                                                                        'before_sha256': '769566d34e93447a4ebb1053710768c17a17c00c62cbb8aef895b99e195d6ef9',
                                                                        'reverse_edits': [(6,
                                                                                           11,
                                                                                           ['  - path: '
                                                                                            'source/triton_w8a8_block_int8_matmul.py\n',
                                                                                            '    scope: '
                                                                                            'symbols\n',
                                                                                            '    symbols:\n',
                                                                                            '    - '
                                                                                            '_w8a8_block_int8_matmul\n',
                                                                                            '    '
                                                                                            'allow_new_helpers: '
                                                                                            'true\n'],
                                                                                           ['  - '
                                                                                            'source/triton_w8a8_block_int8_matmul.py\n'])]},
 'tasks/triton2triton/vllm/triton_wy_fast_recompute_wu/README.md': {'after_sha256': 'ea7cf3e5da9544062e0c09852a5f5da98da42edd97c5c91907f61e25e2288c17',
                                                                    'before_sha256': '5fe011ce830fc585fc7c00832b068a75739c2037079a17db140248d506d657f7',
                                                                    'reverse_edits': [(45,
                                                                                       54,
                                                                                       ['\n',
                                                                                        '\n',
                                                                                        '### Editable device '
                                                                                        'implementation '
                                                                                        'boundary\n',
                                                                                        '\n',
                                                                                        'The editable device '
                                                                                        'functions are '
                                                                                        '`recompute_w_u_fwd_kernel`.\n',
                                                                                        'Their bodies and '
                                                                                        'additional '
                                                                                        'implementation '
                                                                                        'helpers may be '
                                                                                        'optimized. Module\n',
                                                                                        'imports, '
                                                                                        'initialization, '
                                                                                        'host dispatch and '
                                                                                        'pointer preparation '
                                                                                        'remain protected,\n',
                                                                                        'along with the '
                                                                                        'existing numerical '
                                                                                        'checks and '
                                                                                        'benchmark protocol. '
                                                                                        'The structural\n',
                                                                                        'guard enforces this '
                                                                                        'boundary; it does '
                                                                                        'not isolate '
                                                                                        'arbitrary Python '
                                                                                        'execution.\n'],
                                                                                       [])]},
 'tasks/triton2triton/vllm/triton_wy_fast_recompute_wu/config.yaml': {'after_sha256': '897ba21b4c27ade847aa27d9cd09973c52d907734c65fb6f3d9cfb267d8aca65',
                                                                      'before_sha256': 'bdcd2bcf65cf8406df3f9d6f196874d203e1866c3e111f9890dad55fa4be96e6',
                                                                      'reverse_edits': [(6,
                                                                                         11,
                                                                                         ['  - path: '
                                                                                          'source/triton_wy_fast_recompute_wu.py\n',
                                                                                          '    scope: '
                                                                                          'symbols\n',
                                                                                          '    symbols:\n',
                                                                                          '    - '
                                                                                          'recompute_w_u_fwd_kernel\n',
                                                                                          '    '
                                                                                          'allow_new_helpers: '
                                                                                          'true\n'],
                                                                                         ['  - '
                                                                                          'source/triton_wy_fast_recompute_wu.py\n'])]}}

ATTENTION_SAMPLING_COVERAGE_FILES = {'tasks/triton2triton/vllm/triton_correct_attn_cp_out/README.md': {'after_sha256': '0b874a9c778ced794e4e258f145d03ba70f9e2b1437e4eca836fd1ddc1378dd5',
                                                                   'before_sha256': 'dc3283ad5b4a43e8c46c26a7a20f6123854ed2f68cc8dcec88ab0fb97537757f',
                                                                   'reverse_edits': [(63,
                                                                                      75,
                                                                                      ['\n',
                                                                                       '### Additional '
                                                                                       'unscored interface '
                                                                                       'coverage\n',
                                                                                       '\n',
                                                                                       'Three explicit '
                                                                                       'correctness-only '
                                                                                       'rows exercise base-2 '
                                                                                       'exponent/logarithm '
                                                                                       'correction,\n',
                                                                                       'nonzero current-rank '
                                                                                       'indices, and both '
                                                                                       'accepted '
                                                                                       'four-dimensional LSE '
                                                                                       'layouts\n',
                                                                                       '`[N,B,H,1]` / '
                                                                                       '`[N,1,B,H]` with '
                                                                                       '`[B,1,H,D]` output '
                                                                                       'inputs. Returned '
                                                                                       'corrected output\n',
                                                                                       'and global LSE '
                                                                                       'retain normalized '
                                                                                       'shapes `[B,H,D]` / '
                                                                                       '`[B,H]`. Each real '
                                                                                       'wrapper call\n',
                                                                                       'uses the protected '
                                                                                       'pristine reference, '
                                                                                       'full-output checks, '
                                                                                       'immutable inputs '
                                                                                       'and\n',
                                                                                       'Triton dispatch '
                                                                                       'audit. Manifest '
                                                                                       'parameters are '
                                                                                       'matched to the '
                                                                                       'actual generator.\n',
                                                                                       'These are finite '
                                                                                       'representative '
                                                                                       'controls, not an '
                                                                                       'arbitrary-layout '
                                                                                       'guarantee.\n',
                                                                                       'All five original '
                                                                                       'scored shapes, '
                                                                                       'seeds, values, '
                                                                                       'tolerances, ten '
                                                                                       'warmups, 100 '
                                                                                       'samples,\n',
                                                                                       'reference and timing '
                                                                                       'code are unchanged; '
                                                                                       'the three new rows '
                                                                                       'do not affect '
                                                                                       'scoring.\n'],
                                                                                      [])]},
 'tasks/triton2triton/vllm/triton_correct_attn_cp_out/workloads.json': {'after_sha256': 'a456cd982e5a9327d077d2b6c54fcb2d03488eddf5828d1cfe0adebfe08d0e52',
                                                                        'before_sha256': 'c3dbea0d0cf60a99c050e4795a8ab4d10411ec2f8ea01e5e59e3ea50155502a1',
                                                                        'reverse_edits': [(138,
                                                                                           271,
                                                                                           ['    },\n',
                                                                                            '    {\n',
                                                                                            '      '
                                                                                            '"test_case_id": '
                                                                                            '"base2_nonzero_rank",\n',
                                                                                            '      "dtype": '
                                                                                            '"float16,float32",\n',
                                                                                            '      "params": '
                                                                                            '{\n',
                                                                                            '        '
                                                                                            '"contract_case": '
                                                                                            '"base2_nonzero_rank",\n',
                                                                                            '        "B": '
                                                                                            '2,\n',
                                                                                            '        "H": '
                                                                                            '3,\n',
                                                                                            '        "D": '
                                                                                            '64,\n',
                                                                                            '        "N": '
                                                                                            '4,\n',
                                                                                            '        '
                                                                                            '"lse_idx": 2,\n',
                                                                                            '        '
                                                                                            '"is_base_e": '
                                                                                            'false,\n',
                                                                                            '        '
                                                                                            '"out_layout": '
                                                                                            '"BHD",\n',
                                                                                            '        '
                                                                                            '"lse_layout": '
                                                                                            '"NBH",\n',
                                                                                            '        "seed": '
                                                                                            '1901,\n',
                                                                                            '        '
                                                                                            '"input_shapes": '
                                                                                            '{\n',
                                                                                            '          '
                                                                                            '"out": [\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            '
                                                                                            '3,\n',
                                                                                            '            '
                                                                                            '64\n',
                                                                                            '          ],\n',
                                                                                            '          '
                                                                                            '"lses": [\n',
                                                                                            '            '
                                                                                            '4,\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            3\n',
                                                                                            '          ]\n',
                                                                                            '        },\n',
                                                                                            '        '
                                                                                            '"output_shapes": '
                                                                                            '{\n',
                                                                                            '          '
                                                                                            '"corrected": '
                                                                                            '[\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            '
                                                                                            '3,\n',
                                                                                            '            '
                                                                                            '64\n',
                                                                                            '          ],\n',
                                                                                            '          '
                                                                                            '"final_lse": '
                                                                                            '[\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            3\n',
                                                                                            '          ]\n',
                                                                                            '        }\n',
                                                                                            '      },\n',
                                                                                            '      "checks": '
                                                                                            '[\n',
                                                                                            '        '
                                                                                            '"correctness"\n',
                                                                                            '      ],\n',
                                                                                            '      "status": '
                                                                                            '"PASS"\n',
                                                                                            '    },\n',
                                                                                            '    {\n',
                                                                                            '      '
                                                                                            '"test_case_id": '
                                                                                            '"basee_4d_trailing_lse",\n',
                                                                                            '      "dtype": '
                                                                                            '"float16,float32",\n',
                                                                                            '      "params": '
                                                                                            '{\n',
                                                                                            '        '
                                                                                            '"contract_case": '
                                                                                            '"basee_4d_trailing_lse",\n',
                                                                                            '        "B": '
                                                                                            '2,\n',
                                                                                            '        "H": '
                                                                                            '3,\n',
                                                                                            '        "D": '
                                                                                            '64,\n',
                                                                                            '        "N": '
                                                                                            '4,\n',
                                                                                            '        '
                                                                                            '"lse_idx": 3,\n',
                                                                                            '        '
                                                                                            '"is_base_e": '
                                                                                            'true,\n',
                                                                                            '        '
                                                                                            '"out_layout": '
                                                                                            '"B1HD",\n',
                                                                                            '        '
                                                                                            '"lse_layout": '
                                                                                            '"NBH1",\n',
                                                                                            '        "seed": '
                                                                                            '1902,\n',
                                                                                            '        '
                                                                                            '"input_shapes": '
                                                                                            '{\n',
                                                                                            '          '
                                                                                            '"out": [\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            '
                                                                                            '1,\n',
                                                                                            '            '
                                                                                            '3,\n',
                                                                                            '            '
                                                                                            '64\n',
                                                                                            '          ],\n',
                                                                                            '          '
                                                                                            '"lses": [\n',
                                                                                            '            '
                                                                                            '4,\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            '
                                                                                            '3,\n',
                                                                                            '            1\n',
                                                                                            '          ]\n',
                                                                                            '        },\n',
                                                                                            '        '
                                                                                            '"output_shapes": '
                                                                                            '{\n',
                                                                                            '          '
                                                                                            '"corrected": '
                                                                                            '[\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            '
                                                                                            '3,\n',
                                                                                            '            '
                                                                                            '64\n',
                                                                                            '          ],\n',
                                                                                            '          '
                                                                                            '"final_lse": '
                                                                                            '[\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            3\n',
                                                                                            '          ]\n',
                                                                                            '        }\n',
                                                                                            '      },\n',
                                                                                            '      "checks": '
                                                                                            '[\n',
                                                                                            '        '
                                                                                            '"correctness"\n',
                                                                                            '      ],\n',
                                                                                            '      "status": '
                                                                                            '"PASS"\n',
                                                                                            '    },\n',
                                                                                            '    {\n',
                                                                                            '      '
                                                                                            '"test_case_id": '
                                                                                            '"base2_4d_middle_lse",\n',
                                                                                            '      "dtype": '
                                                                                            '"float16,float32",\n',
                                                                                            '      "params": '
                                                                                            '{\n',
                                                                                            '        '
                                                                                            '"contract_case": '
                                                                                            '"base2_4d_middle_lse",\n',
                                                                                            '        "B": '
                                                                                            '2,\n',
                                                                                            '        "H": '
                                                                                            '3,\n',
                                                                                            '        "D": '
                                                                                            '64,\n',
                                                                                            '        "N": '
                                                                                            '2,\n',
                                                                                            '        '
                                                                                            '"lse_idx": 1,\n',
                                                                                            '        '
                                                                                            '"is_base_e": '
                                                                                            'false,\n',
                                                                                            '        '
                                                                                            '"out_layout": '
                                                                                            '"B1HD",\n',
                                                                                            '        '
                                                                                            '"lse_layout": '
                                                                                            '"N1BH",\n',
                                                                                            '        "seed": '
                                                                                            '1903,\n',
                                                                                            '        '
                                                                                            '"input_shapes": '
                                                                                            '{\n',
                                                                                            '          '
                                                                                            '"out": [\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            '
                                                                                            '1,\n',
                                                                                            '            '
                                                                                            '3,\n',
                                                                                            '            '
                                                                                            '64\n',
                                                                                            '          ],\n',
                                                                                            '          '
                                                                                            '"lses": [\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            '
                                                                                            '1,\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            3\n',
                                                                                            '          ]\n',
                                                                                            '        },\n',
                                                                                            '        '
                                                                                            '"output_shapes": '
                                                                                            '{\n',
                                                                                            '          '
                                                                                            '"corrected": '
                                                                                            '[\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            '
                                                                                            '3,\n',
                                                                                            '            '
                                                                                            '64\n',
                                                                                            '          ],\n',
                                                                                            '          '
                                                                                            '"final_lse": '
                                                                                            '[\n',
                                                                                            '            '
                                                                                            '2,\n',
                                                                                            '            3\n',
                                                                                            '          ]\n',
                                                                                            '        }\n',
                                                                                            '      },\n',
                                                                                            '      "checks": '
                                                                                            '[\n',
                                                                                            '        '
                                                                                            '"correctness"\n',
                                                                                            '      ],\n',
                                                                                            '      "status": '
                                                                                            '"PASS"\n'],
                                                                                           [])]},
 'tasks/triton2triton/vllm/triton_decode_attn_grouped_stage1/README.md': {'after_sha256': '90d757c895f403e857760fc4bd129ac04bf90dd30444ac359d959bffb0dcd9af',
                                                                          'before_sha256': '0f2a551ce51340a3709600bc78d91b0acc2d2205c10756e4dfe45d5b6d380d7d',
                                                                          'reverse_edits': [(98,
                                                                                             113,
                                                                                             ['\n',
                                                                                              '### Group '
                                                                                              'boundaries '
                                                                                              'crossing a '
                                                                                              '16-head '
                                                                                              'tile\n',
                                                                                              '\n',
                                                                                              'Two '
                                                                                              'additional '
                                                                                              'correctness-only '
                                                                                              'controls use '
                                                                                              '2 KV heads '
                                                                                              'with 34 or 48 '
                                                                                              'query heads\n',
                                                                                              '(group sizes '
                                                                                              '17 and 24), '
                                                                                              'ragged '
                                                                                              'lengths 63/37 '
                                                                                              'and '
                                                                                              'nonidentity '
                                                                                              'page routing. '
                                                                                              'The\n',
                                                                                              'second also '
                                                                                              'uses logit '
                                                                                              'capping. They '
                                                                                              'reuse the '
                                                                                              'existing '
                                                                                              'independent '
                                                                                              'FP32 '
                                                                                              'reference,\n',
                                                                                              'full '
                                                                                              'active-output/unchanged '
                                                                                              'inactive-slot '
                                                                                              'checks and '
                                                                                              'unscored '
                                                                                              'graph '
                                                                                              'replay.\n',
                                                                                              'The launch '
                                                                                              'grid assigns '
                                                                                              '`ceil(group_size/16)` '
                                                                                              'blocks to '
                                                                                              'each KV head '
                                                                                              'independently;\n',
                                                                                              'query-head '
                                                                                              'offsets start '
                                                                                              'at that KV '
                                                                                              "group's base "
                                                                                              'and the final '
                                                                                              'block is '
                                                                                              'masked at\n',
                                                                                              'the group '
                                                                                              'boundary. '
                                                                                              'This prevents '
                                                                                              'cross-group '
                                                                                              'routing when '
                                                                                              'a group needs '
                                                                                              'a partial\n',
                                                                                              'second block. '
                                                                                              'The original '
                                                                                              'group sizes '
                                                                                              '1/4/8 have '
                                                                                              'exactly the '
                                                                                              'same launch '
                                                                                              'grid and\n',
                                                                                              'head '
                                                                                              'assignments '
                                                                                              'as before. '
                                                                                              'All original '
                                                                                              'five scored '
                                                                                              'cases and the '
                                                                                              'three '
                                                                                              'earlier\n',
                                                                                              'controls '
                                                                                              'remain, along '
                                                                                              'with seeds, '
                                                                                              'inputs, '
                                                                                              'tolerances, '
                                                                                              'ten warmups '
                                                                                              'and 100 '
                                                                                              'samples.\n',
                                                                                              'No score '
                                                                                              'weight or '
                                                                                              'timing '
                                                                                              'boundary '
                                                                                              'changes. '
                                                                                              'Kernel source '
                                                                                              'changes '
                                                                                              'require '
                                                                                              'fresh\n',
                                                                                              'baseline/candidate '
                                                                                              'measurements; '
                                                                                              'historical '
                                                                                              'GPU '
                                                                                              'qualification '
                                                                                              'does not '
                                                                                              'transfer.\n'],
                                                                                             [])]},
 'tasks/triton2triton/vllm/triton_decode_attn_grouped_stage1/source/_fwd_grouped_kernel_stage1.py': {'after_sha256': 'edd7803c83a25802b72386b774651218aedb6bdd6880d45f270ac4d49527ae12',
                                                                                                     'before_sha256': '3f0d728ac011a7b9bd5eee1ae9d3eb69ef1ad91367918c4660c05a743cac7e4f',
                                                                                                     'reverse_edits': [(53,
                                                                                                                        56,
                                                                                                                        ['    '
                                                                                                                         'head_blocks_per_kv: '
                                                                                                                         'tl.constexpr '
                                                                                                                         '= '
                                                                                                                         'tl.cdiv(kv_group_num, '
                                                                                                                         'BLOCK_H)\n',
                                                                                                                         '    '
                                                                                                                         'cur_kv_head '
                                                                                                                         '= '
                                                                                                                         'cur_head_id '
                                                                                                                         '// '
                                                                                                                         'head_blocks_per_kv\n',
                                                                                                                         '    '
                                                                                                                         'head_block_in_group '
                                                                                                                         '= '
                                                                                                                         'cur_head_id '
                                                                                                                         '% '
                                                                                                                         'head_blocks_per_kv\n'],
                                                                                                                        ['    '
                                                                                                                         'cur_kv_head '
                                                                                                                         '= '
                                                                                                                         'cur_head_id '
                                                                                                                         '// '
                                                                                                                         'tl.cdiv(kv_group_num, '
                                                                                                                         'BLOCK_H)\n']),
                                                                                                                       (58,
                                                                                                                        60,
                                                                                                                        ['    '
                                                                                                                         'cur_head '
                                                                                                                         '= '
                                                                                                                         'cur_kv_head '
                                                                                                                         '* '
                                                                                                                         'kv_group_num '
                                                                                                                         '+ '
                                                                                                                         'head_block_in_group '
                                                                                                                         '* '
                                                                                                                         'BLOCK_H '
                                                                                                                         '+ '
                                                                                                                         'tl.arange(0, '
                                                                                                                         'BLOCK_H)\n',
                                                                                                                         '    '
                                                                                                                         'mask_h '
                                                                                                                         '= '
                                                                                                                         'cur_head '
                                                                                                                         '< '
                                                                                                                         '(cur_kv_head '
                                                                                                                         '+ '
                                                                                                                         '1) '
                                                                                                                         '* '
                                                                                                                         'kv_group_num\n'],
                                                                                                                        ['    '
                                                                                                                         'VALID_BLOCK_H: '
                                                                                                                         'tl.constexpr '
                                                                                                                         '= '
                                                                                                                         'BLOCK_H '
                                                                                                                         'if '
                                                                                                                         'kv_group_num '
                                                                                                                         '> '
                                                                                                                         'BLOCK_H '
                                                                                                                         'else '
                                                                                                                         'kv_group_num\n',
                                                                                                                         '    '
                                                                                                                         'cur_head '
                                                                                                                         '= '
                                                                                                                         'cur_head_id '
                                                                                                                         '* '
                                                                                                                         'VALID_BLOCK_H '
                                                                                                                         '+ '
                                                                                                                         'tl.arange(0, '
                                                                                                                         'BLOCK_H)\n',
                                                                                                                         '    '
                                                                                                                         'mask_h '
                                                                                                                         '= '
                                                                                                                         'cur_head '
                                                                                                                         '< '
                                                                                                                         '(cur_head_id '
                                                                                                                         '+ '
                                                                                                                         '1) '
                                                                                                                         '* '
                                                                                                                         'VALID_BLOCK_H\n']),
                                                                                                                       (222,
                                                                                                                        223,
                                                                                                                        ['        '
                                                                                                                         'k_buffer.shape[-2] '
                                                                                                                         '* '
                                                                                                                         'triton.cdiv(kv_group_num, '
                                                                                                                         'BLOCK_H),\n'],
                                                                                                                        ['        '
                                                                                                                         'triton.cdiv(head_num, '
                                                                                                                         'min(BLOCK_H, '
                                                                                                                         'kv_group_num)),\n'])]},
 'tasks/triton2triton/vllm/triton_rejection_sample/README.md': {'after_sha256': 'a44594d5c324187274f560d787fb2d30532ba2c396f10c3d0301a8ad203fe0b7',
                                                                'before_sha256': '28eb4b42d001e80476767edbf85efc79768418095ed8181e4dde26576a8e8bb7',
                                                                'reverse_edits': [(48,
                                                                                   62,
                                                                                   ['\n',
                                                                                    '### Ragged '
                                                                                    'speculative-decoding '
                                                                                    'offsets\n',
                                                                                    '\n',
                                                                                    'Two additional '
                                                                                    'correctness-only rows '
                                                                                    'use per-request '
                                                                                    'positive spans '
                                                                                    '`[1,3,5,2]`\n',
                                                                                    'and `[4,2,6,1]` in '
                                                                                    'cumulative logits '
                                                                                    'offsets. The output '
                                                                                    'capacity stays '
                                                                                    'uniform\n',
                                                                                    'while each active '
                                                                                    'request may have a '
                                                                                    'different number of '
                                                                                    'draft/target logits.\n',
                                                                                    'They cover bonus-only '
                                                                                    'requests, full '
                                                                                    'acceptance, and '
                                                                                    'early/late rejection. '
                                                                                    'In\n',
                                                                                    'addition to the '
                                                                                    'retained protected '
                                                                                    'reference and read-only '
                                                                                    'checks, an '
                                                                                    'independent\n',
                                                                                    'scalar oracle selects '
                                                                                    'the first mismatch and '
                                                                                    'checks every accepted '
                                                                                    'prefix and\n',
                                                                                    'exact count. Undefined '
                                                                                    'unused output tails '
                                                                                    'remain unchecked. The '
                                                                                    'original five\n',
                                                                                    'scored cases and '
                                                                                    'earlier controls, '
                                                                                    'kernel, seeds, integer '
                                                                                    'comparison, ten '
                                                                                    'warmups,\n',
                                                                                    '100 samples and timer '
                                                                                    'are unchanged. These '
                                                                                    'added cases do not '
                                                                                    'contribute a score.\n',
                                                                                    'A zero speculative-step '
                                                                                    'request has one bonus '
                                                                                    'logit; no zero-logit '
                                                                                    'request is\n',
                                                                                    'introduced by these '
                                                                                    'active-request '
                                                                                    'controls.\n'],
                                                                                   [])]}}


def _exact_reviewed_file_inverse(text, record):
    assert hashlib.sha256(text.encode()).hexdigest() == record['after_sha256'], 'Unreviewed current file bytes'
    lines = text.splitlines(keepends=True)
    for start, end, expected, replacement in reversed(record['reverse_edits']):
        assert lines[start:end] == expected
        lines[start:end] = replacement
    restored = ''.join(lines)
    assert hashlib.sha256(restored.encode()).hexdigest() == record['before_sha256']
    return restored


def historical_attention_sampling_file(relative, text):
    record = ATTENTION_SAMPLING_COVERAGE_FILES.get(relative)
    return _exact_reviewed_file_inverse(text, record) if record is not None else text


def historical_triton_device_boundary_file(relative, text):
    record = TRITON_DEVICE_BOUNDARY_FILES.get(relative)
    if record is None:
        return text
    # Only three later coverage-doc appendices are composed before the exact
    # source20 boundary inverse. Both complete versions are individually pinned.
    coverage = ATTENTION_SAMPLING_COVERAGE_FILES.get(relative)
    if coverage is not None and hashlib.sha256(text.encode()).hexdigest() == coverage['after_sha256']:
        text = _exact_reviewed_file_inverse(text, coverage)
    return _exact_reviewed_file_inverse(text, record)
