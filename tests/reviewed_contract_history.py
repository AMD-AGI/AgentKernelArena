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
# Only contract files are pinned; README text is documentation and may change.
TRITON_DEVICE_BOUNDARY_FILES = {'tasks/triton2triton/vllm/triton_apply_grammar_bitmask/config.yaml': {'after_sha256': 'fb54d74399b7e25da2e50e4ad8a7f6a95d0cbc8f0ea76d61b7b6b10567d5028e',
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

ATTENTION_SAMPLING_COVERAGE_FILES = {'tasks/triton2triton/vllm/triton_correct_attn_cp_out/workloads.json': {'after_sha256': 'a456cd982e5a9327d077d2b6c54fcb2d03488eddf5828d1cfe0adebfe08d0e52',
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
                                                                                                                         'kv_group_num)),\n'])]}}


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


# Tunable launch configurations: each record moves a wrapper's tuning values into
# its editable `_launch_config` helper and adds the protected validator. Full
# current/before bytes bound every inverse; the boundary inverse applies after it.
TRITON_LAUNCH_CONFIG_FILES = {'tasks/triton2triton/vllm/triton_bmm/config.yaml': {'after_sha256': 'd67c2742d3e92815508850fb6ff378fcfb60960f0d8075e98ea0585a64617d0f',
                                                     'before_sha256': '6e04236bbc34f35a60c1dcaaa47f218ac2e42d75a4cf85778240feaa8088462e',
                                                     'reverse_edits': [(10,
                                                                        11,
                                                                        ['    - _launch_config\n'],
                                                                        []),
                                                                       (12,
                                                                        14,
                                                                        ['    configuration_symbols:\n',
                                                                         '    - _launch_config\n'],
                                                                        [])]},
 'tasks/triton2triton/vllm/triton_bmm/source/triton_bmm.py': {'after_sha256': '6a072c2f033459d042d0662fb5ed58b43a2d8f8c1c65f218d0be2bfa5f740c06',
                                                              'before_sha256': 'cb7ef5165d6e7adab326435a534adf667eeeda98d497bf5f86727b15c8ef7feb',
                                                              'reverse_edits': [(103,
                                                                                 141,
                                                                                 ['def '
                                                                                  '_launch_config(dtype_name, '
                                                                                  'batch, m, n, k):\n',
                                                                                  '    """Return the '
                                                                                  'bmm_kernel tile sizes, '
                                                                                  'num_warps and '
                                                                                  'num_stages.\n',
                                                                                  '\n',
                                                                                  '    Tune these values, '
                                                                                  'optionally per dtype or '
                                                                                  'problem size. The '
                                                                                  'protected\n',
                                                                                  '    wrapper validates '
                                                                                  'them and derives the grid '
                                                                                  'from the tile sizes.\n',
                                                                                  '    """\n',
                                                                                  '    if dtype_name == '
                                                                                  '"float32":\n',
                                                                                  '        return '
                                                                                  '{"BLOCK_SIZE_M": 64, '
                                                                                  '"BLOCK_SIZE_N": 64, '
                                                                                  '"BLOCK_SIZE_K": 32,\n',
                                                                                  '                '
                                                                                  '"num_stages": 2, '
                                                                                  '"num_warps": 4}\n',
                                                                                  '    return '
                                                                                  '{"BLOCK_SIZE_M": 128, '
                                                                                  '"BLOCK_SIZE_N": 128, '
                                                                                  '"BLOCK_SIZE_K": 32,\n',
                                                                                  '            "num_stages": '
                                                                                  '2, "num_warps": 8}\n',
                                                                                  '\n',
                                                                                  '\n',
                                                                                  '_DTYPE_NAMES = '
                                                                                  '{torch.bfloat16: '
                                                                                  '"bfloat16", '
                                                                                  'torch.float16: "float16", '
                                                                                  'torch.float32: '
                                                                                  '"float32"}\n',
                                                                                  '_TILE_KEYS = '
                                                                                  '("BLOCK_SIZE_M", '
                                                                                  '"BLOCK_SIZE_N", '
                                                                                  '"BLOCK_SIZE_K")\n',
                                                                                  '\n',
                                                                                  '\n',
                                                                                  'def '
                                                                                  '_checked_launch_config(cfg):\n',
                                                                                  '    """Accept only tile '
                                                                                  'sizes, num_warps and '
                                                                                  'num_stages within '
                                                                                  'supported bounds."""\n',
                                                                                  '    expected = '
                                                                                  'set(_TILE_KEYS) | '
                                                                                  '{"num_warps", '
                                                                                  '"num_stages"}\n',
                                                                                  '    if not '
                                                                                  'isinstance(cfg, dict) or '
                                                                                  'set(cfg) != expected:\n',
                                                                                  '        raise '
                                                                                  'ValueError(f"Launch '
                                                                                  'configuration must set '
                                                                                  'exactly '
                                                                                  '{sorted(expected)}")\n',
                                                                                  '    checked = {}\n',
                                                                                  '    for key in '
                                                                                  'sorted(expected):\n',
                                                                                  '        value = '
                                                                                  'cfg[key]\n',
                                                                                  '        if type(value) is '
                                                                                  'not int:\n',
                                                                                  '            raise '
                                                                                  'ValueError(f"Launch '
                                                                                  'setting {key} must be an '
                                                                                  'int")\n',
                                                                                  '        checked[key] = '
                                                                                  'value\n',
                                                                                  '    for key in '
                                                                                  '_TILE_KEYS:\n',
                                                                                  '        if not (16 <= '
                                                                                  'checked[key] <= 256 and '
                                                                                  'checked[key] & '
                                                                                  '(checked[key] - 1) == '
                                                                                  '0):\n',
                                                                                  '            raise '
                                                                                  'ValueError(f"{key} must '
                                                                                  'be a power of two in [16, '
                                                                                  '256]")\n',
                                                                                  '    if '
                                                                                  'checked["num_warps"] not '
                                                                                  'in (1, 2, 4, 8, 16):\n',
                                                                                  '        raise '
                                                                                  'ValueError("num_warps '
                                                                                  'must be 1, 2, 4, 8 or '
                                                                                  '16")\n',
                                                                                  '    if not 1 <= '
                                                                                  'checked["num_stages"] <= '
                                                                                  '4:\n',
                                                                                  '        raise '
                                                                                  'ValueError("num_stages '
                                                                                  'must be in [1, 4]")\n',
                                                                                  '    return checked\n',
                                                                                  '\n',
                                                                                  '\n'],
                                                                                 []),
                                                                                (154,
                                                                                 155,
                                                                                 ['    cfg = '
                                                                                  '_checked_launch_config(_launch_config(_DTYPE_NAMES[dtype], '
                                                                                  'B_size, M, N, K))\n'],
                                                                                 ['    configs = {\n',
                                                                                  '        torch.bfloat16: '
                                                                                  '{\n',
                                                                                  '            '
                                                                                  '"BLOCK_SIZE_M": 128,\n',
                                                                                  '            '
                                                                                  '"BLOCK_SIZE_N": 128,\n',
                                                                                  '            '
                                                                                  '"BLOCK_SIZE_K": 32,\n',
                                                                                  '            "num_stages": '
                                                                                  '2,\n',
                                                                                  '            "num_warps": '
                                                                                  '8,\n',
                                                                                  '        },\n',
                                                                                  '        torch.float16: '
                                                                                  '{\n',
                                                                                  '            '
                                                                                  '"BLOCK_SIZE_M": 128,\n',
                                                                                  '            '
                                                                                  '"BLOCK_SIZE_N": 128,\n',
                                                                                  '            '
                                                                                  '"BLOCK_SIZE_K": 32,\n',
                                                                                  '            "num_stages": '
                                                                                  '2,\n',
                                                                                  '            "num_warps": '
                                                                                  '8,\n',
                                                                                  '        },\n',
                                                                                  '        torch.float32: '
                                                                                  '{\n',
                                                                                  '            '
                                                                                  '"BLOCK_SIZE_M": 64,\n',
                                                                                  '            '
                                                                                  '"BLOCK_SIZE_N": 64,\n',
                                                                                  '            '
                                                                                  '"BLOCK_SIZE_K": 32,\n',
                                                                                  '            "num_stages": '
                                                                                  '2,\n',
                                                                                  '            "num_warps": '
                                                                                  '4,\n',
                                                                                  '        },\n',
                                                                                  '    }\n',
                                                                                  '\n',
                                                                                  '    cfg = '
                                                                                  'configs[dtype]\n'])]}}


def historical_triton_launch_config_file(relative, text):
    record = TRITON_LAUNCH_CONFIG_FILES.get(relative)
    return _exact_reviewed_file_inverse(text, record) if record is not None else text


def historical_triton_device_boundary_file(relative, text):
    text = historical_triton_launch_config_file(relative, text)
    record = TRITON_DEVICE_BOUNDARY_FILES.get(relative)
    return _exact_reviewed_file_inverse(text, record) if record is not None else text
