"""CPU contract/oracle and measured-replay negative controls for 20 token tasks.

These tests do not claim GPU compilation, graph timing or semantic validator PASS.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import hashlib
from pr107_integration_helpers import original_manifest
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'tasks/triton2triton/vllm'
NAMES = ('bad_words combine_sampled_and_draft_tokens logit_bias min_p pack_seq penalties '
         'post_update prepare_eagle_docode prepare_eagle_inputs prepare_mrope_positions '
         'prepare_pos_seq_lens prepare_prefill_inputs prompt_logprobs_token_ids ranks '
         'rejection_greedy_sample rejection_sample temperature topk_topp update_eagle_inputs '
         'write_zeros_to_output').split()


# Frozen original contracts: canonical JSON uses sorted keys and compact separators.
# No Git history, network, GPU or optional checkout is needed to verify these.
ORIGINAL_CONTRACTS = {'bad_words': {'original_count': 8,
               'rows_sha256': '352cd350659fb16e24922fd4870227d717f65e1eece073ecf845cf88f88c47d5',
               'manifest_metadata_sha256': 'c4c931178226487a52691cc6108bdabc72174ff5dd52825c0bcfba02e316bb8e',
               'candidate_sha256': '6ab1618f2a6442d0f743b27b69ba7be715901e5a6efe619d65aeba76553c1a92',
               'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'combine_sampled_and_draft_tokens': {'original_count': 5,
                                      'rows_sha256': '6dfd6535b70867cfa63db7c893d668404413f24754da893385f2e2488b1689e5',
                                      'manifest_metadata_sha256': '4900d05033948baeaf3888d4c272a4a78a2866f0137d187fd54353a3b3f2532c',
                                      'candidate_sha256': 'ffa5bdf178be5da5c46f1070852fef09d9a6fa704d11390c61013f6725a97a49',
                                      'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'logit_bias': {'original_count': 6,
                'rows_sha256': 'b8be11fad30df931464925a803bd1f46b5e2a139c3d1acd3825061abba8e9369',
                'manifest_metadata_sha256': '40d12c2f9d77fa4a457565b9b9aa1b4992f3ae6d07419f586bf720b8991c10c8',
                'candidate_sha256': '9c28b1d35e40914bb4173db80a5311f32ad3fa1f08f91e962f2f2f402bca8e18',
                'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'min_p': {'original_count': 5,
           'rows_sha256': '9cefadd8f5accf2901ae43bc6a826dce2ea2cb57a1daf505086bfbda7052edb1',
           'manifest_metadata_sha256': 'cd82fbbffcb9503e950d8a80b0fc6f59aa6667c5211d749328a5f3fc779478f7',
           'candidate_sha256': '9a75bcdeec87702035d4795bd1d5503b06cf9e8f8c68546fd2906903f07e9ae0',
           'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'pack_seq': {'original_count': 5,
              'rows_sha256': '8bc24c65b8314726cc01eece5c4373138c8fbc0fd20fcc20f419e412fca36a47',
              'manifest_metadata_sha256': 'f23322bb608fb19f20e11cbcde1428d993b13ac0dd720ad7b1e8f14c453094d2',
              'candidate_sha256': 'f1a7c27e2adc099b7679a744776f20d5640a4875a72b75ab072d7685d5c6bd25',
              'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'penalties': {'original_count': 5,
               'rows_sha256': '15de59163343854857802e2342950e5b3a70e49c360c39e063d88f25b9d38f0b',
               'manifest_metadata_sha256': '6274d72fd75a0c01e68b386a11abb93c770d1d43584ebd190e4dd3cae8871b61',
               'candidate_sha256': '5657440ea29a80bdbded7efbda2b1c1a8e38f4ddca28f8631d595edffd6320cb',
               'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'post_update': {'original_count': 5,
                 'rows_sha256': '2e3fbc32c5c66cf9b68965ae195606f5ee3023f2542818aafdc263506bf39ea9',
                 'manifest_metadata_sha256': '7e63fc45c421a9571d31879b85ce9b11c8b7257dd70984162e56e437cfd2d114',
                 'candidate_sha256': '75621ac47d17889f933904052c4c70b1e820a5c363d4d15699c7b22ccf9c2177',
                 'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'prepare_eagle_docode': {'original_count': 5,
                          'rows_sha256': 'e6ef53399595a421d76b8583967236c68e5de515a88bc18b916d1c05d7277fe8',
                          'manifest_metadata_sha256': '06d91f29d051b9faf647b783fa04e999be9e0f38411a7ec92017ced3ed35dae4',
                          'candidate_sha256': '98804483f8b91e03ea25875adb94e8a3c57b37a5d5cdcc25555755c9bcb31ed2',
                          'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'prepare_eagle_inputs': {'original_count': 5,
                          'rows_sha256': '27b8501be696cb8eb7d3e479ab7230448e69b6a1a7984ae0467bc09f5e5609a2',
                          'manifest_metadata_sha256': 'b4669c27753786f2a61d7703e2dac15ee3d5ce7783ed32bfa00f6d972817b47a',
                          'candidate_sha256': 'db37af5fdc4b2196c7db61a98e1242cc156976525e2a62dd84cd9819ac63b33d',
                          'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'prepare_mrope_positions': {'original_count': 5,
                             'rows_sha256': 'f57554fa1d6c57692630bd9eced907ea41579e34f8f35b36a426d296ec52c9ea',
                             'manifest_metadata_sha256': '36073ccea48b60bfce6bc9ab3493ff6e10f88904e3a3ff915bd6d9650db6fdd7',
                             'candidate_sha256': '09a53d4552665691c4c4a4218b37ed6df40441621c2196ddf57fa2b2305dd5e3',
                             'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'prepare_pos_seq_lens': {'original_count': 5,
                          'rows_sha256': '6b3860c5892bbbe6e3553cc9721f2454506aa0795d00eb6cfba344cb73df6904',
                          'manifest_metadata_sha256': '959f075b42ea8c7b0b1b7fcc510706312aa632cd00a231a8b379512882dc2fac',
                          'candidate_sha256': '52895e4c698546b43ab3ff03319265879d848e2ce204550d99c615327f7c7948',
                          'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'prepare_prefill_inputs': {'original_count': 5,
                            'rows_sha256': '2f7bbcca2786d13658414b0c4750c4136417b52a9bc52409b8b7b0e933b9647c',
                            'manifest_metadata_sha256': 'a1209e765db83d48ea74be8c13cf60b8935e70e7227f5dc69d400433f24e5a2f',
                            'candidate_sha256': '24a2d51c3285d7b7aedd6d1966b9f6e71843b905a7c3358198ac4c976eea41e2',
                            'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'prompt_logprobs_token_ids': {'original_count': 5,
                               'rows_sha256': '181a6905fc0b8e8351a4cf2596e2c6e1686e116d4a3e29ea22573fd3cf11b57c',
                               'manifest_metadata_sha256': '7cdec0d1941efde754f5ec5586edc68001c2fc2ef330a902d6a74eb58a93b626',
                               'candidate_sha256': '756423eb1515135a96096f7497c5b00e3d70415f74a6366d3b07d4d38cb01aaa',
                               'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'ranks': {'original_count': 5,
           'rows_sha256': '576b1dc2ac7f6e6466c1caca35bd03161568e7aa9002d4cb9905388225b1e162',
           'manifest_metadata_sha256': 'cec978e1cd06a457de84ca01365837d2696ef155895e36f4c5f53daad3b63f52',
           'candidate_sha256': '959b4f2dec0e1edbda191d31d2f3d91b8d30edeb868268c8edfd3cc50b672695',
           'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'rejection_greedy_sample': {'original_count': 5,
                             'rows_sha256': '905364d72d9bcdecea955354de0ab1a30efbf05ffd84e61bae50509e954535a4',
                             'manifest_metadata_sha256': 'dd8eb513f072ebb93ea42862518bf3e28eb2da64688cad87e6d0c713f39cfe3b',
                             'candidate_sha256': 'ad7b23b903814c2ca4a907a04fdc3a59e332a3b35090841653a275202883dd3e',
                             'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'rejection_sample': {'original_count': 5,
                      'rows_sha256': 'b29b3f5f0c6f6eb9dd82a22980d39ac73047859c0df47fc9b6bbcf215ce461ef',
                      'manifest_metadata_sha256': '6d04bf914463b741324df654fc33cb11d38aeca9b9ce3a02e7d22efbbe19969d',
                      'candidate_sha256': '4b81f3bf429f6f5adb1868ee3a33df6452211109a9a4be2ee1a830c92df16d2d',
                      'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'temperature': {'original_count': 5,
                 'rows_sha256': '234f894ae80db2c3aecf2f900b9db4ed6de0087739b6eb52458563382a5cafe5',
                 'manifest_metadata_sha256': 'e31cc9c64143a54afd23765014b13af68f006feaa2f10fbd3001df0848577032',
                 'candidate_sha256': '3d4f066294b260234b14cccd4a22136d450480c56d7947f89ec3b1f0bc609def',
                 'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'topk_topp': {'original_count': 5,
               'rows_sha256': '521b36a7313a97a185f1635f83b3924b9a6e0e90a333ebac7961c7649ae877c5',
               'manifest_metadata_sha256': '97dc64f0b24d6cee02346cbf371d529747779fabe5330f15229bb3583d8bec69',
               'candidate_sha256': '97415c9da61d643c3f70aeb7c1f20ca9bddb6db930355696b8ae5ba4afa4e1e0',
               'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'update_eagle_inputs': {'original_count': 5,
                         'rows_sha256': 'a7952bf4bec575ad7eee766b248a23077a079ce8f50a3391ba36de0c3585adfe',
                         'manifest_metadata_sha256': '5617c95cd4a66cb93fad6a37f9d4c7f8ce7d0aff72616031c3f0bc97ab47350b',
                         'candidate_sha256': '24d9fb649c84f69c284373635eb7d6bf5990f48442008a234867ebcc8bb3a770',
                         'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'},
 'write_zeros_to_output': {'original_count': 5,
                           'rows_sha256': '24a443eaed795e2cc27cfeba67090b230df059900d59691cc4ce064d639a88d7',
                           'manifest_metadata_sha256': '2dcaf32d1636de23bb6313d8652d9744c71ec3821ad947dbe86485d6c0bfa8b2',
                           'candidate_sha256': '1da8b35a599bfbb0340b2bd25e3b6a791a312a85b186f43bcc7966670ef809e3',
                           'generated_region_sha256': 'fa991aa44ae5fbfae028aaf2a13afd3381faa4f12005e54c8818cb6b1bedf89a'}}



def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


def modules(name):
    task = TASKS / ('triton_' + name)
    return (load(task / '_arena_replay.py', 'replay_' + name),
            load(task / '_arena_contract.py', 'contract_' + name),
            load(task / 'scripts/task_runner.py', 'harness_' + name))


@pytest.mark.parametrize('name', NAMES)
def test_original_cases_and_benchmark_contract_preserved(name):
    task = TASKS / ('triton_' + name)
    manifest = json.loads((task / 'workloads.json').read_text())
    _, contract, harness = modules(name)
    assert manifest['input_table'] == json.loads(json.dumps(harness.TEST_SHAPES))
    controls = [row for row in manifest['cases'] if row['test_case_id'] == 'contract_controls']
    assert len(controls) == 1 and controls[0]['checks'] == ['correctness']
    assert controls[0]['params']['case_index'] == contract.CONTROL_INDEX
    assert harness.WARMUP_ITERATIONS == 10 and harness.BENCHMARK_ITERATIONS == 100
    tree = ast.parse((task / 'source' / ('triton_' + name + '.py')).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == contract.FUNCTION)
    for args in contract.control_inputs(harness):
        assert len(args) == len(function.args.args)
    # Never depend on the Arena repository at task runtime.
    for path in (task / '_arena_contract.py', task / '_arena_replay.py'):
        tree = ast.parse(path.read_text())
        assert not any(isinstance(n, ast.ImportFrom) and (n.module or '').split('.')[0] in ('src', 'agents') for n in ast.walk(tree))


@pytest.mark.parametrize('name', NAMES)
def test_oracles_do_not_mutate_inputs_and_enforce_metadata(name):
    replay, contract, harness = modules(name)
    for args in contract.control_inputs(harness):
        saved = replay.clone(args)
        answer = replay.expected(harness, contract, args)
        replay.unchanged(args, saved, ())
        replay.check(harness, contract, replay.clone(answer), answer, args)
        bad = replay.clone(answer)
        if isinstance(bad, tuple):
            bad = (bad[0].double(), *bad[1:])
        else:
            bad = bad.double()
        with pytest.raises(AssertionError):
            replay.check(harness, contract, bad, answer, args)


def test_small_independent_known_answers():
    expected = {
        'temperature': [[2,-4,6,8],[2,1,-1,3],[2,3,-1,0],[2,3,4,5]],
        'min_p': [[-torch.inf,1,2,3],[0,1,2,3],[-torch.inf,-torch.inf,-torch.inf,3],
                  [-torch.inf,-torch.inf,-torch.inf,3],[-torch.inf,1,2,3]],
        'ranks': [3,4,4],
        'prompt_logprobs_token_ids': [35,3,4,5,22,23],
        'rejection_greedy_sample': [[80,-1,-1,-1],[-1,-1,-1,-1],[10,11,12,82],[20,99,-1,-1]],
        'prepare_mrope_positions': [[94,3,4,5,61,62,-9],[104,3,4,5,71,72,-9],[114,3,4,5,81,82,-9]],
    }
    for name, values in expected.items():
        _, contract, harness = modules(name)
        answer = contract.reference(harness, next(contract.control_inputs(harness)))
        assert torch.equal(answer, torch.tensor(values, dtype=answer.dtype))
    _, contract, harness = modules('combine_sampled_and_draft_tokens')
    out, indices = contract.reference(harness, next(contract.control_inputs(harness)))
    assert out.tolist() == [-9,-9,10,1,12,7,8] and indices.tolist() == [0,2,3,4,5,6]
    _, contract, harness = modules('prepare_pos_seq_lens')
    pos, seq = contract.reference(harness, next(contract.control_inputs(harness)))
    assert pos.tolist() == [4,2,3,4,7,8] and seq.tolist() == [5,5,9,0,0]
    _, contract, harness = modules('prepare_prefill_inputs')
    ids, next_tokens = contract.reference(harness, next(contract.control_inputs(harness)))
    assert ids.tolist() == [34,2,3,4,-9,-9] and next_tokens.tolist() == [81,82,83,35]


def test_bad_word_speculative_prefix_controls():
    _, contract, harness = modules('bad_words')
    answer = contract.reference(harness, next(contract.control_inputs(harness)))
    assert torch.isneginf(answer).nonzero().tolist() == [[1,10],[2,9],[3,8]]


def test_pack_padding_and_copy_known_answers():
    _, contract, harness = modules('pack_seq')
    for args in contract.control_inputs(harness):
        out = contract.reference(harness, args)
        assert out[0,0].tolist() == [0,1,2]
        assert out[1].tolist() == [[3,4,5],[6,7,8],[9,10,11]]
        assert out[2,0].tolist() == [12,13,14]
        assert torch.all(out[0,1:] == args[2]) and torch.all(out[2,1:] == args[2])


def test_post_update_zero_accept_and_routing_known_answer():
    _, contract, harness = modules('post_update')
    nct, last, hist, tokens, length = contract.reference(harness, next(contract.control_inputs(harness)))
    assert nct.tolist() == [4,4,7,6] and last.tolist() == [4,11,8,13]
    assert length.tolist() == [3,3,6,5]
    assert hist.nonzero().tolist() == [[0,4],[2,7],[2,8]]
    assert tokens[0,2] == 4 and tokens[2,4:6].tolist() == [7,8]
    assert torch.all(tokens[3] == -9)  # Zero accepted tokens retain history.


def test_eagle_clamps_copies_and_padding_known_answers():
    for name in ('prepare_eagle_docode', 'update_eagle_inputs'):
        _, contract, harness = modules(name)
        args = next(contract.control_inputs(harness))
        result = contract.reference(harness, args)
        if name == 'prepare_eagle_docode':
            pos, seq, qsl, ids, hs = result
            assert pos[:3].tolist() == [31,31,31] and seq.tolist() == [32,32,32,0,0]
            assert qsl.tolist() == [0,1,2,3,3,3]
            assert torch.equal(hs[:3], args[1][args[2].long()])
        else:
            ids, pos, hs, seq = result
            assert pos.tolist() == [15,15,15] and seq.tolist() == [16,16,16]
            assert torch.equal(hs, args[1])
        assert torch.equal(ids[:3], args[0])
    _, contract, harness = modules('prepare_eagle_inputs')
    args = next(contract.control_inputs(harness))
    last, ids, pos = contract.reference(harness, args)
    assert last.tolist() == [3,6,9]
    assert ids[3] == args[4][4] and ids[6] == args[3][0] and ids[9] == args[4][2]
    assert pos.tolist() == [0,1,2,3,0,1,2,0,0,1,0,0]


def test_penalty_speculative_history_and_disabled_branch():
    _, contract, harness = modules('penalties')
    args = next(contract.control_inputs(harness))
    out = contract.reference(harness,args)
    # Request 0 disables all penalties; request 2 adds speculative token 4.
    torch.testing.assert_close(out[2:4],args[0][2:4],atol=0,rtol=0)
    assert out[0,3].item() == pytest.approx((-1.85*0.8)+1+1)
    assert out[1,4].item() == pytest.approx((-1.0*0.8)+0.5+1)


def test_logit_bias_allows_masks_and_stop_minimum():
    _, contract, harness = modules('logit_bias')
    out = contract.reference(harness,next(contract.control_inputs(harness)))
    assert torch.isfinite(out[0]).nonzero().flatten().tolist() == [0,2]
    assert out[0,2].item() == pytest.approx(2.2)
    assert out[1,1].item() == pytest.approx(1.4)
    assert torch.isneginf(out[2,2])
    assert torch.isfinite(out[3]).nonzero().flatten().tolist() == [1,3]
    assert out[3,3].item() == pytest.approx(1.7)


def test_rejection_defined_prefix_only_and_wrong_count_rejected():
    replay, contract, harness = modules('rejection_sample')
    args = next(contract.control_inputs(harness))
    answer = contract.reference(harness,args)
    assert answer[1].tolist() == [4,1,2]
    assert answer[0][0].tolist() == [10,11,12,13]
    assert answer[0][1,0] == 20 and answer[0][2,:2].tolist() == [30,31]
    changed = replay.clone(answer)
    changed[0][1,1:] = -123
    replay.check(harness,contract,changed,answer,args)  # Undefined suffix is not data.
    changed[1][1] = 2
    with pytest.raises(AssertionError):
        replay.check(harness,contract,changed,answer,args)


@pytest.mark.parametrize('name', ['min_p','topk_topp','logit_bias','bad_words'])
def test_nan_cannot_masquerade_as_masked_logit(name):
    replay, contract, harness = modules(name)
    args = next(contract.control_inputs(harness))
    answer = contract.reference(harness,args)
    bad = answer.clone()
    index = torch.isneginf(bad).nonzero()[0]
    bad[tuple(index)] = float('nan')
    with pytest.raises(AssertionError):
        replay.check(harness,contract,bad,answer,args)


def test_zero_subnormal_is_rejected():
    replay, contract, harness = modules('write_zeros_to_output')
    args = next(contract.control_inputs(harness))
    answer = contract.reference(harness,args)
    bad = torch.zeros_like(answer)
    bad[0,0] = torch.nextafter(torch.tensor(0.),torch.tensor(1.))
    with pytest.raises(AssertionError):
        replay.check(harness,contract,bad,answer,args)


@pytest.mark.parametrize('dtype,integer,sign', [(torch.float16,torch.int16,-(1<<15)),
                                              (torch.float32,torch.int32,-(1<<31))])
def test_zero_bit_gate_rejects_each_subnormal_sign_without_float_comparison(dtype,integer,sign):
    replay,contract,harness=modules('write_zeros_to_output')
    control=contract.residual_input(dtype,'cpu')
    assert control.shape==(1,7)
    assert control.view(integer)[0,:4].tolist()==[1,sign+1,2,sign+2]
    for index in range(7):
        bad=control[:,index:index+1].contiguous()
        with pytest.raises(AssertionError,match='nonzero bit pattern'):
            replay.check(harness,contract,bad,torch.zeros_like(bad),(bad,))
    signed_zeros=torch.tensor([[0,sign]],dtype=integer).view(dtype)
    replay.check(harness,contract,signed_zeros,torch.zeros_like(signed_zeros),(signed_zeros,))
    for value in (float('nan'),float('inf'),-float('inf')):
        bad=torch.tensor([[value]],dtype=dtype)
        with pytest.raises(AssertionError,match='nonzero bit pattern'):
            replay.check(harness,contract,bad,torch.zeros_like(bad),(bad,))
    contract.verify_subnormal_rejection('cpu')


def test_zero_on_device_negative_control_fails_if_checker_stops_rejecting(monkeypatch):
    _,contract,_=modules('write_zeros_to_output')
    monkeypatch.setattr(contract,'check',lambda *args:None)
    with pytest.raises(AssertionError,match='accepted a subnormal'):
        contract.verify_subnormal_rejection('cpu')


def test_zero_manifest_records_actual_2d_residual_storage():
    _,contract,harness=modules('write_zeros_to_output')
    manifest=json.loads((TASKS/'triton_write_zeros_to_output/workloads.json').read_text())
    declared=manifest['cases'][-1]['params']['control_inputs']
    inputs=list(contract.control_inputs(harness))
    assert len(inputs)==len(declared)==3
    for (value,),row in zip(inputs,declared):
        assert list(value.shape)==row['shape'] and str(value.dtype)=='torch.'+row['dtype']
        if 'storage_values' in row:
            assert value.view(getattr(torch,row['storage_dtype'])).flatten().tolist()==row['storage_values']
        else:
            assert value.tolist()==row['values']


def test_topk_original_boundary_allowance_preserved_but_large_error_rejected():
    replay, contract, harness = modules('topk_topp')
    args = (torch.arange(256,dtype=torch.float32)[None,:],torch.tensor([8],dtype=torch.int32),None,-torch.inf)
    answer = contract.reference(harness,args)
    one_boundary = answer.clone(); one_boundary[0,247]=247
    replay.check(harness,contract,one_boundary,answer,args)
    with pytest.raises(AssertionError):
        replay.check(harness,contract,args[0],answer,args)


class Timed:
    def _bind(self, fn, output):
        self.fn, self.outputs = fn, output
    def rerun(self):
        self.outputs = self.fn()
        return self.outputs


def fake_benchmark(fn, timed_run, prepare_fn=None, **kwargs):
    def run():
        if prepare_fn:
            prepare_fn()
        return fn()
    output = run()
    timed_run._bind(run,output)
    return 1.0, {'benchmark_method':'cpu_test_only'}


def fake_function(replay, contract, harness, arity, *, cached=False, fail=False):
    state = {}
    def impl(args):
        if fail and 'called' in state:
            raise RuntimeError('injected timing failure')
        state['called'] = True
        answer = replay.expected(harness,contract,args)
        if cached:
            state.setdefault('answer',replay.clone(answer))
            answer = replay.clone(state['answer'])
        if contract.FUNCTION == 'combine_sampled_and_draft_tokens':
            args[0].copy_(answer[0]); return answer[1]
        if contract.MUTABLE:
            outputs = answer if len(contract.MUTABLE)>1 else (answer,)
            for index,value in zip(contract.MUTABLE,outputs):
                args[index].copy_(value)
            return args[0] if len(contract.MUTABLE)==1 else None
        return answer
    ns={'impl':impl}
    names=','.join('arg'+str(i) for i in range(arity))
    exec('def fn('+names+'):\n    return impl(('+names+',))',ns)
    return ns['fn']


@pytest.mark.parametrize('name', [n for n in NAMES if n not in ('pack_seq','topk_topp')])
def test_actual_replay_and_finally_restore_cpu_simulation(name):
    replay, contract, harness = modules(name)
    args = next(contract.control_inputs(harness))
    pristine = replay.clone(args)
    recorder = replay.Recorder(harness,contract)
    harness._TimedRun = Timed
    fn = recorder.wrap(fake_function(replay,contract,harness,len(args)))
    _, metadata = recorder.benchmark(fake_benchmark,lambda: fn(*args))
    assert metadata['timed_output_checked'] and metadata['input_state_restored']
    replay.unchanged(args,pristine,())


@pytest.mark.parametrize('name', ['temperature','ranks','post_update','prepare_eagle_inputs','rejection_sample'])
def test_cached_answer_rejected_and_state_restored(name):
    replay, contract, harness = modules(name)
    args = next(contract.control_inputs(harness)); pristine = replay.clone(args)
    recorder = replay.Recorder(harness,contract); harness._TimedRun=Timed
    fn = recorder.wrap(fake_function(replay,contract,harness,len(args),cached=True))
    with pytest.raises(AssertionError):
        recorder.benchmark(fake_benchmark,lambda: fn(*args))
    replay.unchanged(args,pristine,())


def test_exception_restores_buffers_and_readonly_mutation_fails():
    replay, contract, harness = modules('temperature')
    args = next(contract.control_inputs(harness)); pristine = replay.clone(args)
    recorder = replay.Recorder(harness,contract); harness._TimedRun=Timed
    fn=recorder.wrap(fake_function(replay,contract,harness,len(args),fail=True))
    with pytest.raises(RuntimeError,match='injected'):
        recorder.benchmark(fake_benchmark,lambda:fn(*args))
    replay.unchanged(args,pristine,())
    args[1][0]=0
    with pytest.raises(AssertionError,match='read-only'):
        replay.unchanged(args,pristine,contract.MUTABLE)


@pytest.mark.parametrize('name', ['pack_seq','topk_topp'])
@pytest.mark.parametrize('cached', [False,True])
def test_direct_preallocated_launch_is_observed_and_checked(name,cached):
    replay,contract,harness=modules(name)
    harness._TimedRun=Timed
    recorder=replay.Recorder(harness,contract)
    memory={}
    if name=='pack_seq':
        x=torch.arange(15,dtype=torch.float16).reshape(5,3)
        lengths=torch.tensor([1,3,1],dtype=torch.int32)
        out=torch.empty((3,3,3),dtype=x.dtype)
        def launch():
            value=harness.reference_pack_seq(x,lengths.tolist(),0.)
            memory.setdefault('answer',value.clone())
            out.copy_(memory['answer'] if cached else value)
        args=(x,lengths)
    else:
        logits=torch.arange(256,dtype=torch.float32)[None,:]
        k=torch.tensor([32],dtype=torch.int32)
        kernel_args=(logits,None,None,None,k,None)
        kernel_meta={'TOPK_ENABLED':True,'TOPP_ENABLED':False,'MASK_VALUE':-torch.inf}
        def launch():
            value=harness.reference_apply_top_k_top_p(kernel_args[0],kernel_args[4],None)
            assert kernel_meta['TOPK_ENABLED']
            memory.setdefault('answer',value.clone())
            kernel_args[0].copy_(memory['answer'] if cached else value)
        args=(logits,k)
    before=replay.clone(args)
    if cached:
        with pytest.raises(AssertionError):
            recorder.benchmark(fake_benchmark,launch)
    else:
        recorder.benchmark(fake_benchmark,launch)
    replay.unchanged(args,before,())


@pytest.mark.parametrize('name',NAMES)
def test_original_manifest_rows_and_generated_region_are_byte_preserved(name):
    task=TASKS/('triton_'+name)
    expected=ORIGINAL_CONTRACTS[name]
    current=original_manifest(json.loads((task/'workloads.json').read_text()))
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert digest(current['cases'][:expected['original_count']])==expected['rows_sha256']
    assert digest({k:v for k,v in current.items() if k!='cases'})==expected['manifest_metadata_sha256']
    after=(task/'scripts/task_runner.py').read_text()
    region=after.split('# >>> AKA-GENERATED:')[1].split('# <<< AKA-GENERATED <<<')[0]
    assert hashlib.sha256(region.encode()).hexdigest()==expected['generated_region_sha256']
    candidate=task/'source'/('triton_'+name+'.py')
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==expected['candidate_sha256']


@pytest.mark.parametrize('is_prefill',[False,True])
def test_mrope_scored_setup_honors_declared_scenario(is_prefill):
    path=TASKS/'triton_prepare_mrope_positions/scripts/task_runner.py'
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run_performance')
    assignments=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('prefill_lens','num_computed_tokens') for t in n.targets)]
    assert len(assignments)==2
    ns=dict(torch=torch,max_num_reqs=8,max_model_len=128,device='cpu',is_prefill=is_prefill)
    exec(compile(ast.Module(body=assignments,type_ignores=[]),str(path),'exec'),ns)
    assert torch.all(ns['prefill_lens']==(128 if is_prefill else 10))
    assert torch.all(ns['num_computed_tokens']==(0 if is_prefill else 50))


@pytest.mark.parametrize('name',['bad_words','logit_bias','penalties','topk_topp'])
def test_additional_scored_paths_have_declared_correctness_and_cpu_oracles(name):
    replay,contract,harness=modules(name)
    manifest=json.loads((TASKS/('triton_'+name)/'workloads.json').read_text())
    rows={r['test_case_id']:r for r in manifest['cases']}
    observed=[]
    for index,(case_id,args) in enumerate(contract.scored_inputs(harness)):
        observed.append(case_id)
        assert rows[case_id]['checks']==['correctness','performance']
        assert rows[case_id]['params']['case_index']==contract.CONTROL_INDEX+1+index
        before=replay.clone(args)
        answer=replay.expected(harness,contract,args)
        replay.unchanged(args,before,())
        replay.check(harness,contract,answer,answer,args)
        assert args[0].shape[0]>=8 and args[0].shape[1]>=1024
        if name in ('bad_words','logit_bias','topk_topp'):
            assert torch.isfinite(answer).any() and torch.isneginf(answer).any()
        else:
            assert not torch.equal(answer,args[0])
    assert observed==list(contract.SCORED_CASE_IDS)


@pytest.mark.parametrize('name',['bad_words','logit_bias','penalties'])
def test_added_scored_launches_preserve_reset_and_observe_real_output(name,monkeypatch):
    replay,contract,harness=modules(name)
    # Exercise extension + shared Recorder composition on CPU, not GPU timing.
    monkeypatch.setitem(__import__('sys').modules,'_arena_replay',replay)
    extension=load(TASKS/('triton_'+name)/'_arena_additional.py','additional_'+name)
    real_to_device=replay.to_device
    monkeypatch.setattr(extension,'to_device',lambda values,device:real_to_device(values,'cpu'))
    original_count=ORIGINAL_CONTRACTS[name]['original_count']
    sentinel=[{'test_case_id':'untouched-original','execution_time_ms':2.}]
    harness.run_performance=lambda:sentinel.copy()
    harness.run_correctness=lambda **kw:(True,None)
    arity=len(next(contract.scored_inputs(harness))[1])
    function=fake_function(replay,contract,harness,arity)
    harness.load_module=lambda:types.SimpleNamespace(**{contract.FUNCTION:function})
    harness._TimedRun=Timed
    harness._benchmark_cuda_graph_or_events=fake_benchmark
    extension.install(harness,contract)
    replay.install(harness,contract)
    rows=harness.run_performance()
    assert rows[0]==sentinel[0]
    assert len(rows)==1+len(contract.SCORED_CASE_IDS)
    assert all(r['execution_time_ms']>0 and r['replay_input_control_checked'] and r['input_state_restored'] for r in rows[1:])
    assert rows[1]['test_case_id']==contract.SCORED_CASE_IDS[0]
    assert original_count>=5


def test_added_top_p_and_combined_direct_launch_composition(monkeypatch):
    replay,contract,harness=modules('topk_topp')
    monkeypatch.setitem(__import__('sys').modules,'_arena_replay',replay)
    extension=load(TASKS/'triton_topk_topp/_arena_additional.py','additional_topk')
    real_to_device=replay.to_device
    monkeypatch.setattr(extension,'to_device',lambda values,device:real_to_device(values,'cpu'))
    harness.run_performance=lambda:[]
    harness.run_correctness=lambda **kw:(True,None)
    harness.load_module=lambda:types.SimpleNamespace(**{contract.FUNCTION:fake_function(replay,contract,harness,4)})
    def direct(module,logits,k,p,mask_value):
        kernel_args=(logits,None,None,None,k,p)
        kernel_meta={'TOPK_ENABLED':k is not None,'TOPP_ENABLED':p is not None,'MASK_VALUE':mask_value}
        def launch():
            assert kernel_meta['TOPP_ENABLED']
            logits.copy_(harness.reference_apply_top_k_top_p(kernel_args[0],kernel_args[4],kernel_args[5]))
        return {'launch':launch}
    harness.prepare_direct_launch=direct
    harness._TimedRun=Timed
    harness._benchmark_cuda_graph_or_events=fake_benchmark
    extension.install(harness,contract)
    replay.install(harness,contract)
    rows=harness.run_performance()
    assert [r['test_case_id'] for r in rows]==list(contract.SCORED_CASE_IDS)
    assert all(r['execution_time_ms']>0 and r['replay_input_control_checked'] for r in rows)


def test_zero_controls_satisfy_the_actual_public_dimension_precondition():
    _,contract,harness=modules('write_zeros_to_output')
    path=TASKS/'triton_write_zeros_to_output/source/triton_write_zeros_to_output.py'
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='write_zeros')
    preconditions=[n for n in fn.body if isinstance(n,ast.Assert)]
    assert preconditions
    code=compile(ast.Module(body=preconditions,type_ignores=[]),str(path),'exec')
    for (output,) in contract.control_inputs(harness):
        exec(code,{'output':output})
    with pytest.raises(AssertionError,match='2D'):
        exec(code,{'output':torch.zeros(7)})


@pytest.mark.parametrize('name',['bad_words','logit_bias','penalties','topk_topp'])
def test_scored_manifest_describes_exact_constructed_inputs(name):
    _,contract,harness=modules(name)
    rows={r['test_case_id']:r for r in json.loads((TASKS/('triton_'+name)/'workloads.json').read_text())['cases']}
    for case_id,args in contract.scored_inputs(harness):
        row=rows[case_id]; params=row['params']; recipe=params['logits']
        assert list(args[0].shape)==row['shape']
        assert str(args[0].dtype).removeprefix('torch.')==row['dtype']
        if recipe['generator']=='linspace':
            expected=torch.linspace(recipe['start'],recipe['stop'],recipe['numel']).reshape(recipe['reshape'])
        else:
            assert recipe['generator']=='linspace_row_repeat'
            expected=torch.linspace(recipe['start'],recipe['stop'],recipe['row_numel']).repeat(recipe['repeat_rows'],1)
        torch.testing.assert_close(args[0],expected,atol=0,rtol=0)
        if name!='topk_topp':
            assert args[1].tolist()==params['request_mapping_pattern']*params['pattern_repeats']
        if name=='bad_words':
            decoded=[]
            for req,count in enumerate(args[4].tolist()):
                decoded.append([args[2][req,int(args[3][req,j]):int(args[3][req,j+1])].tolist() for j in range(count)])
            assert decoded==params['bad_words_by_request']
            assert args[5].tolist()==params['token_history_by_request']
            assert args[6].tolist()==params['prompt_lengths'] and args[7].tolist()==params['total_lengths']
            assert args[8].tolist()==params['speculative_input_ids_pattern']*params['pattern_repeats']
            assert args[9].tolist()==params['local_positions_pattern']*params['pattern_repeats']
            assert args[10]==params['max_num_bad_words']
        elif name=='logit_bias':
            assert args[2].tolist()==params['positions_pattern']*params['pattern_repeats']
            assert [args[4][req,:count].tolist() for req,count in enumerate(args[3].tolist())]==params['allowed_tokens_by_request']
            for req,count in enumerate(args[5].tolist()):
                assert args[6][req,:count].tolist()==params['biases_by_request'][req]['token_ids']
                torch.testing.assert_close(args[7][req,:count],torch.tensor(params['biases_by_request'][req]['values']),atol=0,rtol=0)
            assert args[8].tolist()==params['minimum_lengths']
            assert [args[10][req,:count].tolist() for req,count in enumerate(args[9].tolist())]==params['stop_tokens_by_request']
        elif name=='penalties':
            assert args[2].tolist()==params['token_ids_pattern']*params['pattern_repeats']
            assert args[3].tolist()==params['local_positions_pattern']*params['pattern_repeats']
            for index,key in [(4,'repetition_penalties'),(5,'frequency_penalties'),(6,'presence_penalties')]:
                torch.testing.assert_close(args[index],torch.tensor(params[key]),atol=0,rtol=0)
            prompt=[[token for token in range(row['shape'][1]) if (int(mask[token//32])>>(token%32))&1] for mask in args[7]]
            counts=[{str(token):int(count[token]) for token in count.nonzero().flatten().tolist()} for count in args[8]]
            assert prompt==params['prompt_tokens_by_request'] and counts==params['output_token_counts_by_request']
            assert list(args[7].shape)==params['prompt_mask_shape'] and list(args[8].shape)==params['output_counts_shape']
            assert args[9]==params['num_speculative_tokens']
        else:
            assert (None if args[1] is None else args[1].tolist())==params['top_k']
            torch.testing.assert_close(args[2],torch.tensor(params['top_p']),atol=0,rtol=0)
            assert args[3]==float(params['mask_value'])
            assert params['timed_entrypoint']=='_topk_topp_kernel'
            assert params['scratch_allocation']=='outside_timing'
