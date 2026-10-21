"""Task device kernels remain editable; module and host comparison state is protected.

Structural CPU checks do not establish GPU correctness, timing or Python isolation.
"""
from pathlib import Path
import ast
import shutil
import pytest
import yaml
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec
ROOT=Path(__file__).resolve().parents[1]
TASKS = [{'task': 'triton2triton/vllm/triton_apply_grammar_bitmask',
  'source': 'source/triton_apply_grammar_bitmask.py',
  'symbols': ['_apply_grammar_bitmask_kernel'],
  'host_functions': ['apply_grammar_bitmask'],
  'matched_API': 'allclose'},
 {'task': 'triton2triton/vllm/triton_awq_gemm',
  'source': 'source/triton_awq_gemm.py',
  'symbols': ['awq_gemm_kernel'],
  'host_functions': ['awq_gemm_triton'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_bmm',
  'source': 'source/triton_bmm.py',
  'symbols': ['bmm_kernel'],
  'host_functions': ['bmm_triton'],
  'matched_API': 'allclose'},
 {'task': 'triton2triton/vllm/triton_causal_conv1d_update',
  'source': 'source/triton_causal_conv1d_update.py',
  'symbols': ['_causal_conv1d_update_kernel'],
  'host_functions': ['causal_conv1d_update'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_chunked_prefill_paged_decode',
  'source': 'source/triton_chunked_prefill_paged_decode.py',
  'symbols': ['cdiv_fn', 'kernel_paged_attention_2d'],
  'host_functions': ['chunked_prefill_paged_decode'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_correct_attn_cp_out',
  'source': 'source/triton_correct_attn_cp_out.py',
  'symbols': ['_correct_attn_cp_out_kernel'],
  'host_functions': ['correct_attn_cp_out'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_decode_attn_grouped_stage1',
  'source': 'source/_fwd_grouped_kernel_stage1.py',
  'symbols': ['tanh', '_fwd_grouped_kernel_stage1'],
  'host_functions': ['decode_grouped_att_m_fwd'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_decode_attn_stage1',
  'source': 'source/_fwd_kernel_stage1.py',
  'symbols': ['tanh', '_fwd_kernel_stage1'],
  'host_functions': ['decode_att_m_fwd'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_decode_attn_stage2',
  'source': 'source/_fwd_kernel_stage2.py',
  'symbols': ['_fwd_kernel_stage2'],
  'host_functions': ['decode_softmax_reducev_fwd'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_ep_gather',
  'source': 'source/triton_ep_gather.py',
  'symbols': ['_fwd_kernel_ep_gather'],
  'host_functions': ['ep_gather'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_ep_scatter_1',
  'source': 'source/triton_ep_scatter_1.py',
  'symbols': ['round_up_128', '_fwd_kernel_ep_scatter_1'],
  'host_functions': ['ep_scatter_1'],
  'matched_API': 'equal'},
 {'task': 'triton2triton/vllm/triton_expert_kernel',
  'source': 'source/triton_expert_kernel.py',
  'symbols': ['expert_triton_kernel'],
  'host_functions': ['expert_gemm'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_fla_cumsum_scalar',
  'source': 'source/triton_fla_cumsum_scalar.py',
  'symbols': ['chunk_local_cumsum_scalar_kernel'],
  'host_functions': ['chunk_local_cumsum_scalar'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_fla_cumsum_vector',
  'source': 'source/triton_fla_cumsum_vector.py',
  'symbols': ['chunk_local_cumsum_vector_kernel'],
  'host_functions': ['chunk_local_cumsum_vector'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_fla_layernorm',
  'source': 'source/triton_fla_layernorm.py',
  'symbols': ['layer_norm_fwd_kernel'],
  'host_functions': ['layer_norm_fwd'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_flash_prefill_attention',
  'source': 'source/triton_flash_prefill_attention.py',
  'symbols': ['_fwd_kernel'],
  'host_functions': ['get_block_size', 'context_attention_fwd'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_fused_gdn_gating',
  'source': 'source/triton_fused_gdn_gating.py',
  'symbols': ['fused_gdn_gating_kernel'],
  'host_functions': ['fused_gdn_gating'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_fused_moe_lora',
  'source': 'source/triton_fused_moe_lora.py',
  'symbols': ['_get_lora_id', '_get_expert_id', '_get_token_offs', 'fused_moe_lora_kernel'],
  'host_functions': ['_next_power_of_2', '_get_ptr', '_adjust_kernel_inputs', 'fused_moe_lora'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_gumbel_sample',
  'source': 'source/triton_gumbel_sample.py',
  'symbols': ['_gumbel_sample_kernel'],
  'host_functions': ['gumbel_sample'],
  'matched_API': 'equal'},
 {'task': 'triton2triton/vllm/triton_lightning_attn_kv_reduce',
  'source': 'source/triton_lightning_attn_kv_reduce.py',
  'symbols': ['_fwd_kv_reduce'],
  'host_functions': ['lightning_attn_kv_reduce_forward'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_lora_expand',
  'source': 'source/triton_lora_expand.py',
  'symbols': ['mm_k', 'do_expand_kernel', '_lora_expand_kernel'],
  'host_functions': ['_get_lora_b_ptr', 'lora_expand', '_next_power_of_2'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_lora_shrink',
  'source': 'source/triton_lora_shrink.py',
  'symbols': ['mm_k', 'do_shrink_kernel', '_lora_shrink_kernel'],
  'host_functions': ['_get_lora_a_ptr', 'lora_shrink'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_matmul_persistent',
  'source': 'source/triton_matmul_persistent.py',
  'symbols': ['_compute_pid', 'matmul_kernel_persistent'],
  'host_functions': ['_requires_int64_index', 'matmul_persistent'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_mean',
  'source': 'source/triton_mean.py',
  'symbols': ['mean_kernel'],
  'host_functions': ['mean_dim'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_merge_16x16_to_32x32',
  'source': 'source/triton_merge_16x16_to_32x32.py',
  'symbols': ['merge_16x16_to_32x32_inverse_kernel'],
  'host_functions': ['merge_16x16_to_32x32'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_merge_attn_states',
  'source': 'source/triton_merge_attn_states.py',
  'symbols': ['merge_attn_states_kernel'],
  'host_functions': ['merge_attn_states'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_min_p',
  'source': 'source/triton_min_p.py',
  'symbols': ['_min_p_kernel'],
  'host_functions': ['apply_min_p'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_mrope',
  'source': 'source/triton_mrope.py',
  'symbols': ['_triton_mrope_forward'],
  'host_functions': ['triton_mrope'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_paged_prefix_prefill',
  'source': 'source/triton_paged_prefix_prefill.py',
  'symbols': ['_fwd_kernel'],
  'host_functions': ['context_attention_fwd'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_paged_prefix_prefill_alibi',
  'source': 'source/triton_paged_prefix_prefill_alibi.py',
  'symbols': ['_fwd_kernel_alibi'],
  'host_functions': ['context_attention_fwd_alibi'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_penalties',
  'source': 'source/triton_penalties.py',
  'symbols': ['_penalties_kernel'],
  'host_functions': ['apply_penalties'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_per_token_group_quant_fp8_colmajor',
  'source': 'source/triton_per_token_group_quant_fp8_colmajor.py',
  'symbols': ['_per_token_group_quant_fp8_colmajor'],
  'host_functions': ['_get_fp8_dtype', '_get_fp8_min_max', 'per_token_group_quant_fp8_colmajor'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_per_token_group_quant_int8',
  'source': 'source/triton_per_token_group_quant_int8.py',
  'symbols': ['_per_token_group_quant_int8'],
  'host_functions': ['per_token_group_quant_int8'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_per_token_quant_int8',
  'source': 'source/triton_per_token_quant_int8.py',
  'symbols': ['_per_token_quant_int8'],
  'host_functions': ['per_token_quant_int8'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_rejection_sample',
  'source': 'source/triton_rejection_sample.py',
  'symbols': ['_rejection_sample_kernel'],
  'host_functions': ['rejection_sample'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_sample_recovered_tokens',
  'source': 'source/triton_sample_recovered_tokens.py',
  'symbols': ['sample_recovered_tokens_kernel'],
  'host_functions': ['sample_recovered_tokens'],
  'matched_API': 'equal'},
 {'task': 'triton2triton/vllm/triton_scaled_mm',
  'source': 'source/triton_scaled_mm.py',
  'symbols': ['scaled_mm_kernel'],
  'host_functions': ['is_weak_contiguous', 'triton_scaled_mm'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_silu_mul_fp8_quant_dg',
  'source': 'source/triton_silu_mul_fp8_quant_dg.py',
  'symbols': ['_silu_mul_fp8_quant_deep_gemm'],
  'host_functions': ['silu_mul_fp8_quant'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_silu_mul_quant_fp8',
  'source': 'source/triton_silu_mul_quant_fp8.py',
  'symbols': ['_silu_mul_per_token_group_quant_fp8_colmajor'],
  'host_functions': ['_get_fp8_dtype', '_get_fp8_min_max', 'silu_mul_per_token_group_quant_fp8_colmajor'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_solve_tril_16x16',
  'source': 'source/triton_solve_tril_16x16.py',
  'symbols': ['solve_tril_16x16_kernel'],
  'host_functions': ['solve_tril_16x16'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_topk_topp',
  'source': 'source/triton_topk_topp.py',
  'symbols': ['_topk_topp_kernel'],
  'host_functions': ['_next_power_of_2', 'apply_top_k_top_p_triton'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_unified_attention_2d',
  'source': 'source/triton_unified_attention_2d.py',
  'symbols': ['cdiv_fn', 'apply_softcap', 'find_seq_idx', 'kernel_unified_attention_2d'],
  'host_functions': ['unified_attention_2d'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_unified_attention_3d',
  'source': 'source/triton_unified_attention_3d.py',
  'symbols': ['cdiv_fn', 'apply_softcap', 'find_seq_idx', 'kernel_unified_attention_3d'],
  'host_functions': ['unified_attention_3d'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_w8a8_block_int8_matmul',
  'source': 'source/triton_w8a8_block_int8_matmul.py',
  'symbols': ['_w8a8_block_int8_matmul'],
  'host_functions': ['w8a8_block_int8_matmul'],
  'matched_API': 'assert_close'},
 {'task': 'triton2triton/vllm/triton_wy_fast_recompute_wu',
  'source': 'source/triton_wy_fast_recompute_wu.py',
  'symbols': ['recompute_w_u_fwd_kernel'],
  'host_functions': ['wy_fast_recompute_wu'],
  'matched_API': 'assert_close'}]


def copy_workspace(tmp_path,row,legacy=False):
    root=tmp_path/'task';shutil.copytree(ROOT/'tasks'/row['task'],root)
    if legacy:
        path=root/'config.yaml';data=yaml.safe_load(path.read_text())
        data['candidate']['editable']=[row['source']]
        path.write_text(yaml.safe_dump(data,sort_keys=False))
    spec=load_task_spec(root/'config.yaml',task_id=row['task'])
    return root,snapshot_workspace_harness(root,task_spec=spec)


def source_payload(row):
    expression='torch.testing.assert_close' if row['matched_API']=='assert_close' else 'torch.'+row['matched_API']
    return expression+' = lambda *args, **kwargs: '+('None' if row['matched_API']=='assert_close' else 'True')+'\n'


@pytest.mark.parametrize('row',TASKS,ids=lambda r:r['task'].split('/')[-1])
def test_every_original_device_function_and_new_implementation_helper_can_change(tmp_path,row):
    root,guard=copy_workspace(tmp_path,row);source=root/row['source'];tree=ast.parse(source.read_text())
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    for name in row['symbols']:functions[name].body.append(ast.Pass())
    assert 'implementation_kernel_helper' not in functions
    tree.body+=ast.parse('@triton.jit\ndef implementation_kernel_helper(value):\n    return value\n').body
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    verify_workspace_harness(guard,discard_added=False)


@pytest.mark.parametrize('row',TASKS,ids=lambda r:r['task'].split('/')[-1])
@pytest.mark.parametrize('attack',['actual_comparator','allclose','isfinite','module_import','host_dispatch','kernel_binding','helper_default','outer_decorator','oracle','timer'])
def test_real_guard_rejects_module_host_policy_and_oracle_mutations(tmp_path,row,attack):
    root,guard=copy_workspace(tmp_path,row);source=root/row['source'];text=source.read_text()
    if attack=='host_dispatch':
        tree=ast.parse(text)
        if row['host_functions']:
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==row['host_functions'][0]);node.body=[ast.Return(ast.Constant(None))]
            source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
        else:source.write_text(text+'\n_host_dispatch_override = True\n')
    elif attack=='outer_decorator':
        tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==row['symbols'][-1])
        node.decorator_list.insert(0,ast.Name(id='implementation_helper',ctx=ast.Load()))
        tree.body+=ast.parse('def implementation_helper(value):\n    return value\n').body
        source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    elif attack in ['oracle','timer']:
        oracle='_arena_checks.py' if (root/'_arena_checks.py').is_file() else '_arena_contract.py'
        path=root/(oracle if attack=='oracle'else'scripts/task_runner.py')
        path.write_text(path.read_text()+'\n_override_contract = True\n')
    else:
        payloads={
            'actual_comparator':source_payload(row),
            'allclose':'torch.allclose = lambda *args, **kwargs: True\n',
            'isfinite':'torch.isfinite = lambda value: torch.ones_like(value,dtype=torch.bool)\n',
            'module_import':'import sys\n_candidate_registry = sys.modules\n',
            'kernel_binding':row['symbols'][-1]+' = lambda *args, **kwargs: None\n',
            'helper_default':'def implementation_helper(value=torch.empty(1)):\n    return value\n',
        }
        source.write_text(text+'\n'+payloads[attack])
    with pytest.raises(RuntimeError,match='[Pp]rotected|[Hh]arness|policy'):
        verify_workspace_harness(guard,discard_added=False)


@pytest.mark.parametrize('row',TASKS,ids=lambda r:r['task'].split('/')[-1])
def test_identical_actual_gate_payload_is_rejected_with_either_editable_scope(tmp_path,row):
    # A whole-file scope admits the statement; the runtime-patching rule still refuses the reassigned gate.
    root,old=copy_workspace(tmp_path/'old',row,legacy=True);source=root/row['source']
    source.write_text(source.read_text()+'\n'+source_payload(row))
    with pytest.raises(RuntimeError,match='runtime patching'):verify_workspace_harness(old,discard_added=False)
    current,new=copy_workspace(tmp_path/'new',row);source=current/row['source']
    source.write_text(source.read_text()+'\n'+source_payload(row))
    with pytest.raises(RuntimeError):verify_workspace_harness(new,discard_added=False)


@pytest.mark.parametrize('row',TASKS,ids=lambda r:r['task'].split('/')[-1])
def test_all_original_device_functions_are_declared_and_host_definitions_protected(row):
    root=ROOT/'tasks'/row['task'];spec=load_task_spec(root/'config.yaml',task_id=row['task']);edit,=spec.candidate.editable
    assert edit.scope=='symbols'and list(edit.symbols)==row['symbols']and edit.allow_new_helpers is True
    assert all(e.symbol in row['symbols']for e in spec.candidate.entrypoints)
    source=ast.parse((root/row['source']).read_text())
    actual=[n.name for n in source.body if isinstance(n,ast.FunctionDef)and any('triton.jit'in ast.unparse(d)for d in n.decorator_list)]
    assert actual==row['symbols']
