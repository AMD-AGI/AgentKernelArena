"""Bad words kernel, adapted from vLLM bad_words.py."""
import torch
import triton
import triton.language as tl


@triton.jit
def _bad_words_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    bad_word_token_ids_ptr,
    bad_word_token_ids_stride,
    bad_word_offsets_ptr,
    bad_word_offsets_stride,
    num_bad_words_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
):
    logit_idx = tl.program_id(0)
    bw_idx = tl.program_id(1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx)
    num_bad_words = tl.load(num_bad_words_ptr + req_state_idx)

    if bw_idx >= num_bad_words:
        return

    pos = tl.load(expanded_local_pos_ptr + logit_idx)
    cur_req_first_pos = logit_idx - pos

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    total_len = tl.load(total_len_ptr + req_state_idx)
    output_len = total_len - prompt_len
    effective_len = output_len + pos

    bd_offsets_base = bad_word_offsets_ptr + req_state_idx * bad_word_offsets_stride
    bd_tokens_base = bad_word_token_ids_ptr + req_state_idx * bad_word_token_ids_stride
    output_base = all_token_ids_ptr + req_state_idx * all_token_ids_stride + prompt_len

    start = tl.load(bd_offsets_base + bw_idx)
    end = tl.load(bd_offsets_base + bw_idx + 1)
    bad_word_len = end - start
    prefix_len = bad_word_len - 1

    if prefix_len > effective_len:
        return

    last_token = tl.load(bd_tokens_base + end - 1)
    match = 1
    for i in range(prefix_len):
        expected = tl.load(bd_tokens_base + start + i)
        actual_pos = effective_len - prefix_len + i

        from_spec_input = actual_pos >= output_len
        if from_spec_input:
            spec_offset = actual_pos - output_len
            actual = tl.load(input_ids_ptr + cur_req_first_pos + spec_offset)
        else:
            actual = tl.load(output_base + actual_pos)

        match = match & (expected == actual)

    if match:
        tl.store(logits_ptr + logit_idx * logits_stride + last_token, -float("inf"))


def apply_bad_words(
    logits, expanded_idx_mapping, bad_word_token_ids, bad_word_offsets,
    num_bad_words, all_token_ids, prompt_len, total_len,
    input_ids, expanded_local_pos, max_num_bad_words,
):
    """Apply bad words filtering to logits."""
    total_num_tokens = logits.shape[0]
    _bad_words_kernel[(total_num_tokens, max_num_bad_words)](
        logits, logits.stride(0), expanded_idx_mapping,
        bad_word_token_ids, bad_word_token_ids.stride(0),
        bad_word_offsets, bad_word_offsets.stride(0),
        num_bad_words, all_token_ids, all_token_ids.stride(0),
        prompt_len, total_len, input_ids, expanded_local_pos,
    )
