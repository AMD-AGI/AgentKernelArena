"""Exact reference of the penalty-state builder: prompt token bitmask and output token counts."""
import torch


VOCAB_SIZE = 151936  # the scenario's constant vocab_size axis


def reference(*, idx_mapping, all_token_ids, prompt_len, prefill_len):
    n = idx_mapping.shape[0]
    vocab_size = VOCAB_SIZE
    words = (vocab_size + 31) // 32
    mask_out = torch.zeros(n, words, dtype=torch.int32, device=all_token_ids.device)
    counts_out = torch.zeros(n, vocab_size, dtype=torch.int32, device=all_token_ids.device)
    weights = (1 << torch.arange(32, dtype=torch.int64, device=mask_out.device))
    for row in idx_mapping.tolist():
        plen = int(prompt_len[row])
        flen = int(prefill_len[row])
        tokens = all_token_ids[row].to(torch.int64)
        present = torch.zeros(words * 32, dtype=torch.int64, device=mask_out.device)
        present[tokens[:plen]] = 1
        packed = (present.view(words, 32) * weights).sum(-1)
        packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed)  # bit 31 is the int32 sign bit
        mask_out[row] = packed.to(torch.int32)
        counts_out[row] = torch.bincount(tokens[plen:flen], minlength=vocab_size).to(torch.int32)
    return {"prompt_bin_mask": mask_out, "output_bin_counts": counts_out}
