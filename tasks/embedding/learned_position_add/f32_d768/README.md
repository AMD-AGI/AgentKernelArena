# Learned position embedding add, float32, width 768

`out[b, s, :] = x[b, s, :] + pos_embedding[s, :]` with positions `arange(seq_len)`, as in GPT-2 /
BERT / ViT learned position embeddings. Scenario: 512-token sequences, width 768, a 1024-row
position table (only the first 512 rows are used); the batch varies. Correctness is bitwise
(`exact`): a float32 add of two float32 values is correctly rounded, and the float64 reference
sum rounded to float32 is identical, so no tolerance is needed.

Origin: the old task wrapped the GPU-mode `PositionEmbedder` module with a toy configuration
(embedding_dim 4, max sequence 4, batches up to 64), far below the timing floor. The scenario is
rescaled to a production width and sequence length; those values are an assumption. Batches
under 16 are correctness-only (below 10 us).

Baseline: a standalone Triton rewrite of the pointwise broadcast-add kernel TorchInductor emits
for `x + F.embedding(arange(S), table)` (PyTorch, BSD-3-Clause); the position lookup is the
identity row `s = row % seq_len`. One program per token row.

The `exact` comparison departs from the default bound deliberately: the operation is a single correctly-rounded float32 add, so any difference from the
float64-then-rounded reference is a defect.
