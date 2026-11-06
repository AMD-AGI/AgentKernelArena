"""Inputs of the DeepSeek-V4 paged top-512 transform: decode rows of a 1M-token-context batch.

* ``seq_lens``: row ``r`` takes range ``r % 16`` of ``length_ranges``, a decode mix from an empty
  row to the full score width that covers every path of the operator (empty, shorter than k,
  exactly k, k + 1, the in-register sizes up to 16384, streamed long rows, width - 1 and width).
  Ranged entries draw their exact length per input draw, so the lengths change between draws
  while the amount of work, and hence the timing, stays the same.
* ``scores``: each row is a random permutation of ``width`` distinct float32 values, so the valid
  prefix is tie-free. Rows in even groups of 16 (rows 0-15, 32-47, ...) use the standard-normal
  quantiles ``Phi^-1((i + 0.5) / width)``, smoothly distributed like indexer logits, whose top
  tail is sparse. Rows in odd groups of 16 use evenly spaced values over ``[CROWDED_LOW, 1)``,
  like the earlier task's uniform scores: their top 512 crowd into a few coarse value bins (1365
  scores share the fp16 bin of the 512th largest at full width), so an implementation must
  resolve a crowded threshold exactly. Every score past the row's length is raised by
  ``INVALID_OFFSET`` above all valid scores, so an implementation that reads beyond ``seq_lens``
  selects wrong positions.
* ``page_tables``: each row maps its 4097 pages to distinct random page ids of a large pool.
* ``metadata``: the top-k v2 routing plan that routes no row to the cluster path
  (``cluster_threshold = 2^31 - 1``, ``N = 0``), which is what the ROCm kernel does for every
  length; ROCm compiles the cluster path out and never reads the plan.
"""
import math

import torch

INVALID_OFFSET = 16.0
CROWDED_LOW = -0.5
PAGE_ID_LIMIT = 1 << 24
PLAN_NO_CLUSTER = 2**31 - 1
JITTER = 4096


def length_ranges(width):
    """(low, high) of the valid length of row r, for r % 16; each draw is uniform in the range."""
    return (
        (width - JITTER + 1, width),      # near-full context: the row of a batch of one
        (0, 0),                           # empty row (idle or padded request)
        (131072 - JITTER + 1, 131072),    # half context
        (1, 1),                           # one token
        (width, width),                   # exactly the full width
        (511, 511),                       # one short of k
        (65536 - JITTER + 1, 65536),      # quarter context
        (512, 512),                       # exactly k
        (width - 1, width - 1),           # width - 1
        (513, 513),                       # k + 1, the smallest selection
        (16384, 16384),                   # the largest in-register row
        (16385, 16385),                   # the smallest streamed row
        (196608 - JITTER + 1, 196608),    # three-quarter context
        (2, 510),                         # short row, padded
        (63, 64),                         # one page short / exactly one page
        (514, 16383),                     # in-register selection
    )


def _normal_quantiles(width, device):
    p = (torch.arange(width, dtype=torch.float64, device=device) + 0.5) / width
    return (math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)).to(torch.float32)


def _crowded_values(width, device):
    p = (torch.arange(width, dtype=torch.float64, device=device) + 0.5) / width
    return (CROWDED_LOW + (1.0 - CROWDED_LOW) * p).to(torch.float32)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, width, num_pages = axes["batch"], axes["width"], axes["num_pages"]
    page_size = int(case["scalars"]["page_size"])
    if num_pages * page_size < width:
        raise ValueError("the page table must cover the score width")

    ranges = torch.tensor(length_ranges(width), dtype=torch.int64, device=device)
    entry = ranges[torch.arange(batch, device=device) % ranges.shape[0]]
    draw = torch.rand(batch, generator=generator, device=device, dtype=torch.float64)
    span = entry[:, 1] - entry[:, 0] + 1
    lengths = entry[:, 0] + torch.minimum((draw * span).to(torch.int64), span - 1)
    seq_lens = lengths.to(torch.int32)

    families = (_normal_quantiles(width, device), _crowded_values(width, device))
    group = ranges.shape[0]
    scores = torch.empty((batch, width), dtype=torch.float32, device=device)
    for row, length in enumerate(lengths.tolist()):
        values = families[(row // group) % 2]
        scores[row] = values[torch.randperm(width, generator=generator, device=device)]
        if length < width:
            scores[row, length:] += INVALID_OFFSET

    keys = torch.rand((batch, num_pages), generator=generator, device=device)
    base = torch.randint(0, PAGE_ID_LIMIT - num_pages, (batch, 1), generator=generator,
                         device=device)
    page_tables = (keys.argsort(dim=1) + base).to(torch.int32)

    metadata = torch.zeros((batch + 1, 2), dtype=torch.int32, device=device)
    metadata[0, 0] = PLAN_NO_CLUSTER
    return {"scores": scores, "seq_lens": seq_lens, "page_tables": page_tables,
            "metadata": metadata}
