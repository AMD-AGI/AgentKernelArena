# batch_memcpy, 256 KiB slots

Copies a batch of whole state slots. vLLM's v1 model runner (`vllm/v1/worker/mamba_utils.py`) collects, for
every request, layer and state tensor of a hybrid Mamba model, a (source address, destination address, size)
triple and moves them all with one launch, one Triton program per triple.

Scenario: the mean copy is one 262144-byte slot (the order of magnitude of a conv or SSM state block);
the number of copies in a call (`batch`) varies. The baseline is the production kernel and its copy loop
(1024 bytes per iteration, byte access).

Interface change: the production kernel receives raw device addresses (`int64` pointers). A task input is
copied into evaluator-owned storage, so an address cannot be an input. The task therefore passes the source
and destination pools and byte offsets into them; the baseline adds the offset to the pool base. The copy
loop is unchanged. Copy sizes vary like the old task's slightly varying sizes: pairs of 262144 -/+ d bytes with arbitrary d
(so most copies end in a masked tail and start at unaligned offsets), summing to the pool size. Copies
are packed back to back in two independent random orders in the source and destination pool, so every
destination byte is written exactly once and the comparison is bitwise (`exact`).

Comparison: `exact` (bitwise) instead of the default bound, because the output is a byte copy with no
rounding; any difference is a wrong byte.

Workloads: batch counts of 256 KiB copies. A single copy takes the baseline well over 10 us (one program
walking 256 slices of 1 KiB); large batches become bandwidth-bound. Production copy sizes vary per state
tensor, so one representative mean size is fixed. Batches of 3 and 37 are correctness-only boundary cases.
