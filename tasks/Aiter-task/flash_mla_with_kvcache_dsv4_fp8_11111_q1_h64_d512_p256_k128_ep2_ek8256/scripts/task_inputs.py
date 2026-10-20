# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Inputs, valid lengths and index patterns for DSv4 sparse flash MLA.

The buffers are allocated here, in the definition's declared shapes and dtypes,
and filled by ``task_initialize``, which is the schema bundle's own
``initialize`` callback: queries at a moderate logit scale, packed FP8 KV pools
with their scale slots, uniform legal slot indices in each pool, random prefix
lengths (row 0 full, row 1 empty) whose suffixes are padded with -1, and normal
attention sinks. Nothing about these distributions is decided here.

The callback's lengths alone do not exercise the operator: the valid lengths
are values inside the length vectors, and a decode step sees them change while
every pool and index-table capacity stays fixed. Each case therefore names its
lengths and index pattern. A ``bundle`` case keeps the callback's own. Every
other case writes its declared lengths, and then updates the index tables that
depend on them: a prefix position the callback padded with -1 (because its own
random length was shorter) receives a legal slot drawn by the callback's index
rule, uniform over the pool's slots; positions past the prefix are -1; declared
holes set prefix positions to -1, and a legal tail fills the positions past the
prefix with legal slots instead, which the operator must ignore.

One full callback run per batch shape provides the base draw and the extra
pool, which stays fixed like a deployed KV cache, together with the sinks. A
further draw of the call-varying operands runs the callback without its
optional extra pool, which redraws the queries, the main pool and its indices,
and draws the extra indices by the same uniform slot rule; running the whole
callback again would rewrite the extra pool, which takes seconds per draw.

Every constant lives in the configured workload JSON, so these helpers carry no
operator dimensions of their own. Arena copies each task directory into its own
workspace, so a task cannot import from a sibling.
"""

from __future__ import annotations

from typing import Any

import torch

import task_compare
import task_contract
import task_initialize

WORKLOAD = task_contract.load_workload()

DEFINITION = str(WORKLOAD["definition"])
AXES: dict[str, int] = dict(WORKLOAD["axes"])
VARIABLE_AXES: list[str] = list(WORKLOAD["variable_axes"])
INPUTS: dict[str, dict] = dict(WORKLOAD["inputs"])
OUTPUTS: dict[str, dict] = dict(WORKLOAD["outputs"])
OUTPUT_NAMES: tuple[str, ...] = tuple(OUTPUTS)
SCALARS: dict[str, Any] = dict(WORKLOAD["scalars"])
POOL_NAMES: list[str] = task_contract.pools(WORKLOAD)
POOLS = {pool: task_contract.POOLS[pool] for pool in POOL_NAMES}
SEED = int(WORKLOAD["seed"])

DTYPES = {"bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn,
          "int32": torch.int32, "float32": torch.float32}

# The entrypoint is explicit task data; it is not derived from operator identity.
BUILDER_SYMBOL = task_contract.candidate_entry()["symbol"]

# Benchmark parameters used for both baseline and candidate. Timing must be
# CUDA-graph based: small batches run for microseconds and eager timing would
# be dominated by per-call host dispatch.
BENCH_WARMUP = int(WORKLOAD["bench"]["warmup"])
BENCH_REPETITION = int(WORKLOAD["bench"]["repetition"])
BENCH_TARGET_MS = float(WORKLOAD["bench"]["target_ms"])

CASES: tuple[dict[str, Any], ...] = tuple(WORKLOAD["cases"])
CASE_IDS: tuple[str, ...] = tuple(str(case["case_id"]) for case in CASES)

GATE_EXPLANATION = (
    "gate: scripts/task_compare.py, the schema bundle's own comparison callback. "
    "It requires the BF16 output within additive 1e-2, the FP32 LSE within additive "
    "1e-3 and +inf exactly where a row has no KV entry -- nothing here relaxes it."
)

# Operands that change from call to call; everything else is held per batch shape.
CALL_VARYING: tuple[str, ...] = ("q", "kv_cache", *(POOLS[pool]["indices"] for pool in POOL_NAMES))
# The operands a decode step keeps while the call-varying ones change: the
# extra KV pool, the attention sinks and the case's lengths.
PERSISTENT_INPUTS: tuple[str, ...] = tuple(
    name for name in INPUTS if INPUTS[name]["shape"] is not None and name not in CALL_VARYING)


def dimensions(case: dict[str, Any]) -> dict[str, int]:
    return {**AXES, **{axis: int(case[axis]) for axis in VARIABLE_AXES}}


def declared_shape(spec: dict, dims: dict[str, int]) -> tuple[int, ...]:
    return tuple(dims[name] for name in spec["shape"])


def shape_key(case: dict[str, Any]) -> tuple:
    return tuple(int(case[axis]) for axis in VARIABLE_AXES)


def capacity(inputs: dict[str, Any], pool: str) -> int:
    cache = inputs[POOLS[pool]["cache"]]
    return cache.shape[0] * cache.shape[1]


def call_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    """The operator's full argument set, in the schema's input order."""
    return {name: inputs[name] for name in INPUTS}


def launch_args(inputs: dict[str, Any]) -> tuple[Any, ...]:
    """Positional candidate arguments, in the schema's input order."""
    return tuple(inputs[name] for name in INPUTS)


def _uniform_slots(like: torch.Tensor, slots: int, generator: torch.Generator) -> torch.Tensor:
    """The callback's index rule: uniform over the pool's slots."""
    return torch.empty_like(like).random_(0, slots, generator=generator)


class BatchInputs:
    """Live buffers of one batch shape, its base draw, and per-case application."""

    def __init__(self, case: dict[str, Any], device: str = "cuda"):
        dims = dimensions(case)
        self.inputs = {
            name: (SCALARS[name] if spec["shape"] is None else
                   torch.empty(declared_shape(spec, dims), dtype=DTYPES[spec["dtype"]], device=device))
            for name, spec in INPUTS.items()
        }
        task_initialize.run(self.inputs, seed=SEED)
        self.device = self.inputs["q"].device
        self.bundle_lengths = {pool: self.inputs[POOLS[pool]["lengths"]].clone() for pool in POOL_NAMES}
        self.base = self._raw(SEED, from_live=True)

    def _raw(self, seed: int, *, from_live: bool = False) -> dict[str, torch.Tensor]:
        """Call-varying operands as drawn, before a case's lengths are applied.

        Index entries are the callback's own where it left them legal; ``fill``
        holds a legal uniform slot for every index position.
        """
        live = self.inputs
        if not from_live:
            # The callback without its optional extra pool redraws the queries,
            # the main pool and its indices; lengths and sinks go to scratch.
            scratch = {name: torch.empty_like(live[name]) for name in ("sparse_lens", "sinks")}
            held = {name: live[name].clone() for name in ("q", "kv_cache", "sparse_indices")}
            task_initialize.run({"q": live["q"], "kv_cache": live["kv_cache"],
                                 "sparse_indices": live["sparse_indices"], "sm_scale": live["sm_scale"],
                                 **scratch}, seed=seed)
            raw = {name: live[name].clone() for name in held}
            for name, value in held.items():
                live[name].copy_(value)
        else:
            raw = {name: live[name].clone() for name in ("q", "kv_cache", "sparse_indices")}
            if "extra" in POOLS:
                raw["extra_sparse_indices"] = live["extra_sparse_indices"].clone()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        for pool in POOL_NAMES:
            indices = POOLS[pool]["indices"]
            raw["fill_" + pool] = _uniform_slots(live[indices], capacity(live, pool), generator)
            if indices not in raw:
                raw[indices] = raw["fill_" + pool].clone()
        return raw

    def case_lengths(self, case: dict[str, Any]) -> dict[str, torch.Tensor]:
        declared = task_contract.row_lengths(case, WORKLOAD)
        if declared is None:
            return {pool: value.clone() for pool, value in self.bundle_lengths.items()}
        return {pool: torch.tensor(values, dtype=torch.int32, device=self.device)
                for pool, values in declared.items()}

    def draw(self, case: dict[str, Any], raw: dict[str, torch.Tensor],
             lengths: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Final call-varying operands of a case for one raw draw."""
        lengths = self.case_lengths(case) if lengths is None else lengths
        rule = task_contract.index_rule(case)
        result = {"q": raw["q"], "kv_cache": raw["kv_cache"]}
        for pool in POOL_NAMES:
            indices = raw[POOLS[pool]["indices"]]
            width = indices.shape[-1]
            positions = torch.arange(width, device=self.device)[None, None, :]
            in_prefix = positions < lengths[pool].to(torch.int64)[:, None, None]
            legal = torch.where(indices >= 0, indices, raw["fill_" + pool])
            table = torch.where(in_prefix, legal, legal if rule["legal_tail"] else torch.full_like(legal, -1))
            period = rule["holes"].get(pool)
            if period:
                table = torch.where(in_prefix & ((positions + 1) % period == 0), -1, table)
            result[POOLS[pool]["indices"]] = table.contiguous()
        return result

    def prepare(self, case: dict[str, Any], lengths: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
        """Load the base draw with the case's lengths and index pattern into the live buffers."""
        lengths = self.case_lengths(case) if lengths is None else lengths
        for pool in POOL_NAMES:
            self.inputs[POOLS[pool]["lengths"]].copy_(lengths[pool])
        load_draw(self.inputs, self.draw(case, self.base, lengths))
        return self.inputs

    def draws(self, case: dict[str, Any], seeds: list[int]) -> list[dict[str, torch.Tensor]]:
        """Independent draws of the call-varying operands for a case, one per seed."""
        return [self.draw(case, self._raw(seed)) for seed in seeds]


def load_draw(inputs: dict[str, Any], draw: dict[str, torch.Tensor]) -> None:
    """Copy a snapshot into the live buffers, keeping their storage."""
    for name, value in draw.items():
        inputs[name].copy_(value)


def named_outputs(value: Any) -> tuple[Any, Any]:
    """Map a returned (output, lse) pair or mapping onto the declared order.

    Raises AssertionError for a return value that does not carry exactly the
    declared outputs, which is a candidate contract failure.
    """
    if isinstance(value, dict):
        if set(value) != set(OUTPUT_NAMES):
            raise AssertionError(f"outputs must be named {list(OUTPUT_NAMES)}")
        return tuple(value[name] for name in OUTPUT_NAMES)
    if not isinstance(value, (tuple, list)) or len(value) != len(OUTPUT_NAMES):
        raise AssertionError("operator must return (output, lse)")
    return tuple(value)


def effective_entries(inputs: dict[str, Any]) -> torch.Tensor:
    """Per row, the number of nonnegative indices inside the valid prefixes."""
    total = None
    for pool in POOL_NAMES:
        spec = POOLS[pool]
        indices = inputs[spec["indices"]]
        positions = torch.arange(indices.shape[-1], device=indices.device)[None, None, :]
        in_prefix = positions < inputs[spec["lengths"]].to(torch.int64)[:, None, None]
        count = ((indices >= 0) & in_prefix).sum(dim=(1, 2))
        total = count if total is None else total + count
    return total


def check_case_inputs(inputs: dict[str, Any], case: dict[str, Any], state: BatchInputs) -> dict:
    """Validate a case's inputs before they are checked or timed.

    Lengths must lie within their index widths and equal the declared ones;
    inside a prefix every index is a legal slot of its pool except declared -1
    holes; past it, -1 or (for a legal tail) a legal slot; queries and sinks are
    finite.
    """
    expected = state.case_lengths(case)
    rule = task_contract.index_rule(case)
    evidence = {}
    for pool in POOL_NAMES:
        spec = POOLS[pool]
        lengths = inputs[spec["lengths"]].to(torch.int64)
        indices = inputs[spec["indices"]].to(torch.int64)
        width, slots = indices.shape[-1], capacity(inputs, pool)
        if bool((lengths < 0).any()) or bool((lengths > width).any()):
            raise RuntimeError(f"{spec['lengths']} outside [0, {width}]")
        if not torch.equal(inputs[spec["lengths"]], expected[pool]):
            raise RuntimeError(f"{spec['lengths']} differ from the case's lengths")
        positions = torch.arange(width, device=indices.device)[None, None, :]
        in_prefix = positions < lengths[:, None, None]
        period = rule["holes"].get(pool)
        holes = in_prefix & ((positions + 1) % period == 0) if period else torch.zeros_like(in_prefix)
        legal = (indices >= 0) & (indices < slots)
        if not bool(torch.where(holes, indices == -1, legal).masked_select(in_prefix).all()):
            raise RuntimeError(f"{spec['indices']} prefix holds an illegal or undeclared -1 index")
        outside = legal if rule["legal_tail"] else indices == -1
        if not bool(outside.masked_select(~in_prefix).all()):
            raise RuntimeError(f"{spec['indices']} past the prefix violates the case's tail rule")
        evidence[pool] = {"lengths_min": int(lengths.min()), "lengths_max": int(lengths.max()),
                          "distinct_lengths": len({int(v) for v in lengths.unique().tolist()}),
                          "holes": int(holes.sum()), "pool_slots": slots}
    for name in ("q", "sinks"):
        if not bool(torch.isfinite(inputs[name].float()).all()):
            raise RuntimeError(f"{name} contains nonfinite values")
    evidence["rows_without_kv"] = int((effective_entries(inputs) == 0).sum())
    return evidence


def verdict(got: tuple, expected: tuple) -> tuple[bool, str]:
    """Apply the bundle's comparison callback and report what it decided.

    The callback raises AssertionError for a candidate that fails its contract
    or its tolerance, and ValueError for a reference it considers invalid. Only
    the first is a candidate verdict, so only the first is caught.
    """
    try:
        task_compare.run(got, expected)
    except AssertionError as failure:
        return False, str(failure)
    return True, "within tolerance"


def assert_candidate_is_independent(source: str) -> None:
    """Apply the documented dependency policy before candidate import."""
    task_contract.assert_source_independent(source)
