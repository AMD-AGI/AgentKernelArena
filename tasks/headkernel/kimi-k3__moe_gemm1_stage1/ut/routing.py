"""Legal fresh Kimi routes and correctness checks for retained populations.

Only the padded-row histogram has empirical frequencies. The random population
law below is a test-input policy, not an estimate of the missing routing law.
"""
from array import array
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from fresh_runner import FreshCallbacks


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _bounded_counts(total, lower, upper, rng):
    """Draw a bounded integer composition, keeping every feasible choice legal."""
    remaining = total - sum(lower)
    room = [hi - lo for lo, hi in zip(lower, upper)]
    if any(n < 0 for n in room) or not 0 <= remaining <= sum(room):
        raise ValueError('Route population cannot represent the required live routes')
    counts = list(lower)
    order = list(range(len(counts)))
    rng.shuffle(order)
    following_room = sum(room)
    for index in order:
        following_room -= room[index]
        addition = rng.randint(max(0, remaining - following_room), min(room[index], remaining))
        counts[index] += addition
        remaining -= addition
    if remaining:
        raise AssertionError('Unassigned live routes')
    return counts


def route_population(tokens, tile_m, valid_rows, rng, population=None):
    """Return expert-ID/block-count pairs and realizable live route degrees.

    For b blocks an expert needs (b-1)*tile_m+1 through b*tile_m live
    routes, capped at tokens so a token never selects an expert twice.
    These current Kimi token extents are all divisible by their tile size.
    """
    if (type(tokens) is not int or tokens < 16 or tokens >= 1 << 24
            or type(tile_m) is not int or tile_m < 2 or tokens % tile_m
            or type(valid_rows) is not int or valid_rows % tile_m
            or valid_rows < tokens * 16):
        raise ValueError('Invalid current Kimi routing geometry')
    blocks = valid_rows // tile_m
    max_blocks = tokens // tile_m
    if population is None:
        # Sum of minimum live counts is valid_rows-active*(tile_m-1).
        # Together with the per-expert cap, these bound all feasible active
        # counts. Random bounded compositions admit skew instead of imposing
        # equal block counts; they are not uniform over all compositions.
        least = max(16, (blocks + max_blocks - 1) // max_blocks,
                    (valid_rows - tokens * 16 + tile_m - 2) // (tile_m - 1))
        most = min(896, blocks, tokens * 16)
        if least > most:
            raise ValueError('Observed work count has no legal expert population')
        active = rng.randint(least, most)
        block_counts = _bounded_counts(blocks, [1] * active, [max_blocks] * active, rng)
        population = sorted(zip(rng.sample(range(896), active), block_counts))
    else:
        population = list(population)
    if not population or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                             or any(type(n) is not int for n in pair) for pair in population):
        raise ValueError('Malformed expert population')
    experts, block_counts = zip(*population)
    if (len(set(experts)) != len(experts) or any(not 0 <= e < 896 for e in experts)
            or any(not 1 <= b <= max_blocks for b in block_counts)
            or sum(block_counts) != blocks):
        raise ValueError('Expert population does not fit the native work count')
    counts = _bounded_counts(tokens * 16,
                             [(b - 1) * tile_m + 1 for b in block_counts],
                             [min(b * tile_m, tokens) for b in block_counts], rng)
    return population, counts


def make_routes(tokens, tile_m, valid_rows, capacity_rows, expert_slots, seed,
                population=None, *, padding_token=None):
    """Realize fresh distinct top-16 assignments, optionally at an exact population.

    Concatenated expert segments traverse a shuffled token ring exactly 16
    times. Each expert's segment is at most one ring long, so its tokens are
    unique. Padding follows live rows inside each expert's allocated blocks.
    """
    if (type(capacity_rows) is not int or type(expert_slots) is not int
            or valid_rows > capacity_rows or tile_m <= 0 or valid_rows // tile_m > expert_slots):
        raise ValueError('Routing exceeds the captured allocation')
    rng = random.Random(seed)
    population, counts = route_population(tokens, tile_m, valid_rows, rng, population)
    token_order = list(range(tokens))
    rng.shuffle(token_order)
    padding_token = tokens if padding_token is None else padding_token
    topk = array('i', [-1]) * (tokens * 16)
    sorted_ids = array('i', [(16 << 24) | padding_token]) * capacity_rows
    sorted_experts = array('i', [-1]) * expert_slots
    slots = [0] * tokens
    row = cursor = 0
    for (expert, block_count), count in zip(population, counts):
        for j in range(count):
            token = token_order[(cursor + j) % tokens]
            slot = slots[token]
            slots[token] += 1
            topk[token * 16 + slot] = expert
            sorted_ids[row + j] = (slot << 24) | token
        for block in range(block_count):
            sorted_experts[row // tile_m + block] = expert
        cursor += count
        row += block_count * tile_m
    if row != valid_rows or any(n != 16 for n in slots):
        raise AssertionError('Fresh routing did not preserve exact native work')
    return {'sorted_token_ids': sorted_ids, 'sorted_expert_ids': sorted_experts,
            'topk_ids': topk, 'num_valid_ids': array('i', [valid_rows, tokens])}


def population_summary(routes, tile_m, retained=None):
    """Bind a work log to the actual generated population without timing weights."""
    blocks = routes['num_valid_ids'][0] // tile_m
    histogram = sorted(Counter(routes['sorted_expert_ids'][:blocks]).items())
    values = {'num_valid_ids': list(routes['num_valid_ids']), 'tile_m': tile_m,
              'expert_sorted_block_histogram': histogram}
    ident = hashlib.sha256(_canonical(values).encode()).hexdigest()[:24]
    if retained is not None and ident != retained['population_id']:
        raise AssertionError('Generated routing missed its retained population')
    return {'population_id': ident, 'active_experts': len(histogram),
            'max_expert_blocks': max(n for _, n in histogram),
            'routing_kind': 'retained_population' if retained is not None else 'fresh_legal_population'}


def retained_populations(case):
    path = Path(__file__).resolve().with_name('retained_route_populations.json')
    data = json.loads(path.read_text())
    if (data.get('schema') != 'kimi-retained-routing-populations-v1'
            or data.get('population_frequencies_recorded') is not False
            or data.get('full_routing_histories_recorded') is not False):
        raise ValueError('Invalid retained population evidence')
    examples = data['cases'][case['case_id']]
    if not examples or len({e['population_id'] for e in examples}) != len(examples):
        raise ValueError('Missing or duplicate retained populations')
    tensors = case['tensors']
    tokens = tensors.get('inputs.a', tensors.get('inputs.out'))['shape'][0]
    controls = case['scalars']['arguments']
    tile = controls.get('tile_m', controls.get('block_m'))
    observed = dict(case['work_distribution']['valid_rows_histogram'])
    for example in examples:
        values = {key: example[key] for key in
                  ('num_valid_ids', 'tile_m', 'expert_sorted_block_histogram')}
        if (hashlib.sha256(_canonical(values).encode()).hexdigest()[:24] != example['population_id']
                or example['num_valid_ids'][1] != tokens or example['tile_m'] != tile
                or example['num_valid_ids'][0] not in observed):
            raise ValueError('Retained population differs from its captured case')
    return examples


def check_retained_populations(prepared, seed):
    """Use the existing graph and deferred oracle once for every retained example."""
    if prepared.route_population is not None:
        raise AssertionError('Nested retained routing override')
    results = []
    try:
        for example in retained_populations(prepared.case):
            prepared.route_population = example
            if prepared.callbacks.check_once(seed) is not True:
                raise AssertionError('Retained population oracle did not pass')
            if _canonical(prepared.observe()) != _canonical(prepared.case):
                raise AssertionError('Retained population changed the native ABI')
            results.append({'population_id': example['population_id'], 'seed': seed,
                            'valid_rows': example['num_valid_ids'][0], 'correct': True})
    finally:
        prepared.route_population = None
    return results


class RoutingCallbacks(FreshCallbacks):
    """Add targeted routing checks; inherit the exact existing timed replay path."""
    def __init__(self, *, prepared, **kwargs):
        self.prepared = prepared
        super().__init__(**kwargs)

    def correctness_row(self, case, policy, *, observe, corrupt_outputs, challenge_seed):
        row = super().correctness_row(case, policy, observe=observe,
                                      corrupt_outputs=corrupt_outputs, challenge_seed=challenge_seed)
        row['retained_route_populations'] = check_retained_populations(self.prepared, challenge_seed)
        return row
