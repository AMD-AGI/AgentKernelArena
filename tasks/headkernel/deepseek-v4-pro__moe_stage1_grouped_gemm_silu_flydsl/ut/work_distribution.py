"""Protected observed-work distributions and legal native MoE input generation."""
from array import array
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import random

SCHEMA = 'moe-observed-work-distributions-v1'
STATUS = 'FROZEN_WITH_OBSERVED_WORK_DISTRIBUTIONS'
KIND = 'generated_observed_work_distribution'
CORRECTNESS_MODES = ('targeted_uncovered_behaviors', 'exhaustive_observed_values')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(value, reason):
    if not value:
        raise ValueError('Observed-work contract: ' + reason)


def checked_file(root, relative):
    root = Path(root).resolve(); relative = Path(relative); path = root / relative
    require(not relative.is_absolute() and '..' not in relative.parts and not path.is_symlink()
            and path.is_file() and path.resolve().is_relative_to(root), 'unsafe or absent file')
    return path


def load_contract(root, manifest):
    """Keep every original case and its measurement policy byte-for-byte in JSON."""
    ref = manifest['observed_work_distributions']
    path = checked_file(root, ref['path'])
    require(sha(path) == ref['sha256'], 'distribution manifest changed')
    registry = json.loads(path.read_text())
    require(registry['schema'] == SCHEMA and manifest['status'] == STATUS, 'unknown schema/status')
    parent = registry['base_manifest']; base_path = checked_file(root, parent['path'])
    require(sha(base_path) == parent['sha256'], 'original manifest changed')
    base = json.loads(base_path.read_text())
    require(digest(base) == parent['fingerprint'], 'original manifest fingerprint changed')
    count = len(base['cases'])
    require(manifest['cases'][:count] == base['cases'], 'original fixed cases or order changed')
    for key in set(base) - {'cases', 'required_case_ids', 'status'}:
        require(manifest[key] == base[key], 'original contract field changed: ' + key)
    extra = manifest['cases'][count:]
    require(len(extra) == len(registry['groups']) and len(extra) > 0, 'distribution case count differs')
    require(set(manifest['required_case_ids']) == {c['case_id'] for c in manifest['cases']}
            and len(manifest['required_case_ids']) == len(manifest['cases']), 'required case set differs')
    fixed = {c['case_id']: c for c in base['cases']}
    for case in extra:
        group = registry['groups'][case['distribution_group']]
        anchor = fixed[group['base_case_id']]
        require(case['case_id'] == group['case_id'] and case['provenance_kind'] == KIND,
                'distribution case identity differs')
        require(case['tensors'] == anchor['tensors'] and case['scalars'] == anchor['scalars']
                and case['fixture'] == anchor['fixture'] and case['stage'] == anchor['stage'], 'native ABI/base fixture differs')
        require(case['calls_per_sample'] == 1 and case['occurrences'] == 1, 'distribution estimator multiplicity differs')
        rows = group['histogram']
        require(rows and len({x['variant_id'] for x in rows}) == len(rows)
                and len({tuple(x['num_valid_ids']) for x in rows}) == len(rows), 'duplicate or empty histogram')
        require(sum(x['occurrences'] for x in rows) == group['observed_calls'], 'histogram total differs')
        require(digest(rows) == group['histogram_sha256'], 'histogram digest differs')
        for row in rows:
            values = {'num_valid_ids': {'dtype': 'torch.int32', 'shape': [2], 'values': row['num_valid_ids']}}
            require(digest(values) == row['variant_id'] and row['num_valid_ids'][1] == group['tokens'], 'observed variant differs')
            require(type(row['occurrences']) is int and row['occurrences'] > 0
                    and set(row['occurrences_by_rank']) <= {str(rank) for rank in range(8)}
                    and all(type(n) is int and n >= 0 for n in row['occurrences_by_rank'].values())
                    and sum(row['occurrences_by_rank'].values()) == row['occurrences'], 'invalid observed frequency')
        if manifest['seam'] == 'moe2':
            require(case['live_fixture'] == group['stage1_parent_fixture'], 'paired stage1 parent differs')
        correctness = group['correctness']
        require(correctness['default_mode'] == CORRECTNESS_MODES[0], 'default correctness mode changed')
        selected = correctness['targeted_variant_ids']
        require(isinstance(selected, list) and len(selected) == len(set(selected))
                and set(selected) <= {r['variant_id'] for r in rows}, 'invalid targeted setting selection')
        require(correctness['review_status'] in ('pending_layout_review', 'reviewed'), 'unknown selection status')
        if correctness['review_status'] == 'reviewed':
            require(selected and correctness.get('reviewed_by') and correctness.get('rationale'),
                    'reviewed targeted settings require an owner and rationale')
    for relative, expected in registry['support_sha256'].items():
        require(sha(checked_file(root, relative)) == expected, 'protected generation source changed: ' + relative)
    return base, registry


def correctness_variants(group, mode=None):
    """Default to reviewed behavior checks; exhaustive coverage is explicit."""
    contract = group['correctness']
    mode = mode or contract['default_mode']
    require(mode in CORRECTNESS_MODES, 'unknown correctness mode')
    if mode == CORRECTNESS_MODES[1]:
        return mode, list(group['histogram'])
    require(contract['review_status'] == 'reviewed' and contract['targeted_variant_ids'],
            'targeted correctness selection is awaiting layout review')
    selected = set(contract['targeted_variant_ids'])
    rows = [row for row in group['histogram'] if row['variant_id'] in selected]
    require(len(rows) == len(selected), 'targeted correctness selection contains an unseen setting')
    return mode, rows


def weighted_variant(group, private_seed):
    rows = group['histogram']; total = group['observed_calls']
    draw = random.Random(private_seed).randrange(total)
    cumulative = 0
    for row in rows:
        cumulative += row['occurrences']
        if draw < cumulative:
            return row
    raise ValueError('Histogram draw escaped its total')


def make_routes(group, num_valid, seed):
    """Exact native padded work with complete unique token/slot assignments."""
    tokens, topk, experts, tile = (group[k] for k in ('tokens', 'topk', 'experts', 'block_m'))
    require(num_valid in {r['num_valid_ids'][0] for r in group['histogram']}, 'unobserved work count')
    capacity, expert_slots = group['sorted_capacity'], group['expert_capacity']
    require(num_valid % tile == 0 and num_valid <= capacity, 'work count exceeds native capacity')
    rng = random.Random(seed); real_routes = tokens * topk; blocks = num_valid // tile
    active = min(experts, blocks, real_routes)
    block_counts = [blocks // active + (i < blocks % active) for i in range(active)]
    counts = [(n - 1) * tile + 1 for n in block_counts]
    remaining = real_routes - sum(counts)
    require(0 <= remaining <= active * (tile - 1), 'infeasible native padded-work count')
    order = list(range(active)); rng.shuffle(order)
    for position, index in enumerate(order):
        following = len(order) - position - 1
        low, high = max(0, remaining - following * (tile - 1)), min(tile - 1, remaining)
        extra = rng.randint(low, high); counts[index] += extra; remaining -= extra
    require(remaining == 0 and max(counts) <= tokens, 'expert degree exceeds token domain')
    expert_order = list(range(experts)); rng.shuffle(expert_order)
    token_order = list(range(tokens)); rng.shuffle(token_order)
    result = {'sorted_token_ids': array('i', [(topk << 24) | tokens]) * capacity,
              'sorted_expert_ids': array('i', [-1]) * expert_slots,
              'topk_ids': array('i', [-1]) * real_routes,
              'num_valid_ids': array('i', [num_valid, tokens])}
    slots = [0] * tokens; row = cursor = 0
    for expert, count, nblocks in zip(expert_order, counts, block_counts):
        for j in range(count):
            token = token_order[(cursor + j) % tokens]; slot = slots[token]; slots[token] += 1
            result['topk_ids'][token * topk + slot] = expert
            result['sorted_token_ids'][row + j] = (slot << 24) | token
        for block in range(nblocks):
            result['sorted_expert_ids'][row // tile + block] = expert
        cursor += count; row += nblocks * tile
    require(row == num_valid and slots == [topk] * tokens, 'generated routing changed exact work')
    return result


def scale_offsets(rows, columns, padded_columns, torch):
    r = rows[:, None]; c = torch.arange(columns, dtype=torch.int64, device=rows.device)[None, :]
    return ((r // 32) * (padded_columns * 32) + (c // 8) * 256 + (c % 4) * 64
            + (r % 16) * 4 + ((c // 4) % 2) * 2 + ((r // 16) % 2))


def cpu_tensor(inputs, name, snapshots, bindings):
    import torch
    value = inputs[name]; pointer = value.untyped_storage().data_ptr()
    aliases = [k for k, t in bindings.items() if t.untyped_storage().data_ptr() == pointer and k in snapshots]
    require(len(aliases) == 1, 'missing or ambiguous CPU input storage: ' + name)
    raw = snapshots[aliases[0]]
    require(raw.device.type == 'cpu' and raw.dtype == torch.uint8, 'CPU byte truth required')
    return torch.empty(0, dtype=value.dtype, device='cpu').set_(raw.untyped_storage(), value.storage_offset(),
               tuple(value.shape), tuple(value.stride())).clone()


class Stage1Inputs:
    def __init__(self, group, inputs, snapshots, api):
        import torch
        self.group, self.inputs, self.pristine, self.api = group, inputs, snapshots, api
        bindings, _ = api['runtime_abi'](inputs, None)
        ids = cpu_tensor(inputs, 'sorted_token_ids', snapshots, bindings).to(torch.int64)
        count = int(cpu_tensor(inputs, 'num_valid_ids', snapshots, bindings)[0])
        encoded = ids[:count]; tokens = encoded & 0xffffff; slots = encoded >> 24
        mask = (tokens < group['tokens']) & (slots >= 0) & (slots < group['topk'])
        rows = torch.arange(count, dtype=torch.int64)[mask]; tokens = tokens[mask]; slots = slots[mask]
        pairs = tokens * group['topk'] + slots
        require(torch.equal(pairs.sort().values, torch.arange(group['tokens'] * group['topk'])), 'parent routing is incomplete')
        columns = int(inputs['a' if 'a' in inputs else 'hidden_states'].shape[1]) // 32
        scale = cpu_tensor(inputs, 'a1_scale', snapshots, bindings).view(torch.uint8).reshape(-1)
        offsets = scale_offsets(rows, columns, ((columns + 7) // 8) * 8, torch)
        route_scales = scale[offsets]
        require(not bool((route_scales == 255).any()), 'nonfinite parent activation scale')
        first = torch.full((group['tokens'],), -1, dtype=torch.int64)
        for index, token in enumerate(tokens.tolist()):
            if first[token] < 0:
                first[token] = index
        require(bool((first >= 0).all()), 'parent misses an activation row')
        self.token_scales = route_scales[first].contiguous()
        require(torch.equal(route_scales, self.token_scales[tokens]), 'parent token scales differ across routes')
        self.columns = columns

    def refresh(self, routes, seed):
        import torch
        from snapshots import raw_storage
        self.api['restore_storages'](self.inputs, self.pristine)
        copy_routes(self.inputs, routes)
        primary = 'a' if 'a' in self.inputs else 'hidden_states'
        value = self.inputs[primary]; generator = torch.Generator(device=value.device).manual_seed(seed)
        value.copy_(torch.randn(value.shape, dtype=torch.float32, device=value.device, generator=generator).to(value.dtype))
        ids = torch.frombuffer(routes['sorted_token_ids'], dtype=torch.int32).to(torch.int64)
        count = routes['num_valid_ids'][0]; active = ids[:count]; tokens = active & 0xffffff; slots = active >> 24
        valid = (tokens < self.group['tokens']) & (slots >= 0) & (slots < self.group['topk'])
        rows = torch.arange(count, dtype=torch.int64)[valid]; tokens = tokens[valid]
        offsets = scale_offsets(rows, self.columns, ((self.columns + 7) // 8) * 8, torch)
        raw = raw_storage(self.inputs['a1_scale']); raw.zero_()
        raw[offsets.to(raw.device)] = self.token_scales[tokens].to(raw.device)
        return rows, tokens, slots[valid]


def copy_routes(inputs, routes):
    import torch
    for name in ('sorted_token_ids', 'sorted_expert_ids', 'num_valid_ids', 'topk_ids'):
        if name in inputs and inputs[name] is not None:
            source = torch.frombuffer(routes[name], dtype=torch.int32).reshape(inputs[name].shape)
            inputs[name].copy_(source)


def load_parent_inputs(root, reference, cfg, module):
    from runtime_capture import restore_phase
    from snapshots import restore
    path = checked_file(root, reference['path'])
    require(sha(path) == reference['sha256'], 'stage1 parent fixture changed')
    record = json.loads(path.read_text())
    require(record['source_sha256'] == cfg['native_source_sha256'] and record['startup_values'] is False,
            'stage1 parent source differs')
    tensors = restore_phase(path.parent, record, 'inputs', device='cuda', max_storage_bytes=64 << 30)
    result = {}
    for name in cfg['signature_parameters']:
        result[name] = tensors[name] if name in tensors else restore({'tree': record['controls'][name], 'storages': {}}, 'cuda', module)
    if cfg.get('has_var_kwargs'):
        result['_kwargs'] = restore({'tree': record['controls'].get('_kwargs', {}), 'storages': {}}, 'cuda', module)
    for name, value in result.items():
        if name in tensors:
            for attr, encoded in record['controls'].get(name + '_attributes', {}).items():
                require(not isinstance(encoded, dict) or 'tensor_binding' not in encoded, 'unsupported parent tensor attribute')
                setattr(value, attr, restore({'tree': encoded, 'storages': {}}, 'cuda', module))
    return result, record


class Provider:
    def __init__(self, root, group, registry, inputs, pristine, api):
        import torch
        self.root, self.group, self.inputs, self.pristine, self.api = Path(root), group, inputs, pristine, api
        self.draws = []; self.producer_calls = 0
        self.stage1 = None; self.producer = None; self.producer_identity = None
        if group['seam'] != 'moe2':
            self.stage1 = Stage1Inputs(group, inputs, pristine, api)
        else:
            gen = registry['generation'][group['stage']]
            cfg = json.loads(checked_file(root, gen['config']).read_text())
            if gen['kind'] == 'frozen_repaired_flydsl_reference':
                module, fn, identity = api['load_native'](self.root / gen['task_root'], 'reference')
                require(identity['emitter_source_sha256'] == gen['emitter_sha256'], 'wrong stage1 repaired source')
            else:
                require(gen['kind'] == 'pinned_image_native', 'unknown stage1 generator')
                from native import runtime_closure
                cfg = runtime_closure(checked_file(root, gen['config']).parent.parent)
                module = importlib.import_module(cfg['module'])
                require(sha(Path(inspect.getsourcefile(module))) == cfg['native_source_sha256'], 'stage1 image wrapper changed')
                fn = getattr(module, cfg['function'])
                identity = {'kind': gen['kind'], 'module': cfg['module'], 'function': cfg['function'], 'source_sha256': cfg['native_source_sha256']}
            parent, parent_record = load_parent_inputs(root, group['stage1_parent_fixture'], cfg, module)
            base_path = checked_file(root, group['stage2_parent_fixture']['path']); base_record = json.loads(base_path.read_text())
            require(sha(base_path) == group['stage2_parent_fixture']['sha256'], 'stage2 parent changed')
            for first, second in (('output', 'inter_states'), ('scale', 'a2_scale')):
                a = parent_record['payload']['outputs'][parent_record['outputs'][first]['alias']]
                b = base_record['payload']['inputs'][base_record['inputs'][second]['alias']]
                require(a == b, 'parent fixtures are not a paired stage1/stage2 chain')
            self.stage1 = Stage1Inputs(group, parent, api['storage_snapshots'](parent), api)
            self.producer, self.producer_identity = fn, identity
            bindings, _ = api['runtime_abi'](inputs, None)
            ids = cpu_tensor(inputs, 'sorted_token_ids', pristine, bindings).to(torch.int64)
            count = int(cpu_tensor(inputs, 'num_valid_ids', pristine, bindings)[0])
            weights = cpu_tensor(inputs, 'sorted_weights', pristine, bindings)
            self.slot_weights = torch.empty(group['tokens'] * group['topk'], dtype=weights.dtype)
            pairs = []
            for row, encoded in enumerate(ids[:count].tolist()):
                token, slot = encoded & 0xffffff, encoded >> 24
                if token < group['tokens'] and 0 <= slot < group['topk']:
                    pair = token * group['topk'] + slot; self.slot_weights[pair] = weights[row]; pairs.append(pair)
            require(sorted(pairs) == list(range(group['tokens'] * group['topk'])), 'stage2 parent weights miss routes')

    def refresh(self, seed, forced=None):
        import torch
        from snapshots import raw_storage
        variants = {row['num_valid_ids'][0]: row for row in self.group['histogram']}
        row = weighted_variant(self.group, seed) if forced is None else variants[forced]
        n = row['num_valid_ids'][0]
        route_seed = int(hashlib.sha256((str(seed) + ':' + row['variant_id']).encode()).hexdigest()[:16], 16)
        routes = make_routes(self.group, n, route_seed)
        live, tokens, slots = self.stage1.refresh(routes, seed)
        if self.producer is not None:
            self.api['restore_storages'](self.inputs, self.pristine)
            copy_routes(self.inputs, routes)
            sorted_weights = self.inputs['sorted_weights']; sorted_weights.zero_()
            sorted_weights[live.to(sorted_weights.device)] = self.slot_weights[tokens * self.group['topk'] + slots].to(sorted_weights.device)
            arguments = dict(self.stage1.inputs); extra = arguments.pop('_kwargs', {})
            produced = self.producer(**arguments, **extra); torch.cuda.synchronize(); self.producer_calls += 1
            require(isinstance(produced, (tuple, list)) and len(produced) == 2, 'stage1 must return payload and scale')
            for value, name in zip(produced, ('inter_states', 'a2_scale')):
                target = self.inputs[name]
                require(value.shape == target.shape and value.dtype == target.dtype
                        and value.stride() == target.stride() and value.storage_offset() == target.storage_offset()
                        and value.untyped_storage().nbytes() == target.untyped_storage().nbytes(), 'stage1 output ABI differs')
                raw_storage(target).copy_(raw_storage(value))
            require(bool(torch.isfinite(produced[0].float()).all()), 'nonfinite fresh stage1 payload')
            columns = produced[0].shape[-1] // 32
            offsets = scale_offsets(live, columns, produced[1].shape[1], torch).reshape(-1)
            meaningful = raw_storage(produced[1])[offsets.to(produced[1].device)]
            require(not bool((meaningful == 255).any()), 'nonfinite fresh stage1 live scales')
            del produced
        require(self.inputs['num_valid_ids'].detach().cpu().tolist() == row['num_valid_ids'], 'restored work differs from observed value')
        self.draws.append({'variant_id': row['variant_id'], 'num_valid_ids': row['num_valid_ids'], 'input_seed': seed,
                           'route_seed': route_seed, 'routing_origin': 'generated_legal_routes_not_actual_other_rank_capture',
                           'activation_origin': 'fresh_seeded_fp8_values_with_remapped_captured_token_scales',
                           'fresh_stage1_reference_output': self.producer is not None})
        return self.api['storage_snapshots'](self.inputs)
