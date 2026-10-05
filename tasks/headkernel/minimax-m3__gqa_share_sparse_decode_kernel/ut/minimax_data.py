"""Exact captured geometry, physical views and fresh legal floating inputs."""
import math

from served_contract import FLOAT_INPUTS, DTYPE_BYTES, decode_bytes, span_bytes


def flatten_output(value):
    if isinstance(value, tuple):
        return {f"result.{i}": item for i, item in enumerate(value)}
    return {"result": value}


def tensor_from_bytes(item, spec, root=None):
    import torch
    data = decode_bytes(item, limit=2 << 30, root=root)
    if not data:
        return torch.empty(spec["shape"], dtype=getattr(torch, spec["dtype"]))
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return raw.view(getattr(torch, spec["dtype"])).reshape(spec["shape"])


def byte_equal(left, right):
    import torch
    return torch.equal(left.view(torch.uint8), right.view(torch.uint8))


def work_controls(args):
    values = {name: value for name, value in args.items()
              if name in {"seq_lens", "cu_seqlens", "prefix_lens", "cu_seqblocks_q"}
              and value is not None}
    if args.get("topk_idx") is not None:
        values["capture_topk_counts"] = (args["topk_idx"] >= 0).sum(-1)
    result = {}
    for name, value in values.items():
        runs = []
        for item in value.detach().cpu().reshape(-1).tolist():
            if runs and runs[-1][0] == item:
                runs[-1][1] += 1
            else:
                runs.append([item, 1])
        result["work." + name] = {"shape": list(value.shape), "runs": runs}
    return result


class Inputs:
    def __init__(self, case, definition, device="cuda", root=None):
        import torch
        self.case, self.definition, self.device = case, definition, device
        self.specs = {name: spec for name, spec in case["tensors"].items() if spec["role"] == "input"}
        self.aliases = case["input_aliases"]
        sizes = {}
        for name, spec in self.specs.items():
            size = case.get("original_storage_nbytes", {}).get(name, span_bytes(spec))
            if size < span_bytes(spec):
                raise ValueError("captured storage is shorter than its physical view")
            sizes[self.aliases[name]] = max(sizes.get(self.aliases[name], 0), size)
        if sum(sizes.values()) > definition["max_replay_storage_bytes"]:
            raise ValueError("complete physical input span exceeds the explicit replay budget")
        self.storage = {alias: torch.zeros(size, dtype=torch.uint8, device=device) for alias, size in sizes.items()}
        self.tensors = self.views(self.storage)
        self.external = definition.get("fixture_format") == "minimax-representatives-v1"
        self.verified_blobs = set()
        if self.external:
            from minimax_fixtures import load_bundle, geometry
            self.entries = load_bundle(root, case, definition)
            self.states = [geometry(entry, self.specs, self.verified_blobs) for entry in self.entries]
        else:
            self.entries = []
            self.states = [{name: tensor_from_bytes(item, self.specs[name], root)
                            for name, item in state["geometry"].items()} for state in case["states"]]
        self.scalars = {name: value for name, value in case["scalars"].items()
                        if not name.startswith(("result", "work."))}
        self.args = {**self.scalars, **self.tensors}

    def _fresh_geometry(self, state, seed):
        """Preserve work bounds/counts while changing concrete page/routing values.

        A cyclic page translation preserves the original contiguous runs. Sparse
        block permutations are bijections within fully causal and fully masked
        regions; boundary blocks stay fixed, preserving each query's causal work.
        """
        import torch
        self.validate_geometry(state)
        geometry = dict(state)
        table = state["req_to_token"].clone()
        slots = state["slot_ids"].clone()
        lengths = state["seq_lens"].reshape(-1)
        total = self.tensors["k_cache"].shape[0]
        generator = torch.Generator(device="cpu").manual_seed((int(seed)*104729 + 17) % (2**63-1))
        page_shift = int(torch.randint(max(total, 1), (1,), generator=generator))
        request_shift = int(torch.randint(max(table.shape[0], 1), (1,), generator=generator))
        for index, (count, slot) in enumerate(zip(lengths.tolist(), slots.reshape(-1).tolist())):
            if count:
                row = (int(slot) + total) % total
                if not 0 <= row < table.shape[0]:
                    raise ValueError("active captured request is outside the physical table")
                target = (row + request_shift) % table.shape[0]
                table[target, :count] = ((state["req_to_token"][row, :count].to(torch.int64) + total + page_shift) % total).to(table.dtype)
                slots.reshape(-1)[index] = target
        geometry["req_to_token"], geometry["slot_ids"] = table, slots
        if "topk_idx" not in state:
            return geometry
        topk = state["topk_idx"].to(torch.int64)
        width = topk.shape[1]
        past = torch.zeros(width, dtype=torch.int64)
        future_start = torch.zeros_like(past)
        future_count = torch.zeros_like(past)
        if self.definition["kind"] == "sparse_decode":
            block = self.scalars["block_size"]
            # The final partial block retains its exact load-mask work.
            past[:] = lengths.to(torch.int64) // block
        else:
            block, block_q = self.scalars["block_size_k"], self.scalars["block_size_q"]
            cu = state["cu_seqlens"].to(torch.int64).tolist()
            prefix = state["prefix_lens"].to(torch.int64).tolist()
            blocks = (state["cu_seqblocks_q"].to(torch.int64).tolist()
                      if "cu_seqblocks_q" in state else None)
            cursor = 0
            for batch, (start, end) in enumerate(zip(cu, cu[1:])):
                n = (end-start+block_q-1)//block_q
                begin = blocks[batch] if blocks is not None else cursor
                first = prefix[batch] + torch.arange(n)*block_q
                last = torch.minimum(first + block_q - 1, torch.full_like(first, prefix[batch]+end-start-1))
                full_blocks = int(lengths[batch])//block
                past[begin:begin+n] = ((first+1)//block).clamp_max(full_blocks)
                future_start[begin:begin+n] = last//block+1
                future_count[begin:begin+n] = (full_blocks-future_start[begin:begin+n]).clamp_min(0)
                cursor += n
        random = torch.randint(0, 2**31-1, (topk.shape[0], width, 1), generator=generator)
        p = past[None, :, None]
        f = future_start[None, :, None]
        n = future_count[None, :, None]
        mapped = torch.where((topk >= 0) & (topk < p), (topk + random) % p.clamp_min(1), topk)
        mapped = torch.where((topk >= f) & (n > 0) & (topk < f+n), f + (topk-f+random) % n.clamp_min(1), mapped)
        sentinel = torch.iinfo(torch.int64).max
        mapped = torch.where(topk >= 0, mapped, sentinel).sort(-1).values
        geometry["topk_idx"] = torch.where(mapped == sentinel, -1, mapped).to(state["topk_idx"].dtype)
        return geometry

    def validate_geometry(self, state):
        import torch
        table, slots, lengths = (state[name] for name in ("req_to_token", "slot_ids", "seq_lens"))
        if table.ndim != 2 or slots.ndim != 1 or lengths.ndim != 1 or slots.shape != lengths.shape:
            raise ValueError("captured request control shapes differ")
        self._physical_ids(state)
        if "topk_idx" not in state:
            return
        topk = state["topk_idx"]
        if topk.ndim != 3 or topk.shape[0] != self.tensors["k_cache"].shape[1]:
            raise ValueError("captured top-k head geometry differs")
        valid = topk >= 0
        if bool((valid[..., 1:] & ~valid[..., :-1]).any()) or bool((topk[~valid] != -1).any()):
            raise ValueError("captured top-k indices are not right padded with -1")
        if bool((valid[..., 1:] & (topk[..., 1:] < topk[..., :-1])).any()):
            raise ValueError("captured top-k indices are not sorted")
        if self.definition["kind"] == "sparse_decode":
            if topk.shape[1] != lengths.numel() or self.tensors["q"].shape[0] != lengths.numel():
                raise ValueError("captured decode batch geometry differs")
            maximum = (lengths.to(torch.int64) + self.scalars["block_size"] - 1) // self.scalars["block_size"]
        else:
            cu, prefix = state["cu_seqlens"].to(torch.int64), state["prefix_lens"].to(torch.int64)
            if cu.ndim != 1 or cu.numel() != lengths.numel()+1 or prefix.shape != lengths.shape:
                raise ValueError("captured ragged control shapes differ")
            sizes = cu[1:] - cu[:-1]
            if int(cu[0]) != 0 or bool((sizes < 0).any()) or int(cu[-1]) > self.tensors["q"].shape[0]:
                raise ValueError("captured ragged offsets escape the query buffer")
            if bool(((prefix < 0) | (prefix + sizes > lengths)).any()):
                raise ValueError("captured prefix/query lengths escape the sequence")
            block_counts = (sizes+self.scalars["block_size_q"]-1)//self.scalars["block_size_q"]
            blocks = torch.cat((torch.zeros(1, dtype=torch.int64), block_counts.cumsum(0)))
            if "cu_seqblocks_q" in state and not torch.equal(blocks, state["cu_seqblocks_q"].to(torch.int64)):
                raise ValueError("captured query block offsets differ from ragged work")
            if topk.shape[1] != int(blocks[-1]):
                raise ValueError("captured top-k query block geometry differs")
            per_request = (lengths.to(torch.int64)+self.scalars["block_size_k"]-1)//self.scalars["block_size_k"]
            maximum = per_request.repeat_interleave(block_counts)
        if bool((valid & (topk >= maximum[None, :, None])).any()):
            raise ValueError("captured sparse block escapes the sequence")

    def views(self, storage):
        import torch
        return {name: torch.empty(0, dtype=getattr(torch, spec["dtype"]), device=storage[self.aliases[name]].device).set_(
                    storage[self.aliases[name]].untyped_storage(), spec["storage_offset"],
                    tuple(spec["shape"]), tuple(spec["strides"])) for name, spec in self.specs.items()}

    def _physical_ids(self, geometry):
        import torch
        table, lengths, slots = (geometry[name] for name in ("req_to_token", "seq_lens", "slot_ids"))
        total = self.tensors["k_cache"].shape[0]
        result = []
        for length, slot in zip(lengths.reshape(-1).tolist(), slots.reshape(-1).tolist()):
            if not 0 <= length <= table.shape[1]:
                raise ValueError("captured sequence length escapes paging capacity")
            if length:
                row = (int(slot) + total) % total
                if not 0 <= row < table.shape[0]:
                    raise ValueError("captured active request row escapes the paging table")
                result.append((table[row, :length].to(torch.int64) + total) % total)
        return torch.unique(torch.cat(result)) if result else torch.empty(0, dtype=torch.int64)

    def reset(self, seed):
        import torch
        self.current_state = int(seed) % len(self.states)
        geometry = self._fresh_geometry(self.states[self.current_state], seed)
        for name, value in geometry.items():
            self.tensors[name].copy_(value.to(self.device))
        ids = self._physical_ids(geometry).to(self.device)
        for tag, name in enumerate(sorted(FLOAT_INPUTS & set(self.tensors))):
            target = self.tensors[name]
            generator = torch.Generator(device=self.device).manual_seed((int(seed)*17 + tag) % (2**63-1))
            shape = (ids.numel(), *target.shape[1:]) if name in ("k_cache", "v_cache") else tuple(target.shape)
            fresh = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32).to(target.dtype)
            if name == "sink" and int(seed) % 2:
                # Each element retains the standard-normal Q distribution,
                # while correlation makes the sink denominator consequential
                # for at least one query even with thousands of sparse tokens.
                fresh = self.tensors["q"][0].to(target.dtype)
            if name in ("k_cache", "v_cache"):
                target.view(torch.uint8).index_copy_(0, ids, fresh.view(torch.uint8))
            else:
                target.copy_(fresh)
        # Expected outputs are not available in GPU memory during candidate
        # execution. Immutable CPU bytes define the independent oracle inputs.
        return {alias: value.detach().to(device="cpu", copy=True) for alias, value in self.storage.items()}

    def restore_recorded(self, index):
        from minimax_fixtures import restore_recorded_inputs
        if not self.external:
            raise ValueError("recorded parity requires a verified external fixture")
        self.current_state = index
        self.validate_geometry(self.states[index])
        restore_recorded_inputs(self.entries[index], self.storage, self.verified_blobs)
        return {alias: value.detach().to(device="cpu", copy=True) for alias, value in self.storage.items()}

    def recorded_outputs(self, index):
        from minimax_fixtures import restore_cpu
        entry = self.entries[index]
        return restore_cpu(entry, "outputs", set(entry["fixture"]["outputs"]), self.verified_blobs)

    def reference_args(self, before):
        return {**self.scalars, **self.views(before)}

    def assert_immutable(self, before):
        import torch
        for alias, value in self.storage.items():
            if not torch.equal(value.detach().cpu(), before[alias]):
                raise AssertionError("kernel modified an input storage or its padding")

    def observe_arguments(self, result):
        import torch
        outputs = flatten_output(result)
        names = sorted(self.tensors)
        for index, left in enumerate(names):
            if "original_storage_nbytes" in self.case:
                if self.tensors[left].untyped_storage().nbytes() != self.case["original_storage_nbytes"][left]:
                    raise ValueError("runtime physical storage capacity differs")
            for right in names[index+1:]:
                actual = self.tensors[left].untyped_storage()._cdata == self.tensors[right].untyped_storage()._cdata
                if actual != (self.aliases[left] == self.aliases[right]):
                    raise ValueError("runtime input storage alias contract differs")
        structure = [{"name": name, "kind": "none" if value is None else "tensor"}
                     for name, value in sorted(outputs.items())]
        if structure != self.case["output_structure"]:
            raise ValueError("runtime output structure differs")
        tensors = {**self.tensors, **{name: value for name, value in outputs.items() if torch.is_tensor(value)}}
        controls = work_controls(self.args)
        if self.external:
            from minimax_work import structural_work
            raw = {"inputs."+name.removeprefix("work."): value for name, value in controls.items()}
            if raw != self.case["states"][self.current_state]["tensor_controls"]:
                raise ValueError("actual replay work controls differ from the selected real state")
            controls = {"work.variant": structural_work(raw, self.scalars)}
        scalars = {**self.scalars, **{name: value for name, value in outputs.items() if value is None}, **controls}
        return tensors, scalars


def snapshot_output(value):
    import torch
    return {name: None if item is None else item.detach().to(device="cpu", copy=True)
            for name, item in flatten_output(value).items()}


def initialize_outputs(value):
    import torch
    for tensor in flatten_output(value).values():
        if tensor is None:
            continue
        if tensor.is_floating_point():
            tensor.fill_(float("nan"))
        else:
            tensor.fill_(torch.iinfo(tensor.dtype).min)
