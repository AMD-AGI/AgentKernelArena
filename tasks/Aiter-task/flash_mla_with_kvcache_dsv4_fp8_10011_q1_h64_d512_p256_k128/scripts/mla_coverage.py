"""Task-local runtime-length and padding coverage, independent of GPU libraries."""


def boundary_lengths(width):
    values = (0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129,
              255, 256, 257, 511, 512, 513, 1024, 2048, 4096,
              8191, 8192, 8193, width - 1, width)
    return sorted({n for n in values if 0 <= n <= width})


def profiles(main_width, extra_width=None):
    """Vary one branch while holding the other fixed, plus joint empty cases."""
    widths = {"main": main_width}
    if extra_width is not None:
        widths["extra"] = extra_width
    result = []
    for branch, width in widths.items():
        for length in boundary_lengths(width):
            lengths = dict(widths, **{branch: length})
            result.append({"id": f"{branch}_length_{length}",
                           "lengths": lengths, "padding": {}})
        for length in (1, width):
            for mode in ("holes", "all_negative"):
                result.append({"id": f"{branch}_{mode}_{length}",
                               "lengths": dict(widths, **{branch: length}),
                               "padding": {branch: mode}})
    result.append({"id": "all_branches_empty", "lengths": dict.fromkeys(widths, 0),
                   "padding": {}})
    if extra_width is not None:
        for main, extra in ((0, 1), (1, 0), (1, 1), (33, 65)):
            result.append({"id": f"independent_{main}_{extra}",
                           "lengths": {"main": main, "extra": extra}, "padding": {}})
    return result


def apply_profile(values, profile=None, *, replay=False):
    """Keep nonnegative tail indices; length and negative masking are independent."""
    import torch
    for branch, prefix in (("main", ""), ("extra", "extra_")):
        if prefix + "sparse_indices" not in values:
            continue
        indices, lengths = values[prefix + "sparse_indices"], values[prefix + "sparse_lens"]
        width, batch = indices.shape[-1], indices.shape[0]
        choices = boundary_lengths(width)
        if profile is None:
            # Larger shape cases combine boundary rows and independent random rows.
            # The initializer already supplied legal random lengths in every row.
            if batch == 1:
                lengths.fill_(width if not replay else max(1, width // 3))
            else:
                offset = 0 if branch == "main" else 1
                count = min(batch, max(2, batch // 2), len(choices))
                for i in range(count):
                    lengths[i] = choices[(i + offset + int(replay)) % len(choices)]
                lengths[-1] = width
        else:
            length = profile["lengths"][branch]
            if replay:
                # The measured graph must consume new length tensor contents.
                length = choices[(choices.index(length) + 1) % len(choices)]
            lengths.fill_(length)
        positions = torch.arange(width, device=indices.device)[None, None, :]
        active = positions < lengths[:, None, None]
        mode = (profile or {}).get("padding", {}).get(branch, "dense")
        if mode == "all_negative":
            indices.masked_fill_(active, -1)
        elif mode == "holes":
            # Include internal holes, not just a suffix agreeing with lengths.
            indices.masked_fill_(active & ((positions % 7) == 0), -1)
        elif profile is None and batch > 1:
            rows = torch.arange(batch, device=indices.device)[:, None, None]
            indices.masked_fill_(active & (rows % 3 == 1) & (positions % 7 == 0), -1)
