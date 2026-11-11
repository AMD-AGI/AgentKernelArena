# SPDX-License-Identifier: Apache-2.0
# Vendored for AgentKernelArena from sgl-project/sgl-amd-mono at 52486703fbf27e77f05b573e4d931d4eec87118d,
# sgl_amd_mono/kimi_k3/gfx950/tag.py. AKA changes: relative imports only.
"""One Kimi-K3 launch's mailbox tag.

This model bumps the epoch on the host and gives each launch its own slot.
DeepSeek-V4.1 advances its epoch on the device (``common/epoch.py``). The
difference is a decision, not an omission. The rest of this docstring is why.

A tag must tell a pair apart from every other pair that can be live at the same
address. Those are the step's other launches, and the previous step's. The two
models do that differently.

* Here: the host bumps the epoch once a forward. That is a device op, so a
  captured graph replays it. Each launch also carries its own slot. A step's 92
  launches then differ by slot, and the steps differ by epoch. The peer
  buffer's parity follows the epoch, so one step's launches share a half. That
  is safe, because their tags already differ by slot.
* DeepSeek-V4.1: the epoch advances once a launch. That needs ``epoch_end``.
  Before the epoch can move, CTA 0 waits for all ``BLOCKS`` CTAs to mark that
  they have read the old one.

That wait is a device-wide barrier at the end of every launch. DeepSeek-V4.1
runs it twice a layer. Kimi-K3 would run it 92 times a step, to replace a
scheme that already gives distinct tags. The barrier removes a hazard that the
host-bumped epoch does not have. Taking it on would cost something for
uniformity alone. At this model's step time that cost is below what an
end-to-end measurement resolves, so it could not even be shown to be free.

Revisit this if a launch ever has to advance the epoch itself, for example if a
layer needs a second launch. The slot is then no longer enough, and
``epoch_end`` is the answer.
"""

from .ops import ld_i32, uniform

TAG_SLOTS = 256  # a step's mailbox tags: epoch x 256 + a launch's slot


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch (a device counter the runner
    bumps once a step, in the graph) and the launch's slot, so no step's or
    other launch's pair ever passes for this one's -- nothing is zeroed."""
    e = uniform(ld_i32(epoch, 0))
    return (e % (1 << 22)) * TAG_SLOTS + slot + 1
