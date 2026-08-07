# Direct Codex agent

This module is AgentKernelArena's direct-Codex treatment. `launch_agent.py`
constructs one fresh Codex session per attempt, captures bounded structured output,
checks the immutable campaign contract, and exposes only the declared task source to
central evaluation. It does not grade or select its own proposal.

## Formal campaign isolation

Matched Apex-versus-Codex campaigns mount `formal_requirements.toml` read-only at
`/etc/codex/requirements.toml`. Its SHA-256 is pinned in
`src/campaign_isolation.py`. The policy permits only
`aka_formal_kernel_v1`, derived from Codex's workspace profile, with:

- approval policy `never`;
- command network disabled;
- sandboxed reads of `~/.codex/auth.json` denied;
- managed and project hooks disabled;
- no web-search mode allowed.

The outer attempt boundary is a rootless bubblewrap mount/IPC namespace. It hides
all other campaign data and gives the attempt private `/tmp` and `/dev/shm`. It
preserves the Docker worker's private PID namespace and bind-mounts that private
`/proc` read-write because nested Codex bubblewrap needs procfs to create its own
user-namespace sandbox. Docker still runs the worker as non-root with every
capability dropped and `no-new-privileges`; Yama plus live probes must deny access
to the outer process's root, file descriptors, environment, and memory.

Campaign preflight executes the real Codex binary through this exact two-layer
boundary. `codex sandbox` requires an explicit profile selector, so this live probe
selects `aka_formal_kernel_v1` directly; it tests the effective profile rather than
claiming to re-run the agent launcher's legacy-mode normalization. The real Apex and
direct launchers both request legacy `workspace-write`. The managed allowlist forces
that request to the sole permitted profile for the receipt-pinned Codex binary, and
any CLI identity mismatch makes the treatments incomparable. The live probe must
prove workspace writes succeed while credential reads and command network fail. It
must also prove that Codex creates a distinct inner PID namespace.
Its inherited procfs can expose the outer status entry, but the probe requires the
outer root/fd/environ/mem aliases to remain unreadable. The receipt records the
Codex and bubblewrap identities, the managed requirements hash,
and every property result. Init, workers, and postprocess must
reproduce the same receipt, and the Apex and direct-Codex comparison contracts must
match before results are comparable.

Do not treat Codex's legacy `sandbox: workspace-write` session label as proof of the
effective policy. The pinned managed file and successful negative live probes are
the authoritative evidence.
