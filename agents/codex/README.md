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

The outer attempt boundary is a gated rootless bubblewrap mount/PID/IPC namespace.
It hides all other campaign data and gives the attempt private `/tmp`, `/dev/shm`,
and `/proc`. Bubblewrap reports namespace PID 1 over a status pipe before the gate
opens; the launcher binds that exact process with a pidfd and verifies its starttime,
parent, inner PID, and namespace inodes. Docker still runs the worker as non-root
with every capability dropped and `no-new-privileges`. Because formal Docker enables
`systempaths=unconfined` to permit the procfs mount, bubblewrap immediately rebuilds
Docker's exact masked and read-only system paths. Live probes verify mount identities,
prove the worker PID namespace is absent, and compare parent root/fd aliases against
secret bytes rather than trusting numeric PID errno results.

Campaign preflight executes the real Codex binary through this exact two-layer
boundary. `codex sandbox` requires an explicit profile selector, so this live probe
selects `aka_formal_kernel_v1` directly; it tests the effective profile rather than
claiming to re-run the agent launcher's legacy-mode normalization. The real Apex and
direct launchers both request legacy `workspace-write`. The managed allowlist forces
that request to the sole permitted profile for the receipt-pinned Codex binary, and
any CLI identity mismatch makes the treatments incomparable. The live probe must
prove workspace writes succeed while credential reads and command network fail. It
must also prove that Codex creates a distinct inner PID namespace, sees exactly one
ROCm GPU, can open the assigned GPU device nodes read-write, and completes a Torch
allocation plus reduction on that device. Codex normally replaces `/dev` inside
its Linux bubblewrap sandbox. The content-pinned `bin/bwrap` compatibility shim
re-binds only `/dev/kfd` and the render nodes already admitted by Docker; any
missing, extra, non-character, or changed shim/device input fails closed.
The outer namespace mounts this shim read-only at
`/tmp/aka-codex-gpu-bwrap/bwrap`, outside the task workspace that Codex excludes
from system-bubblewrap discovery. Its parent is a dedicated tmpfs mountpoint that
is remounted read-only, so writable `/tmp` cannot rename or replace the trusted
pathname. Before the mount, the verified shim bytes are copied to a sealed memfd
and passed through `--ro-bind-data`, closing the content replacement window between
hashing and execution. The live probe requires directory rename plus file
unlink/replace/write attacks to fail. `--help` and `--version` are delegated
unchanged to the independently content-pinned `/usr/bin/bwrap`; real sandbox
invocations must set the shim activation marker that the same probe verifies.
The fixed `/usr/bin/python3 -I` shebang disables user-site and `PYTHON*` startup
injection before any shim code runs.
The managed probe also blocks PID-1 root/environ/mem credential aliases. The receipt records the
Codex, outer bubblewrap, and GPU-shim identities, the managed requirements hash,
and every property result. Init, workers, and postprocess must reproduce the same
receipt, and the Apex and direct-Codex comparison contracts must match before
results are comparable.

Every formal session receipt directly binds `comparison_contract_sha256`. The
postprocessor recomputes that digest from the immutable campaign manifest and
rejects any missing or mismatched binding. Direct Codex additionally publishes the
exact UTF-8 rendered prompt as a read-only receipt artifact; the verifier hashes
those bytes and requires the result to equal `invocation.prompt_sha256`.

Receipt v4 enforces exact-turn source persistence with
`private_pid_namespace_init_pidfd_v1`. At turn 50, the reader immediately signals the
namespace init through its pidfd. Linux destroys every member, including descendants
that call `setsid()`, double-fork, clear their environment, close stdio, or immediately
`exec`. Source capture occurs only after init exit, wrapper terminal status or the
explicit SIGKILL-status absence case, status EOF, complete stdout/stderr EOF, and an
enumeration with no supervisor-visible namespace member. Inaccessible sibling
`/proc` entries are counted and make the scan explicitly incomplete; the exact
namespace-init pidfd exit and Linux PID-namespace teardown semantics remain the
authority. The natural-exit race must establish the same proof. The post-teardown
checkpoint and retained output tail are digested into the candidate-persistence
receipt. The verifier independently recomputes the source
delta from sealed before/after manifests. A contradictory delta invalidates the
receipt. A valid empty delta is instead an audited `no_gain`: the attempt remains
completed, but its baseline replay is selection-ineligible and contributes no
canonical workspace. If all three sessions are `no_gain`, the task has no canonical
result; if another session produced a valid candidate, selection considers only the
candidate sessions.

Do not treat Codex's legacy `sandbox: workspace-write` session label as proof of the
effective policy. The pinned managed file and successful negative live probes are
the authoritative evidence.
