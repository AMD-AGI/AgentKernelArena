# DeepSeek Harness agent

This integration runs the upstream [`dsh` CLI](https://github.com/deepseek-ai/deepseek-harness)
in its one-shot `headless` profile. It uses Arena's standard task prompt,
works in the copied task workspace, and leaves compilation, correctness,
performance measurement, harness protection, and scoring to Arena.

For interactive use outside Arena, upstream provides `dsh web`. Arena uses
`dsh --profile headless --patch <invocation-patch> --json`, with the complete
prompt on stdin. See the upstream [headless guide](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/bundle/headless/README.md).

## Install and configure

Use a dedicated Linux Node.js 24 installation on the GPU host, with `node`
and npm's global `dsh` executable in the same prefix. The Docker runner mounts
that prefix read-only. It can be selected explicitly through `AKA_NODE_PREFIX`.
Install the version from [agent_config.yaml](agent_config.yaml), not `latest`:
the CLI is in developer preview, and earlier releases lack stdin/JSON support.

From the Arena repository root:

```bash
DSH_VERSION=$(sed -n 's/^cli_version: //p' agents/deepseek_harness/agent_config.yaml)
npm install --global --ignore-scripts "@deepseek-ai/dsh@$DSH_VERSION"
dsh --version

# Enter the key without putting its literal value in shell history.
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
```

Authentication is through `DEEPSEEK_API_KEY`; Arena does not import Web UI
login state or host `~/.dsh` profiles. The launcher requires the configured CLI
version exactly. Record the Node version and installed dependency tree
(`npm ls --global --all --json`) with your experiment environment; the top-level
CLI pin alone does not freeze npm's transitive dependency resolution.

Agent settings live in [agent_config.yaml](agent_config.yaml):

| Setting | Meaning |
| --- | --- |
| `cli_version` | Required installed CLI version; update only after checking compatibility |
| `model` | Model ID for the upstream `deepseek-official` provider |
| `reasoning_effort` | `off`, `low`, `high`, or `max` |
| `protocol` | `chat-completions` or `messages`; must match the endpoint |
| `base_url` | Endpoint override; null uses `DEEPSEEK_BASE_URL`, then upstream's protocol default |
| `max_tokens` | Per-request output limit, not a total session budget |
| `max_iterations` | Optimization guidance appended to the prompt, not a hard turn limit |
| `timeout_seconds` | Wall-clock budget; the launcher stops the process group on expiry |
| `python_path` | Optional in-container Python interpreter; null uses Arena's interpreter |

This adapter configures the DeepSeek provider. Other upstream providers and
custom plugins are not exposed by this integration. For gateways, use a URL
compatible with the chosen protocol; do not put credentials in the URL or YAML.

## Run through Docker

Select the configuration matching the physical GPU:

```bash
CONFIG_PATH=example_configs/quickstart_deepseek_harness_mi300.yaml
# On MI355X instead:
# CONFIG_PATH=example_configs/quickstart_deepseek_harness_mi355x.yaml

make docker-smoke
make docker-check-agents CONFIG="$CONFIG_PATH"
make docker-run CONFIG="$CONFIG_PATH"

# Multiple GPUs, with an independent session per task:
make docker-parallel-run CONFIG="$CONFIG_PATH" GPU_IDS=0,1
```

`docker-check-agents` checks executable availability, the exact CLI version,
and API-key presence without making a paid API request. It does not prove that
the key, endpoint, or selected model will accept requests. An explicit
`AGENTS=deepseek_harness` also works. `AGENTS=all` retains its existing meaning
(Cursor, Claude Code, and Codex); DeepSeek remains opt-in.

On a GPU-less Slurm/Spur login node, install the Node prefix on shared storage,
export the same environment variables, and use:

```bash
CONFIG_PATH=example_configs/quickstart_deepseek_harness_mi355x.yaml
make slurm-smoke
make slurm-check-agents CONFIG="$CONFIG_PATH"
make slurm-run CONFIG="$CONFIG_PATH" RUN_ARGS="--run-suffix deepseek_smoke"
```

See [the Slurm guide](../../docs/how-to/slurm-run.md) for resource selection and
batch submission. The single-GPU MI355X path has been exercised with HIP GELU
and Triton RMSNorm; MI300 and multi-GPU runs remain unverified on hardware.

Every invocation creates a fresh `.deepseek_harness-*` directory inside its
task workspace. It preserves the prompt, credential-free Cordis patch,
invocation settings, stdout/stderr logs, and a separate `home/` for upstream
sessions. Retries create another directory rather than overwriting past runs.
Nonzero CLI exits and timeouts fail the task instead of reporting completion.
Output captured by the adapter redacts the configured API key; upstream session
files are third-party artifacts and should be reviewed before sharing.

## Security and reproducibility review

- Runtime execution uses an installed, version-checked `dsh` binary with an
  argument list and file-backed stdin. The launcher does not invoke a shell,
  `npx`, install packages, clone repositories, or download plugins during a run.
  Installation is an explicit dependency setup step; npm packages are
  third-party code and the documented installation disables install hooks.
  Use a dedicated Node prefix containing only the runtime and required packages.
- Docker mounts that installation read-only and forwards only
  `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` for explicitly selected DeepSeek
  runs. Secrets are forwarded by environment variable name, not embedded in
  Docker arguments, prompts, patches, or invocation metadata.
- Each invocation gets a new `DSH_HOME`. Inherited `DSH_*` overrides are removed,
  OTel telemetry is disabled, and `session-log-deepseek.enabled` is false.
  Model requests still send the task context and tool results to the configured
  provider; the patch disables the additional session-log contribution.
- `danger-full-access` makes GPU compilation and tools noninteractive inside
  the existing Arena container. Like the other Arena integrations, this is
  privileged agent execution, not a security sandbox. Process-group cleanup
  stops ordinary tool descendants before returning to the evaluator, including
  after a normal parent exit. Tools that deliberately create a separate session
  remain outside that process-group boundary.

The upstream interface was inspected at commit
`ddefc45fbc7f8e46dd73185e68295696d1297887` and against the published packages for
the configured version. Relevant sources are the upstream
[CLI reference](https://github.com/deepseek-ai/deepseek-harness/blob/ddefc45fbc7f8e46dd73185e68295696d1297887/apps/cli/reference/README.md),
[DeepSeek provider](https://github.com/deepseek-ai/deepseek-harness/blob/ddefc45fbc7f8e46dd73185e68295696d1297887/packages/llm/llm-deepseek/README.md),
and [safety notice](https://github.com/deepseek-ai/deepseek-harness/blob/ddefc45fbc7f8e46dd73185e68295696d1297887/SAFETY.md).

Offline regression checks:

```bash
python3 -m pytest -q tests/test_deepseek_harness.py
make check-docker-runner
```

These checks exercise a local fake CLI and Docker argument construction, not
model quality, API access, GPU correctness, or performance. Validate those with
a Docker task run on compatible hardware before using results in a comparison.
