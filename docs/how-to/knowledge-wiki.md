---
myst:
    html_meta:
        "description": "Offer a read-only kernel knowledge base to the GEAK agent of an AgentKernelArena run, with every query logged."
        "keywords": "AgentKernelArena, knowledge base, ROCmKernelWiki, GEAK, A/B testing"
---

# Offer a kernel knowledge base to agents

A run can give its agents read-only access to a kernel knowledge base checkout
that follows the [KernelWiki](https://github.com/mit-han-lab/KernelWiki) layout,
such as [ROCmKernelWiki](https://github.com/jhinpan/ROCmKernelWiki). The feature
exists for A/B experiments that compare the same agent with and without the
knowledge base. Both arms of such an experiment must differ only in this setting.

The GEAK agent supports it. A run that enables it for any other agent fails
before its first task.

## Enable it

Add the run-level key to the configuration of the knowledge-base arm only:

```yaml
agent:
  template: geak
knowledge_wiki:
  enabled: true
```

Name the checkout on the host when starting the run:

```bash
AKA_KNOWLEDGE_WIKI_ROOT=/path/to/ROCmKernelWiki make docker-run CONFIG="$CONFIG_PATH"
```

For a run that enables `knowledge_wiki`, the Docker runner mounts that checkout
read-only at the same path and forwards `AKA_KNOWLEDGE_WIKI_ROOT`. Runs that do
not enable it neither mount the checkout nor receive the variable. Keep the checkout
outside the AgentKernelArena repository, because the repository root is mounted
into every container.

The checkout must provide `scripts/query.py`, `scripts/get_page.py` and
`scripts/grep_wiki.py`. The run stops with an error when the key is enabled but
the variable or these scripts are missing.

## What agents see

GEAK appends the instructions from `src/knowledge_wiki.py` to the Arena contract
on every role and Workflow call, and its agents query from the shell:

```text
<python> src/knowledge_wiki.py --root <checkout> --log <log> search "<question>" [filters]
<python> src/knowledge_wiki.py --root <checkout> --log <log> page <page-id>
<python> src/knowledge_wiki.py --root <checkout> --log <log> grep "<regex>"
```

The command passes the knowledge base's own output and exit status through
unchanged. The instructions name only the options these knowledge bases share
and point agents at each subcommand's `--help` for the rest.

## Logs and identity

Every query appends one JSON line to `knowledge_wiki_calls.jsonl` in the hidden
GEAK run directory. Each line records the timestamp, subcommand, arguments, exit
code, duration, output size, and the page IDs or files returned.

The adapter status records the checkout's commit, whether its tree was dirty,
and the instructions version and the SHA-256 of the text it delivered. GEAK also
writes them to `knowledge_wiki.json`, and the run log repeats them at startup.
Compare these values across arms before comparing results.

A knowledge-base arm whose log is empty received the instructions but never used
the knowledge base. Report that count rather than dropping the run.
