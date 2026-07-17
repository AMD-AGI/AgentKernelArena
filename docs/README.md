# AgentKernelArena documentation

This directory contains the source for the AgentKernelArena documentation.
It follows the same structure as other ROCm toolkit component docs and is built
with [Sphinx](https://www.sphinx-doc.org/) and
[rocm-docs-core](https://github.com/ROCm/rocm-docs-core). Both Markdown (MyST,
`.md`) and reStructuredText (`.rst`) sources are supported and can be mixed.

## Building locally

```bash
# From the repository root
python3 -m venv .docvenv
source .docvenv/bin/activate
pip install -r docs/sphinx/requirements.txt

python -m sphinx -T -b html docs docs/_build/html
# Open docs/_build/html/index.html
```

## Layout

| Path | Page | Notes |
| --- | --- | --- |
| `index.rst` | Overview | Landing page with links to the main documentation sections and the GitHub repo. |
| `what-is-aka.rst` | Overview | Product overview, key features, and use cases. |
| `install/install.md` | Installation | Docker runner (`make docker-smoke`/`docker-run`), agent CLIs, and API keys. |
| `reference/release-notes.md` | Release Notes | Per-release feature breakdown; initial release lists all features. |
| `reference/compatibility-matrix.md` | Compatibility Matrix | Verified hardware/software versions. |
| `reference/api-reference.md` | Configuration and API reference | `config.yaml` schema, task `config.yaml` schema, CLI flags, scoring, and the agent registry. |
| `reference/benchmark-methodology.md` | Reference | Timing methodology, benchmark helper materialization, and speedup interpretation. |
| `how-to/run-evaluation.md` | How-to | Configure `config.yaml`, run through Docker, resume runs, and read results. |
| `how-to/parallel-run.md` | How-to | Run one isolated Docker worker per GPU, use the shared `.parallel/` task queue, resume parallel runs, and parallelize `task_validator`. |
| `how-to/agents.md` | How-to | Supported agents, model providers, and A/B testing. |
| `how-to/add-task.md` | How-to | Task directory layout, `config.yaml` fields, and task types. |
| `how-to/task-validator.md` | How-to | The task_validator agent and its 10 checks. |
| `how-to/visualization.md` | How-to | Build dashboard data and serve the comparison dashboard. |
| `examples/examples.md` | Examples | Step-by-step walkthroughs with expected output. |
| `about/license.md` | License | Full Apache 2.0 license reference (mirrors the repo `LICENSE`). |

Navigation (the left sidebar) is defined in `sphinx/_toc.yml.in`.

## Configuration files

| File | Purpose |
| --- | --- |
| `../.readthedocs.yaml` | Read the Docs build configuration. |
| `conf.py` | Sphinx configuration (rocm-docs-core, MyST, mermaid). |
| `sphinx/_toc.yml.in` | Table of contents / sidebar navigation. |
| `sphinx/requirements.in` | Top-level documentation dependencies. |
| `sphinx/requirements.txt` | Pinned documentation dependencies used by the build. |

## Notes on the local build

A local build outside AMD infrastructure prints benign warnings for intersphinx
inventory fetches and a `Current project 'AgentKernelArena' not found in
projects` message until the project is registered in the rocm-docs-core project
registry. The build itself succeeds.
