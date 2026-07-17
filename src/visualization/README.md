# AgentKernelArena visualization module

`src.visualization` builds and serves the static dashboard used to compare
baseline, treatment, and other AgentKernelArena experiment reports.

The module keeps three kinds of files separate:

- Python implementation and static frontend assets live in `src/visualization/`.
- Generated dashboard payloads live in `.visualization/dashboard/`.
- Optional manually collected report bundles live in
  `.visualization/reports/<report_name>/`.

Normal run reports remain under:

```text
workspace_<gpu>_<agent>/run_<timestamp>/reports/
```

## Commands

Run these commands from the AgentKernelArena repository root.

Build a dashboard from normal workspace runs:

```bash
python3 -m src.visualization build --include-workspace-runs
```

Serve an already-built dashboard:

```bash
python3 -m src.visualization serve --host 127.0.0.1 --port 8080
```

Build and serve in one command:

```bash
python3 -m src.visualization run \
  --include-workspace-runs \
  --host 127.0.0.1 \
  --port 8080
```

Equivalent Make targets are available from the repository root:

```bash
make visualization-build
make visualization-serve
make visualization-run
```

Without `--include-workspace-runs`, the builder scans only
`.visualization/reports/<report_name>/`. Each report directory must contain:

- `overall_summary.csv`
- `task_type_breakdown.json`
- `overall_report.txt`

## Structure

```text
src/visualization/
├── __init__.py
├── __main__.py
├── build_data.py
├── paths.py
├── server.py
└── frontend/
    ├── index.html
    └── dashboard/
        ├── app.js
        └── styles.css
```

The server exposes only the dashboard assets plus `.csv`, `.json`, and `.txt`
report artifacts. Directory listings and path traversal are rejected.

For non-default layouts, set `AKA_PROJECT_ROOT` or
`AKA_VISUALIZATION_RUNTIME_ROOT` before invoking the module.
