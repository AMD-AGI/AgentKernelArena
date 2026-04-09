# AgentKernelArena Visualization

Static dashboard for comparing run reports produced by AgentKernelArena.

The app lives under `visualization/`, but it scans the parent AgentKernelArena
repository for report directories that contain:

- `overall_summary.csv`
- `task_type_breakdown.json`
- `overall_report.txt`

In the standard AgentKernelArena layout, those files are usually located at:

```text
workspace_<gpu>_<agent>/run_<timestamp>/reports/
```

## Structure

```text
visualization/
├── backend/
│   ├── server.py
│   └── scripts/
│       └── build_dashboard_data.py
├── frontend/
│   ├── index.html
│   └── dashboard/
│       ├── app.js
│       ├── styles.css
│       ├── data.js
│       └── data.json
├── .gitignore
├── README.md
└── setup.sh
```

`frontend/dashboard/data.js` and `frontend/dashboard/data.json` are generated files.

## Usage

From the `visualization/` directory:

```bash
python backend/scripts/build_dashboard_data.py
python backend/server.py --host 127.0.0.1 --port 8080
```

Then open:

```text
http://127.0.0.1:8080
```

## Serve on Port 80

Port `80` usually requires elevated privileges:

```bash
sudo python backend/server.py --host 0.0.0.0 --port 80
```

Or use the helper script:

```bash
bash setup.sh
```

`setup.sh` rebuilds the dashboard data first, then starts the HTTP service on port `80`.

## Rebuild After New Runs

Whenever AgentKernelArena produces new run reports, rebuild the dashboard payload:

```bash
python backend/scripts/build_dashboard_data.py
```

The dashboard will automatically pick up newly discovered report directories on the next refresh.

## Notes

- The HTTP service serves the dashboard UI from `frontend/`.
- Source-file links inside the dashboard are exposed through an `/artifacts/...` route.
- `/artifacts/...` only allows `.csv`, `.json`, and `.txt` files.
- If no reports are found yet, the dashboard still builds and shows an empty state.
