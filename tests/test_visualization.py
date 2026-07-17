import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from src.visualization import build_data
from src.visualization.__main__ import create_parser
from src.visualization.server import resolve_request_path


def write_report(report_dir: Path, *, agent: str, gpu: str, score: float) -> None:
    report_dir.mkdir(parents=True)
    (report_dir / "overall_summary.csv").write_text(
        "Task Name,Task Type,Score,Speedup,Optimization_summary\n"
        f"hip2hip/gpumode/GELU,hip2hip,{score},1.5,optimized GELU\n"
    )
    (report_dir / "task_type_breakdown.json").write_text(
        json.dumps(
            {
                "agent": agent,
                "target_gpu": gpu,
                "run_timestamp": "20260715_120000",
                "overall": {
                    "total_score": score,
                    "average_speedup": 1.5,
                    "median_speedup": 1.5,
                    "correctness_pass_rate": 1.0,
                    "compilation_pass_rate": 1.0,
                },
                "task_types": {"hip2hip": {"total_score": score}},
            }
        )
    )
    (report_dir / "overall_report.txt").write_text(
        f"PASS hip2hip/gpumode/GELU Score: {score} Speedup: 1.5x\n"
    )


class VisualizationBuildTests(unittest.TestCase):
    def test_local_reports_are_default_and_workspace_reports_are_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            runtime_root = project_root / ".visualization"
            reports_root = runtime_root / "reports"
            data_root = runtime_root / "dashboard"
            local_report = reports_root / "manual_baseline"
            workspace_report = (
                project_root
                / "workspace_MI300_claude_code"
                / "run_20260715_120000"
                / "reports"
            )
            write_report(local_report, agent="claude_code", gpu="MI300", score=200.0)
            write_report(workspace_report, agent="claude_code", gpu="MI300", score=300.0)

            with mock.patch.multiple(
                build_data,
                PROJECT_ROOT=project_root,
                REPORTS_ROOT=reports_root,
                DATA_ROOT=data_root,
                OUTPUT_JSON=data_root / "data.json",
                OUTPUT_JS=data_root / "data.js",
            ):
                local_dataset = build_data.build_dataset()
                self.assertEqual(local_dataset["meta"]["reportCount"], 1)
                self.assertEqual(
                    local_dataset["reports"][0]["sourceFiles"]["summaryCsv"],
                    "reports/manual_baseline/overall_summary.csv",
                )

                with redirect_stdout(io.StringIO()):
                    full_dataset = build_data.write_dashboard_data(
                        include_workspace_runs=True
                    )
                self.assertEqual(full_dataset["meta"]["reportCount"], 2)
                self.assertTrue((data_root / "data.json").is_file())
                self.assertTrue((data_root / "data.js").is_file())
                source_data = (
                    project_root
                    / "src"
                    / "visualization"
                    / "frontend"
                    / "dashboard"
                    / "data.js"
                )
                self.assertFalse(source_data.exists())
                source_paths = {
                    report["sourceFiles"]["summaryCsv"]
                    for report in full_dataset["reports"]
                }
                self.assertIn(
                    "artifacts/workspace_MI300_claude_code/"
                    "run_20260715_120000/reports/overall_summary.csv",
                    source_paths,
                )


class VisualizationServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.frontend = self.root / "src" / "visualization" / "frontend"
        self.data = self.root / ".visualization" / "dashboard"
        self.reports = self.root / ".visualization" / "reports"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def resolve(self, path: str) -> Path | None:
        return resolve_request_path(
            path,
            project_root=self.root,
            frontend_root=self.frontend,
            data_root=self.data,
            reports_root=self.reports,
        )

    def test_routes_static_generated_and_report_files(self) -> None:
        self.assertEqual(self.resolve("/"), self.frontend / "index.html")
        self.assertEqual(
            self.resolve("/dashboard/app.js"), self.frontend / "dashboard" / "app.js"
        )
        self.assertEqual(
            self.resolve("/dashboard/data.js"), self.data / "data.js"
        )
        self.assertEqual(
            self.resolve("/reports/manual/overall_summary.csv"),
            self.reports / "manual" / "overall_summary.csv",
        )
        self.assertEqual(
            self.resolve("/artifacts/workspace_x/run_x/reports/overall_report.txt"),
            self.root / "workspace_x" / "run_x" / "reports" / "overall_report.txt",
        )

    def test_rejects_traversal_hidden_paths_and_non_report_artifacts(self) -> None:
        self.assertIsNone(self.resolve("/%2e%2e/README.md"))
        self.assertIsNone(self.resolve("/artifacts/.git/config"))
        self.assertIsNone(self.resolve("/artifacts/README.md"))
        self.assertIsNone(self.resolve("/reports/manual/kernel.py"))
        self.assertIsNone(self.resolve("/unknown/file.txt"))


class VisualizationCliTests(unittest.TestCase):
    def test_cli_exposes_build_serve_and_run_commands(self) -> None:
        parser = create_parser()
        self.assertEqual(parser.parse_args(["build"]).command, "build")
        self.assertEqual(parser.parse_args(["serve"]).port, 8080)
        run_args = parser.parse_args(["run", "--include-workspace-runs"])
        self.assertTrue(run_args.include_workspace_runs)
        self.assertEqual(run_args.host, "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
