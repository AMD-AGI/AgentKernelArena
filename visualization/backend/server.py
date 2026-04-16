#!/usr/bin/env python3
"""Small static HTTP service for the Arena dashboard."""

from __future__ import annotations

import argparse
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


VISUALIZATION_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = VISUALIZATION_ROOT.parent
FRONTEND_ROOT = VISUALIZATION_ROOT / "frontend"
INDEX_FILE = FRONTEND_ROOT / "index.html"
ROUTE_ROOTS = {
    "dashboard": FRONTEND_ROOT / "dashboard",
    "artifacts": PROJECT_ROOT,
}
ARTIFACT_EXTENSIONS = {".csv", ".json", ".txt"}


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve only the dashboard assets needed by the static UI."""

    server_version = "ArenaDashboardHTTP/1.0"

    def translate_path(self, path: str) -> str:
        normalized = self._normalize_request_path(path)
        if normalized is None:
            return str(VISUALIZATION_ROOT / "__forbidden__")
        return str(normalized)

    def do_GET(self) -> None:
        if self._normalize_request_path(self.path) is None:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self._normalize_request_path(self.path) is None:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        super().do_HEAD()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def list_directory(self, path: str):  # type: ignore[override]
        self.send_error(HTTPStatus.NOT_FOUND, "Directory listing is disabled")
        return None

    def guess_type(self, path: str) -> str:
        if path.endswith(".csv"):
            return "text/csv; charset=utf-8"
        if path.endswith(".json"):
            return "application/json; charset=utf-8"
        if path.endswith(".txt"):
            return "text/plain; charset=utf-8"
        return mimetypes.guess_type(path)[0] or "application/octet-stream"

    def _normalize_request_path(self, raw_path: str) -> Path | None:
        path = raw_path.split("?", 1)[0].split("#", 1)[0]
        if path in ("", "/"):
            return INDEX_FILE

        if path == "/index.html":
            return INDEX_FILE

        clean = path.lstrip("/")
        if not clean:
            return INDEX_FILE

        if clean.startswith(".") or "/." in clean:
            return None

        candidate = Path(clean)
        if candidate.is_absolute():
            return None

        if not candidate.parts:
            return None

        route_name = candidate.parts[0]
        route_root = ROUTE_ROOTS.get(route_name)
        if route_root is None:
            return None

        resolved = (route_root / Path(*candidate.parts[1:])).resolve()
        try:
            resolved.relative_to(route_root.resolve())
        except ValueError:
            return None

        if route_name == "artifacts" and resolved.is_file() and resolved.suffix not in ARTIFACT_EXTENSIONS:
            return None

        if clean == route_name:
            return route_root

        return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Arena dashboard over HTTP.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind. Default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=80, help="TCP port to bind. Default: 80")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        with ThreadingHTTPServer((args.host, args.port), DashboardHandler) as httpd:
            print(f"Serving {FRONTEND_ROOT} at http://{args.host}:{args.port}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        raise SystemExit("Server stopped.")
    except PermissionError:
        raise SystemExit(
            f"Permission denied binding to {args.host}:{args.port}. "
            "Port 80 usually requires elevated privileges."
        )
    except OSError as exc:
        raise SystemExit(f"Failed to bind {args.host}:{args.port}: {exc.strerror or exc}") from exc


if __name__ == "__main__":
    main()
