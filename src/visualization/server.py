"""Small static HTTP service for the AgentKernelArena dashboard."""

from __future__ import annotations

import argparse
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from src.visualization.paths import (
    DATA_ROOT,
    FRONTEND_ROOT,
    MODULE_ROOT,
    PROJECT_ROOT,
    REPORTS_ROOT,
)


ARTIFACT_EXTENSIONS = {".csv", ".json", ".txt"}
GENERATED_DATA_FILES = {"data.js", "data.json"}


def _resolve_under(root: Path, parts: tuple[str, ...]) -> Path | None:
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*parts).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def resolve_request_path(
    raw_path: str,
    *,
    project_root: Path = PROJECT_ROOT,
    frontend_root: Path = FRONTEND_ROOT,
    data_root: Path = DATA_ROOT,
    reports_root: Path = REPORTS_ROOT,
) -> Path | None:
    """Resolve one dashboard request without exposing arbitrary repository files."""

    path = unquote(urlsplit(raw_path).path)
    if path in ("", "/", "/index.html"):
        return frontend_root / "index.html"

    clean = path.lstrip("/")
    candidate = Path(clean)
    parts = candidate.parts
    if (
        not clean
        or candidate.is_absolute()
        or not parts
        or any(part in (".", "..") or part.startswith(".") for part in parts)
    ):
        return None

    route_name = parts[0]
    route_parts = tuple(parts[1:])
    restricted_artifact = False

    if route_name == "dashboard":
        if len(route_parts) == 1 and route_parts[0] in GENERATED_DATA_FILES:
            route_root = data_root
        else:
            route_root = frontend_root / "dashboard"
    elif route_name == "reports":
        route_root = reports_root
        restricted_artifact = True
    elif route_name == "artifacts":
        route_root = project_root
        restricted_artifact = True
    else:
        return None

    resolved = _resolve_under(route_root, route_parts)
    if resolved is None:
        return None
    if restricted_artifact and resolved.suffix not in ARTIFACT_EXTENSIONS:
        return None
    return resolved


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve only the dashboard assets and allowlisted report artifacts."""

    server_version = "ArenaDashboardHTTP/1.0"

    def translate_path(self, path: str) -> str:
        normalized = resolve_request_path(path)
        return str(normalized if normalized is not None else MODULE_ROOT / "__forbidden__")

    def do_GET(self) -> None:
        if resolve_request_path(self.path) is None:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if resolve_request_path(self.path) is None:
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


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Serve the dashboard until interrupted."""

    try:
        with ThreadingHTTPServer((host, port), DashboardHandler) as httpd:
            print(f"Serving {FRONTEND_ROOT} at http://{host}:{port}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        raise SystemExit("Server stopped.")
    except PermissionError:
        hint = " Ports below 1024 usually require elevated privileges." if port < 1024 else ""
        raise SystemExit(
            f"Permission denied binding to {host}:{port}.{hint}"
        )
    except OSError as exc:
        raise SystemExit(f"Failed to bind to {host}:{port}: {exc.strerror or exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Arena dashboard over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
