"""Private, invocation-specific event capture before display formatting."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile


_SECRET_ENV = re.compile(r"TOKEN|API_KEY|SECRET|PASSWORD|AUTHORIZATION", re.I)
_SECRET_FIELDS = frozenset({"accesstoken", "refreshtoken", "apikey", "authorization",
                            "password", "clientsecret", "secret", "token", "cookie"})
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.I)


class StreamLog:
    """Keep full event content in private files even when the caller raises.

    Each reader owns its file and flushes each line. The transcript is diagnostic
    data, never evaluation evidence. Redaction covers known environment secrets
    and credential fields; it does not make arbitrary tool output publishable.
    """

    def __init__(self, workspace, label, logger, *, env=None):
        if not re.fullmatch(r"[a-z0-9_-]+", label):
            raise ValueError("Invalid stream log label")
        workspace = Path(workspace).resolve(strict=True)
        self.directory = Path(tempfile.mkdtemp(
            prefix=f".{workspace.name}-{label}-stream-", dir=workspace.parent))
        environment = os.environ if env is None else env
        self.secrets = tuple(sorted({value for key, value in environment.items()
                                    if _SECRET_ENV.search(key) and isinstance(value, str)
                                    and len(value) >= 6}, key=len, reverse=True))
        self.error = None
        logger.info("Private agent event logs: %s", self.directory)

    def redact(self, line):
        def text(value):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return _BEARER.sub("Bearer [REDACTED]", value)

        def fields(value):
            if isinstance(value, dict):
                return {key: "[REDACTED]" if re.sub(r"[^a-z]", "", key.lower()) in _SECRET_FIELDS
                        else fields(item) for key, item in value.items()}
            if isinstance(value, list):
                return [fields(item) for item in value]
            return text(value) if isinstance(value, str) else value

        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            return text(line)
        return json.dumps(fields(event), ensure_ascii=False) + "\n"

    @contextmanager
    def open(self, stream):
        if stream not in {"stdout", "stderr"}:
            raise ValueError("Invalid event stream")
        try:
            descriptor = os.open(self.directory / (stream + ".jsonl"),
                                 os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", buffering=1) as handle:
                def record(line):
                    sanitized = self.redact(line)
                    handle.write(sanitized)
                    handle.flush()
                    return sanitized
                yield record
        except Exception as exc:
            self.error = f"Agent event recording failed: {type(exc).__name__}"
            raise
