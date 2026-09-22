"""Literal stdin transport for CLIs that accept non-interactive text input."""
from contextlib import contextmanager
import tempfile


@contextmanager
def prompt_input(prompt: str):
    # A seekable anonymous file avoids both exec's per-argument size limit and
    # blocking on a full pipe before the CLI's timeout supervisor starts.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stream:
        stream.write(prompt)
        stream.seek(0)
        yield stream
