"""
temp_manager.py — Per-request isolated temp directory.
Guarantees cleanup on exit even if an exception is raised mid-pipeline.
All Phase 02/04/05 file operations go through this — no raw tempfile calls.
"""
import shutil
import tempfile
from pathlib import Path


class TempFileManager:
    """
    Async context manager. Creates an isolated temp dir per request;
    deletes it unconditionally on __aexit__ (success or exception).

    Usage:
        async with TempFileManager() as tmp:
            path = tmp.write_bytes("input.pdf", pdf_bytes)
            subdir = tmp.make_subdir("figures")
    """

    def __init__(self):
        self._root: Path | None = None

    async def __aenter__(self) -> "TempFileManager":
        self._root = Path(tempfile.mkdtemp(prefix="ragparser_"))
        return self

    async def __aexit__(self, *_) -> None:
        if self._root and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)

    def write_bytes(self, filename: str, data: bytes) -> Path:
        path = self._root / filename
        path.write_bytes(data)
        return path

    def make_subdir(self, name: str) -> Path:
        subdir = self._root / name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    @property
    def root(self) -> Path:
        return self._root