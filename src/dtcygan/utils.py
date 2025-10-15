"""Utility helpers for CLI entrypoints."""

from __future__ import annotations

from pathlib import Path


def prepare_output_path(path: str | Path) -> Path:
    """Return a Path ready for writing, removing any existing file."""
    output_path = Path(path)
    if output_path.exists():
        if output_path.is_file():
            output_path.unlink()
        else:
            raise IsADirectoryError(f"Cannot overwrite directory: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
