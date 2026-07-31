"""Standalone operator autoresearch runtime."""

from pathlib import Path

DEFAULT_LOG_DIR = str(Path.home() / ".op_autoresearch" / "logs")


def get_project_root() -> str:
    """Return the installed package root used to resolve bundled resources."""
    return str(Path(__file__).resolve().parent)


__all__ = ["DEFAULT_LOG_DIR", "get_project_root"]

