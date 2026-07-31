from pathlib import Path

from op_autoresearch import DEFAULT_LOG_DIR


def get_log_dir() -> Path:
    path = Path(DEFAULT_LOG_DIR).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_process_log_dir() -> Path:
    path = get_log_dir() / "processes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_package_dir() -> Path:
    import op_autoresearch
    return Path(op_autoresearch.__file__).resolve().parent

