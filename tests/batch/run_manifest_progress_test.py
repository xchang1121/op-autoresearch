"""Regression tests for batch manifest progress IO.

Usage:  python tests/batch/run_manifest_progress_test.py
"""
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                    # workspace_autoresearch/
sys.path.insert(0, str(REPO / "scripts" / "batch"))

import manifest as mf                     # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        batch_dir = Path(td)
        progress_path = batch_dir / mf.PROGRESS_FILENAME
        progress_path.write_text(
            '{"cases":{"gelu":{"status":"pending"}}}',
            encoding="utf-8-sig",
        )

        progress = mf.load_progress(batch_dir)
        if progress["cases"]["gelu"]["status"] != "pending":
            print("FAIL: did not load UTF-8 BOM progress file")
            return 1

        mf.save_progress(batch_dir, progress)
        raw = progress_path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            print("FAIL: save_progress wrote a BOM")
            return 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
