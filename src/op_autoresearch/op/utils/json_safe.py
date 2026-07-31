"""JSON-safety helper for eval-result payloads.

Recursively replaces non-finite floats (``inf`` / ``-inf`` / ``nan``)
with ``None``. Used at every serialization boundary that crosses out
of Python's in-memory world into a wire format with strict JSON
semantics:

  - FastAPI HTTP responses (``json.dumps(allow_nan=False)`` by default)
  - On-disk JSON artifacts intended to be read by non-Python tooling
  - ``history.jsonl`` / ``progress.json`` in workspace_autoresearch
    (external dashboards may parse them strictly)

Producer code (profilers, roofline arithmetic) is free to return
``float('inf')`` / NaN as sentinel values — this wrapper at the
boundary maps them to ``null`` so the network / disk format stays
clean and downstream consumers see a single missing-data convention.
"""
from __future__ import annotations

import math
from typing import Any


def sanitize_floats(obj: Any) -> Any:
    """Recursively replace ``inf`` / ``-inf`` / ``nan`` with ``None``.

    Walks dicts, lists, and tuples; leaves other types untouched.
    Tuples are returned as tuples to preserve the input shape — callers
    that need list-of-list for JSON should use ``list(...)`` themselves.
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_floats(v) for v in obj)
    return obj
