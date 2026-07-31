"""Single-source-of-truth config loader for ``op-autoresearch worker``.

Replaces four sibling helpers (``load_default_port`` / ``load_default_dsl``
/ ``load_default_backend`` / ``_worker_setting``) that each re-resolved
and re-read the same yaml. One ``WorkerConfig.load(path)`` call returns a
frozen dataclass; everything downstream reads from it instead of poking
yaml again."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from op_autoresearch.op.utils.hw_detect import derive_arch
# The only result of config.yaml resolution/reading lives on eval_config (core level); cli reuses it,
# Only turn the walk_parents/tag into the workingr side semantic. core←cli is the right dependent direction.
from op_autoresearch.core.worker.eval_config import _resolve, _load_yaml


# 11 knobs for working: field name - > env var name. Only table, as_env()/
# Walk it all, load() read yaml directly by field name.
_TIMING_ENV = {
    "ready_timeout": "OP_AUTORESEARCH_WORKER_READY_TIMEOUT",
    "ready_poll_interval": "OP_AUTORESEARCH_WORKER_READY_POLL_INTERVAL",
    "ready_probe_timeout": "OP_AUTORESEARCH_WORKER_READY_PROBE_TIMEOUT",
    "status_timeout": "OP_AUTORESEARCH_WORKER_STATUS_TIMEOUT",
    "lease_ttl": "OP_AUTORESEARCH_WORKER_LEASE_TTL_S",
    "lease_reap_interval": "OP_AUTORESEARCH_WORKER_LEASE_REAP_INTERVAL_S",
    "acquire_timeout": "OP_AUTORESEARCH_WORKER_ACQUIRE_TIMEOUT_S",
    "http_read_margin": "OP_AUTORESEARCH_WORKER_HTTP_READ_MARGIN_S",
    "release_timeout": "OP_AUTORESEARCH_WORKER_RELEASE_TIMEOUT_S",
    "doc_timeout": "OP_AUTORESEARCH_WORKER_DOC_TIMEOUT_S",
    "health_timeout": "OP_AUTORESEARCH_WORKER_HEALTH_TIMEOUT_S",
}


@dataclass(frozen=True)
class WorkerTiming:
    """``worker.*`` timing knobs from config.yaml. All in seconds (float)."""
    ready_timeout: float = 60.0          # How long does it take for daemon/status to be ready?
    ready_poll_interval: float = 5.0     # Heart beat tick interval
    ready_probe_timeout: float = 3.0     # Every /status probe single timeout
    status_timeout: float = 3.0          # idle --status Discovery single timeout
    lease_ttl: float = 120.0             # How long after request closes/clients are missing
    lease_reap_interval: float = 30.0    # daemon scanned timeout space
    acquire_timeout: float = 600.0       # /acquire_device for how long to spare device
    http_read_margin: float = 10.0       # Clint read timeout extra
    release_timeout: float = 10.0        # /release_device Read Timeout
    doc_timeout: float = 20.0            # /docs/<name> Read timeout
    health_timeout: float = 5.0          # daemon / health cycle detection timed out

    def as_env(self) -> Dict[str, str]:
        """Converts to the environmental variable of detached worker daemon consumption.

        Cwd does not necessarily have config.yaml;op-autoresearch first parsed at remote worker startup
        tasker.*, and via env to daemon to avoid daemon side growing a default set.
        """
        return {env: str(getattr(self, field)) for field, env in _TIMING_ENV.items()}


@dataclass(frozen=True)
class WorkerConfig:
    """Top-level worker config — read once, passed around as a value object.

    All non-DSL fields have concrete defaults baked in(``port=9001``,
    ``backend="cuda"``, wait. ``WorkerConfig`` is the single source of
    truth: Callers read ``cfg.port`` / ``cfg.backend`` no longer everywhere `or
    \"cuda\"`` / ``else9001`bar. Overwrite is allowed up only (CLI > env > yaml),
    Fallback will always be this one.

    ``dsl`` retains Optional - ``None`` semantic ( \"unspecified DSL\", trigger classify
    The warn instead of the fatal) cannot be replaced by \"\"\""""
    port: int = 9001
    backend: str = "cuda"
    arch: str = "a100"
    devices: str = "0"
    dsl: Optional[str] = None
    hosts: Dict[str, dict] = field(default_factory=dict)
    timing: WorkerTiming = field(default_factory=WorkerTiming)
    source_path: Optional[str] = None    # Parsing the absolute path of yaml, diag

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "WorkerConfig":
        """Road from exclusive path or default ``cwd/config.yaml``. yaml is missing
        Missing or missing → with default dataclass. Never return None, Callers
        null-check."""
        resolved = _resolve(config_path, walk_parents=False)
        if resolved is None:
            return cls()
        data = _load_yaml(resolved, tag="op-autoresearch")
        if data is None:
            return cls(source_path=resolved)

        worker = data.get("worker") or {}
        defaults = data.get("defaults") or {}
        hosts = ((data.get("remote_worker") or {}).get("hosts") or {})

        port_v = _int_in_range(worker.get("port"), 1, 65535, cls.port)
        backend_v = _str_or(defaults.get("backend"), cls.backend).lower()
        arch_v = _str_or(defaults.get("arch"), cls.arch)
        devices_v = _str_or(defaults.get("devices"), cls.devices)
        dsl_raw = defaults.get("dsl")
        dsl_v: Optional[str] = str(dsl_raw) if isinstance(dsl_raw, str) else None

        # Timing Defaults are taken only from WorkerTiming, avoiding two dataclass duplicates of numbers.
        td = WorkerTiming()
        timing = WorkerTiming(**{
            field: _float(worker.get(field), getattr(td, field))
            for field in _TIMING_ENV
        })
        return cls(
            port=port_v, backend=backend_v, arch=arch_v, devices=devices_v,
            dsl=dsl_v, hosts=dict(hosts), timing=timing, source_path=resolved,
        )

    def host(self, alias: str) -> Optional[dict]:
        """Look up ``remote_worker.hosts.<alias>``. None if absent."""
        return self.hosts.get(alias)


def _float(val, default: float) -> float:
    if isinstance(val, (int, float)) and val > 0:
        return float(val)
    return default


def _int_in_range(val, lo: int, hi: int, default: int) -> int:
    if isinstance(val, int) and lo <= val <= hi:
        return val
    return default


def _str_or(val, default: str) -> str:
    return str(val).strip() if isinstance(val, str) and str(val).strip() else default


def _env_float(key: str, default: float) -> float:
    try:
        v = float(os.environ.get(key, ""))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def worker_timing(config_path: Optional[str] = None) -> WorkerTiming:
    """Parsing the working timing configuration that is finally valid.

    Priority: OP_AUTORESEARCH_WORKER_* env (detached/remote daemon path)
    config.yaml worker.* > WorkerTiming dataclass default.
    """
    cfg = WorkerConfig.load(config_path).timing
    return WorkerTiming(**{
        field: _env_float(env, getattr(cfg, field))
        for field, env in _TIMING_ENV.items()
    })


def probe_local_arch(backend: str, device_id: int = 0) -> Optional[str]:
    """Best-effort local arch probe so ``op-autoresearch worker --start`` (no
    --remote-host) doesn't fall back to the baked ``a100`` default on hosts
    the operator forgot to flag. Delegates to the shared
    :func:`op_autoresearch.op.utils.hw_detect.derive_arch` — one probe
    implementation for both the CLI worker and the workspace scaffold.

    Returns None on any failure (binary not on PATH, non-zero exit,
    unparseable output, unknown backend); caller falls through to ``cfg.arch``.
    """
    return derive_arch(device_id, backend)
