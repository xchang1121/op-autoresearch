from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class EvalDefaults:
    """The centralised default configuration for verify/profile/generate-reference."""

    eval_timeout: int = 600              # single budget (sec)
    reference_timeout: int = 120         # budget (s)
    warmup_times: int = 5                # profile preheats
    run_times: int = 50                  # Number of official measurements

    # Seconds to exit after SIGTERM; NPU > = 2s to PyTorch+CANN
    # Atexit release ACL contact, otherwise SIGKILL leaves TS residual on device
    # 6/17 device 5 wedge and path. Seconds of SIGKILL rear drain pipe.
    kill_grace_s: float = 5.0
    kill_drain_s: float = 2.0

    def as_env(self) -> Dict[str, str]:
        """Converts to the environmental variable of detached / remote worker daemon consumption."""
        return {
            "OP_AUTORESEARCH_EVAL_TIMEOUT_S": str(self.eval_timeout),
            "OP_AUTORESEARCH_EVAL_REFERENCE_TIMEOUT_S": str(self.reference_timeout),
            "OP_AUTORESEARCH_EVAL_WARMUP_TIMES": str(self.warmup_times),
            "OP_AUTORESEARCH_EVAL_RUN_TIMES": str(self.run_times),
            "OP_AUTORESEARCH_EVAL_KILL_GRACE_S": str(self.kill_grace_s),
            "OP_AUTORESEARCH_EVAL_KILL_DRAIN_S": str(self.kill_drain_s),
        }


def eval_defaults(config_path: Optional[str] = None) -> EvalDefaults:
    """Parsing the eval/profile default that is finally valid.

    Priority: OP_AUTORESEARCH_EVAL_* env > config.yaml > EvalDefaults datacas default.
    config.yaml Read ``defaults.eval_timeout``,
    ``defaults.reference_data_timeout``,``eval.warmup``,``eval.repeats``,
    ``defaults.kill_grace_s``,``defaults.kill_drain_s``.
    """
    base = _from_config(config_path)
    return EvalDefaults(
        eval_timeout=_env_int("OP_AUTORESEARCH_EVAL_TIMEOUT_S", base.eval_timeout),
        reference_timeout=_env_int(
            "OP_AUTORESEARCH_EVAL_REFERENCE_TIMEOUT_S", base.reference_timeout),
        warmup_times=_env_int("OP_AUTORESEARCH_EVAL_WARMUP_TIMES", base.warmup_times),
        run_times=_env_int("OP_AUTORESEARCH_EVAL_RUN_TIMES", base.run_times),
        kill_grace_s=_env_float("OP_AUTORESEARCH_EVAL_KILL_GRACE_S", base.kill_grace_s),
        kill_drain_s=_env_float("OP_AUTORESEARCH_EVAL_KILL_DRAIN_S", base.kill_drain_s),
    )


def resolve_eval_timeout(value: Optional[int] = None) -> int:
    return _positive_int(value, eval_defaults().eval_timeout)


def resolve_reference_timeout(value: Optional[int] = None) -> int:
    return _positive_int(value, eval_defaults().reference_timeout)


def resolve_warmup_times(value: Optional[int] = None) -> int:
    return _positive_int(value, eval_defaults().warmup_times)


def resolve_run_times(value: Optional[int] = None) -> int:
    return _positive_int(value, eval_defaults().run_times)


def resolve_kill_grace_s(value: Optional[float] = None) -> float:
    """Sets the value of 0 at the direct SIGKILL (test)."""
    return _positive_float(value, eval_defaults().kill_grace_s)


def resolve_kill_drain_s(value: Optional[float] = None) -> float:
    """The number of seconds to wait for the SIGKIL back-dry stdout/stderr pipe."""
    return _positive_float(value, eval_defaults().kill_drain_s)


def _from_config(config_path: Optional[str]) -> EvalDefaults:
    defaults = EvalDefaults()
    resolved = _resolve(config_path)
    if resolved is None:
        return defaults
    data = _load_yaml(resolved)
    if not isinstance(data, dict):
        return defaults
    defaults_block = data.get("defaults") or {}
    eval_block = data.get("eval") or {}
    return EvalDefaults(
        eval_timeout=_positive_int(
            defaults_block.get("eval_timeout"), defaults.eval_timeout),
        reference_timeout=_positive_int(
            defaults_block.get("reference_data_timeout"),
            defaults.reference_timeout,
        ),
        warmup_times=_positive_int(eval_block.get("warmup"),
                                   defaults.warmup_times),
        run_times=_positive_int(eval_block.get("repeats"),
                                defaults.run_times),
        kill_grace_s=_positive_float(
            defaults_block.get("kill_grace_s"), defaults.kill_grace_s),
        kill_drain_s=_positive_float(
            defaults_block.get("kill_drain_s"), defaults.kill_drain_s),
    )


# config.yaml parses + reads the only reality. worker_config.py returns these two helpers,
# Each expresses the difference only by its parameters: eval side walk parents up to config.yaml, worker
# Look at cwd (walk_parents=False); the failed stderr tag is also used separately.
def _resolve(config_path: Optional[str], *, walk_parents: bool = True) -> Optional[str]:
    if config_path is not None:
        p = Path(config_path)
        return str(p) if p.is_file() else None
    cur = Path.cwd()
    candidates = (cur, *cur.parents) if walk_parents else (cur,)
    for candidate in candidates:
        p = candidate / "config.yaml"
        if p.is_file():
            return str(p)
    return None


def _load_yaml(config_path: str, tag: str = "eval_bridge") -> Optional[dict]:
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[{tag}] failed to read {config_path}: {e}", file=sys.stderr)
        return None


def _env_int(key: str, default: int) -> int:
    return _positive_int(os.environ.get(key), default)


def _env_float(key: str, default: float) -> float:
    return _positive_float(os.environ.get(key), default)


def _positive_int(value, default: int) -> int:
    try:
        v = int(value)
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _positive_float(value, default: float) -> float:
    """Allows 0 (kill_grace_s=0 → skips Graceful), negative number and unresolved fallback."""
    try:
        v = float(value)
        return v if v >= 0 else default
    except (TypeError, ValueError):
        return default
