"""One-shot SSH probe — gathers structured facts about the remote.

Returns raw facts only. Classification (severity / suggestion) belongs to
``diagnostics.py``; this layer doesn't decide what's fatal vs warning so
the same probe can be reused under different DSL contexts (e.g. triton
missing is fatal for triton_ascend, warn for ascendc_catlass).

Single SSH round-trip; ``stdout=PIPE, stderr=DEVNULL`` avoids the Windows
subprocess+capture_output deadlock that bit OpenSSH / PowerShell."""

from __future__ import annotations

import shlex
import subprocess
from typing import Optional

from .remote_env import source_env_var_bash


# Probe checks are split per concern so each can be classified
# independently. The bash here only emits ``KEY:value`` markers + a
# ``LOG_TAIL_BEGIN`` sentinel; no shell-side severity logic.
#
# arch detection matches the requested probe device when provided and
# falls back to the first visible device otherwise. Name spelling is
# normalized by ``arch_normalize`` rather than a hard-coded SKU table.
# Report check to remove exit code instead of tail output - CANN Initialization may go to stderr
# Writing LOG_WarnING (e.g. log directory permissions), even if Import succeeds in contaminating the last line, it is not possible to write about it.
# Tail-1 is miscalculated Fatal. Capture stdout+stderr to var, see first $?
# Okay, it's still stderr's end.
_PROBE_BASH = r"""
env_script={env_script}
repo_path={repo_path}
probe_device={probe_device}
port={port}
log_file={log_file}
echo "ENV_PATH:$env_script"
echo "ENV_OK:$([ -n "$env_script" ] && [ -f "$env_script" ] && echo yes || echo no)"
{env_setup}
if [ -n "$repo_path" ] && [ -d "$repo_path/src" ]; then
  export PYTHONPATH="$repo_path/src:${{PYTHONPATH:-}}"
fi
TORCH_NPU_OUT=$(python -c 'import torch_npu' 2>&1); TORCH_NPU_RC=$?
if [ $TORCH_NPU_RC -eq 0 ]; then
  echo "TORCH_NPU:ok"
else
  echo "TORCH_NPU:$(echo "$TORCH_NPU_OUT" | tail -1)"
fi
TRITON_OUT=$(python -c 'import triton' 2>&1); TRITON_RC=$?
if [ $TRITON_RC -eq 0 ]; then
  echo "TRITON:ok"
else
  echo "TRITON:$(echo "$TRITON_OUT" | tail -1)"
fi
echo "NPU_SMI:$(command -v npu-smi >/dev/null 2>&1 && echo ok || echo missing)"
echo "PROBE_DEVICE:$probe_device"
ASCEND_CHIP="$(npu-smi info 2>/dev/null | awk -v did="$probe_device" '/^\| +[0-9]+ +[0-9A-Z]/{{if (did == "" || $2 == did) {{print $3; exit}}}}')"
echo "ARCH:$(ARCH_NAME="$ASCEND_CHIP" python -c 'import os; from op_autoresearch.op.utils.arch_normalize import normalize_ascend_arch_name; print(normalize_ascend_arch_name(os.environ.get("ARCH_NAME","")) or "")' 2>/dev/null)"
echo "DEVICES:$(npu-smi info 2>/dev/null | grep -cE '^\| +[0-9]+ +[0-9A-Z]')"
echo "NVIDIA_SMI:$(command -v nvidia-smi >/dev/null 2>&1 && echo ok || echo missing)"
if [ -n "$probe_device" ]; then
  CUDA_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$probe_device" 2>/dev/null | head -1)"
else
  CUDA_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)"
fi
echo "CUDA_NAME:$CUDA_NAME"
echo "CUDA_ARCH:$(CUDA_NAME="$CUDA_NAME" python -c 'import os; from op_autoresearch.op.utils.arch_normalize import normalize_cuda_arch_name; print(normalize_cuda_arch_name(os.environ.get("CUDA_NAME","")) or "")' 2>/dev/null)"
echo "CUDA_DEVICES:$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
echo "CPU_ARCH:$(python -c 'from op_autoresearch.op.utils.arch_normalize import normalize_cpu_arch_name; print(normalize_cpu_arch_name() or "")' 2>/dev/null)"
echo "PORT_PID:$(lsof -ti :$port -sTCP:LISTEN 2>/dev/null | head -1)"
# Space left on remote disk (Land disk)MB)—— /tmp + / The smaller one.daemon When you get up, you'll write a journal.
# Python logging flush Failed → "Logging error" cascade Take it. stderr I don't know what to do.
# Upstream should... fatal Intercept.BusyBox / Alpine of df Not supported --output;go. awk
# Parsing No. 4 ColumnsAvailable KB), multi-target least.
echo "DISK_FREE_MB:$(df -kP /tmp / 2>/dev/null | awk 'NR>1 {{print int($4/1024)}}' | sort -n | head -1)"
echo "LOG_TAIL_BEGIN"
[ -f "$log_file" ] && tail -20 "$log_file" || echo "(no log: $log_file)"
"""


def _first_device_id(device_ids) -> Optional[int]:
    if device_ids is None:
        return None
    if isinstance(device_ids, str):
        parts = [p.strip() for p in device_ids.split(",") if p.strip()]
        return int(parts[0]) if parts else None
    if isinstance(device_ids, (list, tuple, set)):
        return int(next(iter(device_ids))) if device_ids else None
    return int(device_ids)


def probe_remote(ssh_alias: str, env_script: Optional[str], port: int,
                 log_file: Optional[str] = None,
                 repo_path: Optional[str] = None,
                 device_ids=None) -> dict:
    """One SSH round-trip → dict of raw facts.

    Returned keys:
      - ``_SSH_ERROR``: present iff SSH transport itself failed
      - ``ENV_PATH``: configured env_script path (empty if None passed in)
      - ``ENV_OK``: "yes" if path configured AND file exists, "no" otherwise
      - ``TORCH_NPU``: "ok" or Python error string (last traceback line)
      - ``TRITON``: "ok" or Python error string
      - ``NPU_SMI``: "ok" or "missing"
      - ``PROBE_DEVICE``: selected device used for arch inference, or ""
      - ``ARCH``: normalized Ascend arch token or "" if not detected
      - ``DEVICES``: chip count as string (e.g. "8")
      - ``NVIDIA_SMI``: "ok" or "missing"
      - ``CUDA_ARCH``: normalized CUDA arch token (e.g. "a100")
      - ``CUDA_DEVICES``: CUDA device count as string
      - ``CPU_ARCH``: normalized CPU arch token (e.g. "x86_64")
      - ``PORT_PID``: pid string of remote :port LISTEN owner, or ""
      - ``DISK_FREE_MB``: min(/tmp, /) free MB as string; 0 if df failed
      - ``LOG_TAIL``: tail of ``log_file`` or default
        ``/tmp/op_autoresearch_worker_<port>.log``"""
    log = log_file or f"/tmp/op_autoresearch_worker_{port}.log"
    probe_device = _first_device_id(device_ids)
    probe = _PROBE_BASH.format(
        env_script=shlex.quote(env_script or ""),
        repo_path=shlex.quote(repo_path or ""),
        probe_device=shlex.quote("" if probe_device is None else str(probe_device)),
        port=port,
        log_file=shlex.quote(log),
        env_setup=source_env_var_bash("env_script"),
    )
    # Stderr goes PIPE instead of DEVNULL - SSH fax failed (VPN does not open/ network)
    # Invalid / unclassified configuration error / host arias does not exist) error only occurs in stderr, thrown
    # This ssh is run-and-exit
    # (non-f backstage) Won't trigger Windows-f kind of pope-deadlock.
    # ConectTimeout limits 10s to allow SSH to give up early (default 60s+), plus
    # BatchMode=yes disables password prompt (CI / non-interactive set of cards stdin).
    # LogLevel =ERROR: That would include "Connection closed"/ "timed out"
    # The IFO level diagnostics were also crushed, and the probes were in need of them.
    try:
        out = subprocess.run(
            ["ssh",
             "-o", "ConnectTimeout=10",
             "-o", "BatchMode=yes",
             ssh_alias, f"bash -lc {shlex.quote(probe)}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"_SSH_ERROR": "ssh probe 30s timeout (env.sh source/ torch_npu) "
                              "Import may not respond; manual ssh run)"}
    except Exception as e:
        return {"_SSH_ERROR": str(e)[:200]}

    if out.returncode != 0:
        # SSH fax failed: condition / mouth / hostlias not found. stderr usually
        # Writes one or two lines of characters ("Connaction timed out" / "Permission written)
        # (publickey) / "Could not resolve hostname xx")
        # Classify, let the root cause be clear in the first line of the diagnostic form.
        err = (out.stderr or "").strip() or f"ssh exit rc={out.returncode}"
        return {"_SSH_ERROR": err[:200]}

    facts: dict = {}
    log_lines: list = []
    in_log = False
    for line in out.stdout.splitlines():
        if in_log:
            log_lines.append(line)
        elif line == "LOG_TAIL_BEGIN":
            in_log = True
        elif ":" in line:
            k, v = line.split(":", 1)
            facts[k] = v.strip()
    facts["LOG_TAIL"] = "\n".join(log_lines)
    return facts
