"""``op-autoresearch worker --remote-host`` dispatch — thin orchestration layer.

Responsibilities: idempotent ``--start``, ``--stop``, ``--status``,
``--reconnect``. All heavy lifting delegated to siblings:

  - ``tunnel.py``           — local ssh -L lifecycle, port ownership
  - ``remote_probe.py``     — one-shot SSH probe → raw facts
  - ``diagnostics.py``      — facts → ``list[Finding]`` + rich.Table render

Module name was ``worker_remote`` previously — too easy to confuse with
``core/worker/remote_worker.py`` (the HTTP client class). Renamed so the
two layers can't be mistyped into each other."""

# pylint: disable=missing-function-docstring,broad-exception-caught,import-outside-toplevel
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from typing import Optional

from .diagnostics import classify, has_fatal, render_findings
from .remote_env import source_env_script_bash
from .remote_probe import probe_remote
from .tunnel import kill_pid_hint, tunnel_start, tunnel_stop_silent, who_holds_port
from .worker_config import WorkerConfig, WorkerTiming, worker_timing
from op_autoresearch.core.worker.eval_config import eval_defaults


# Back-compat thin wrappers — misc.py / eval_bridge.py still import these
# names. New code should construct ``WorkerConfig.load(...)`` directly.

def load_remote_host_config(alias: str,
                            config_path: Optional[str]) -> Optional[dict]:
    return WorkerConfig.load(config_path).host(alias)


def load_default_port(config_path: Optional[str]) -> Optional[int]:
    return WorkerConfig.load(config_path).port


# ---------------------------------------------------------------------------
# HTTP probes (local-tunnel-side)
# ---------------------------------------------------------------------------


def _curl_status(host: str, port: int,
                 timeout: Optional[float] = None) -> Optional[dict]:
    """``/api/v1/status`` probe. ``timeout`` defaults to ``status_timeout``
    from config —— the ready loop should explicitly pass
    ``ready_probe_timeout`` instead since the two have different roles."""
    import urllib.request
    if timeout is None:
        timeout = worker_timing().status_timeout
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/v1/status", timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _is_ready(st: Optional[dict]) -> bool:
    """True if if ``/status`` returns dict and status field is ready/ok. daemon
    server.py returns ``initializing`` (HTTP passed but works) when it comes to spawn
    It's not loaded) - This state can't skip the spawn, it can't count as pol-loop."""
    if not isinstance(st, dict):
        return False
    return str(st.get("status", "")).lower() in ("ready", "ok")


def _curl_health(host: str, port: int,
                 timeout: Optional[float] = None) -> Optional[dict]:
    """``/health``: Unobstructed discovery.

    ``timeout`` defaults to leave an extra amount of clit excess than the daemon side health_timeout.
    The transfer failed or the old daemon returned None when the endpoint was missing.
    """
    import urllib.request
    if timeout is None:
        timing = worker_timing()
        timeout = (
            max(timing.status_timeout, timing.health_timeout)
            + timing.http_read_margin
        )
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/v1/health", timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Remote spawn helpers
# ---------------------------------------------------------------------------


def _build_remote_start_cmd(host_cfg: dict, backend: str, arch: str,
                            devices: str, port: int,
                            timing: WorkerTiming) -> str:
    """Compose the bash payload sent over SSH to spawn the daemon. The
    recursive ``op-autoresearch`` on remote goes through the local branch of
    ``worker_cmd`` → ``worker_service.start``, which Popen-detaches the
    daemon (``preexec_fn=os.setsid`` + ``stdin=DEVNULL``) so this SSH
    returns promptly.

    ``PYTHONPATH`` is pinned to ``<repo_path>/src`` so the
    daemon runs the checkout source, not whatever pip pinned.

    ``env_script`` may contain plain ``conda activate``; bootstrap the
    conda shell hook before sourcing it so non-interactive SSH behaves like
    the user's login shell.

    worker.* timing Pass. env It's all over, so far away. ``worker_service.start`` not
    Hard-code fixed start-up waiting value —— config.yaml worker.* One change, one change, and one transfer.
    Entry into force."""
    repo_path = host_cfg["repo_path"]
    env_script = host_cfg.get("env_script")

    parts: list = [source_env_script_bash(env_script)]
    parts.append(
        f"export PYTHONPATH={shlex.quote(repo_path)}/src:"
        f"${{PYTHONPATH:-}}"
    )
    # Daemon bound only loopback (tunnel forward: <port> to remote 127.0.0.1).
    parts.append("export WORKER_HOST=127.0.0.1")
    # Recursive remote op-autoresearch skips the start-up form and the heartbeat noise; this machine command is responsible for the user's visible output.
    parts.append("export OP_AUTORESEARCH_CLI_QUIET=1")
    for key, value in timing.as_env().items():
        parts.append(f"export {key}={shlex.quote(value)}")
    for key, value in eval_defaults().as_env().items():
        parts.append(f"export {key}={shlex.quote(value)}")
    parts.append(
        " ".join([
            "python", "-m", "op_autoresearch.cli.cli", "worker",
            "--start",
            "--backend", shlex.quote(backend),
            "--arch", shlex.quote(arch),
            "--devices", shlex.quote(devices),
            "--port", str(port),
        ])
    )
    return "\n".join(parts)


def _build_remote_stop_cmd(host_cfg: dict, port: int) -> str:
    """Compose exact daemon termination plus predecessor-tree cleanup.

    A single SIGTERM is not a completed stop: Uvicorn waits for an in-flight
    eval, while that eval may run for many minutes.  Escalate the one listener
    PID after the configured NPU teardown grace, then invoke the shared,
    PID-fingerprinted eval-group reaper from the same checkout/environment.
    """
    repo_path = host_cfg["repo_path"]
    env_script = host_cfg.get("env_script")
    defaults = eval_defaults()
    polls = max(1, int(defaults.kill_grace_s * 10) + 1)
    registry = f"/tmp/op_autoresearch_worker_{port}_process_groups.json"
    state_lookup = (
        "from op_autoresearch.cli.utils.worker_state import live_worker_pid; "
        f"print(live_worker_pid({port}) or '')"
    )
    cleanup = (
        "import json; "
        "from op_autoresearch.utils.process_utils import "
        "reap_orphaned_process_groups; "
        "from op_autoresearch.cli.utils.worker_state import "
        "load_worker_state, remove_worker_entry, save_worker_state; "
        "reaped = reap_orphaned_process_groups(); "
        "state = load_worker_state(); "
        f"remove_worker_entry(state, {port}); "
        "save_worker_state(state); "
        "print(json.dumps({'reaped_process_groups': reaped}))"
    )
    parts: list[str] = [source_env_script_bash(env_script)]
    parts.append(
        f"export PYTHONPATH={shlex.quote(repo_path)}/src:"
        f"${{PYTHONPATH:-}}"
    )
    for key, value in defaults.as_env().items():
        parts.append(f"export {key}={shlex.quote(value)}")
    parts.append(
        f"export OP_AUTORESEARCH_WORKER_PROCESS_REGISTRY={shlex.quote(registry)}"
    )
    parts.extend([
        f'listener_pid="$(lsof -tiTCP:{port} -sTCP:LISTEN | head -n 1)"',
        f"state_pid=\"$(python -c {shlex.quote(state_lookup)})\"",
        'pid="${listener_pid:-$state_pid}"',
        (
            'if [ -n "$pid" ]; then '
            'kill -TERM "$pid" 2>/dev/null || true; '
            f'for _ in $(seq 1 {polls}); do '
            'kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done; '
            'if kill -0 "$pid" 2>/dev/null; then '
            'kill -KILL "$pid" 2>/dev/null || true; fi; '
            'fi'
        ),
        f"python -c {shlex.quote(cleanup)}",
    ])
    return "\n".join(parts)


def _ssh_dispatch(ssh_alias: str, bash_cmd: str) -> int:
    """SSH-run rash_cmd on arias, stdout pass through the terminal (op-autoresearch)
    Recursive print flow back. ``-o LogLevel=ERROR`` inhibit SSH banner is irrelevant
    RemoteForward warning, keep the real ssh error."""
    return subprocess.call([
        "ssh", "-o", "LogLevel=ERROR",
        ssh_alias, f"bash -lc {shlex.quote(bash_cmd)}",
    ])


def _step(msg: str) -> None:
    """Step log to stderr with ``flush=True`` — Windows terminals
    occasionally buffer stderr until the producer terminates, which makes
    a 30s probe look "frozen". Flushing per line eliminates that."""
    print(f"[op-autoresearch] {msg}", file=sys.stderr, flush=True)


def _device_ids_from_arg(devices: Optional[str]) -> Optional[list[int]]:
    if devices is None:
        return None
    ids = [int(p.strip()) for p in str(devices).split(",") if p.strip()]
    return ids or None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch_start(alias: str, host_cfg: dict, backend: Optional[str],
                   arch: Optional[str], devices: Optional[str],
                   port: int, dsl: Optional[str] = None) -> int:
    """SSH-dispatch worker --start with idempotent recovery.

    Flow: probe /status → rebuild tunnel + reprobe → run diagnostic probe
    (also fills missing CLI defaults) → spawn daemon → poll /status
    with heartbeat. Any fatal finding aborts before spawn.

    ``backend / arch / devices`` may be None. Arch is filled from the
    backend-specific remote probe (Ascend: npu-smi, CUDA: nvidia-smi,
    CPU: platform.machine) unless the caller passes ``--arch``. ``dsl`` drives
    ``classify(require_triton=...)``: ``triton_*`` makes missing triton
    fatal, others (catlass / pypto / ...) keep it warn. Pass None to let
    dispatch read ``defaults.dsl`` from config.yaml."""
    if "repo_path" not in host_cfg:
        _step(f"remote_worker.hosts.{alias} Missing repo_path")
        return 2
    ssh_alias = host_cfg.get("ssh_alias") or alias
    log_file = f"/tmp/op_autoresearch_worker_{port}.log"
    env_script = host_cfg.get("env_script")
    repo_path = host_cfg.get("repo_path")

    # Resolve all defaults up front so every classify call sees a
    # consistent effective_backend / dsl / timing — without this,
    # tunnel-fail diagnostics used user-passed `backend` (often None)
    # and read different defaults than the probe-success path.
    cfg = WorkerConfig.load(None)
    effective_backend = (backend or cfg.backend)   # CLI > config.yaml
    effective_dsl = dsl or cfg.dsl
    timing = worker_timing()
    probe_device_ids = _device_ids_from_arg(devices)

    _step(f"[1/4] I'm a detective. 127.0.0.1:{port}/api/v1/status ...")
    st = _curl_status("127.0.0.1", port, timeout=timing.status_timeout)
    if _is_ready(st):
        _step(f"[1/4] daemon Ready — nothing to do")
        print(json.dumps(st, indent=2, ensure_ascii=False))
        return 0
    if st is not None:
        _step(f"[1/4] /status Back {st.get('status')!r}No, it's not. ready —— Keep going down.")
    else:
        _step(f"[1/4] It's not working. → tunnel or daemon At least one of them isn't.")

    _step(f"[2/4] Rebuild your machine. ssh -L :{port} → {ssh_alias} ...")
    tunnel_stop_silent(port, ssh_alias)
    pid = tunnel_start(ssh_alias, port)
    if pid == 0:
        # tunnel ssh-f stderr - run once_remote reuses the same
        # A lias inverted SSH transmissible: VPN not active / network not available / decrypt configuration error /
        # _SSH_ERROR Path does not exist in the first line of the diagnostic form.
        _step(f"[2/4] tunnel Failed to raise —— Reverse diagnosis SSH TRANSFER:")
        facts = probe_remote(ssh_alias, env_script, port, log_file, repo_path,
                             probe_device_ids)
        render_findings(
            classify(facts, port, backend=effective_backend,
                     dsl=effective_dsl, for_start=True),
            facts.get("LOG_TAIL", ""),
        )
        return 1
    _step(f"[2/4] tunnel pid={pid}, I'll be back in a few minutes. /status ...")
    st = _curl_status("127.0.0.1", port, timeout=timing.status_timeout)
    if _is_ready(st):
        _step(f"[2/4] Yes. tunnel The line that breaks.daemon Still there. — Completed")
        print(json.dumps(st, indent=2, ensure_ascii=False))
        return 0
    _step(f"[2/4] tunnel It's all over, but... /status Not Ready → daemon Not running or still active init")

    _step(f"[3/4] Long-range diagnostics (Presentation)env / backend deps / triton / disk / port / log)...")
    facts = probe_remote(ssh_alias, env_script, port, log_file, repo_path,
                         probe_device_ids)
    # Final fallback: program-based only when CLI/env + config.yaml has not given
    # Infer (torch_npu can import aspend, otherwise cuda).
    if effective_backend is None:
        effective_backend = "ascend" if facts.get("TORCH_NPU") == "ok" else "cuda"
    findings = classify(facts, port, backend=effective_backend,
                        dsl=effective_dsl, for_start=True)
    if has_fatal(findings):
        _step(f"[3/4] fatal entry, do not start daemonDiagnosis:")
        render_findings(findings, facts.get("LOG_TAIL", ""))
        return 1

    backend = effective_backend
    if arch is None:
        if backend == "ascend":
            raw_arch = (facts.get("ARCH") or "").strip().lower()
            if not raw_arch:
                _step(f"[3/4] No automatic extrapolation ascend arch,--arch It has to be visible.")
                render_findings(findings, facts.get("LOG_TAIL", ""))
                return 1
            arch = raw_arch
        elif backend == "cuda":
            raw_arch = (facts.get("CUDA_ARCH") or "").strip().lower()
            if not raw_arch:
                _step(f"[3/4] No automatic extrapolation cuda arch,--arch It has to be visible.")
                render_findings(findings, facts.get("LOG_TAIL", ""))
                return 1
            arch = raw_arch
        elif backend == "cpu":
            raw_arch = (facts.get("CPU_ARCH") or "").strip().lower()
            if not raw_arch:
                _step(f"[3/4] No automatic extrapolation cpu arch,--arch It has to be visible.")
                render_findings(findings, facts.get("LOG_TAIL", ""))
                return 1
            arch = raw_arch
        else:
            _step(f"[3/4] backend={backend!r} No remotes. arch It's an automatic inference.--arch It has to be visible.")
            render_findings(findings, facts.get("LOG_TAIL", ""))
            return 1
    if devices is None:
        devices = "0"
    _step(f"[3/4] The probe. OK: backend={backend}, arch={arch}, "
          f"devices={devices}, dsl={effective_dsl or '(any)'}")

    _step(f"[4/4] SSH Up the far end. daemon at {ssh_alias}:{port} ...")
    remote_cmd = _build_remote_start_cmd(
        host_cfg, backend=backend, arch=arch, devices=devices, port=port,
        timing=timing,
    )
    rc = _ssh_dispatch(ssh_alias, remote_cmd)
    if rc != 0:
        _step(f"[4/4] remote daemon launch rc={rc} —— Rediagnosing:")
        facts2 = probe_remote(ssh_alias, env_script, port, log_file, repo_path,
                              probe_device_ids)
        render_findings(
            classify(facts2, port, backend=backend, dsl=effective_dsl,
                     for_start=True),
            facts2.get("LOG_TAIL", ""),
        )
        return rc

    _step(f"[4/4] daemon spawned,poll /status ready(Maximum {timing.ready_timeout}s)...")
    deadline = time.time() + timing.ready_timeout
    last_beat = time.time()
    while time.time() < deadline:
        # Ready phase with ready_probe_timeout (round semantics), different from idle
        # --status's status_timeout (one-time query). _is_ready accepted only
        # Ready/ok;initializing does not count (daemon start-up period HTTP passed but working
        # It's not ready yet. Keep waiting.
        st = _curl_status("127.0.0.1", port,
                          timeout=timing.ready_probe_timeout)
        if _is_ready(st):
            _step(f"[4/4] /status ready — Completed")
            print(json.dumps(st, indent=2, ensure_ascii=False))
            return 0
        now = time.time()
        if now - last_beat >= timing.ready_poll_interval:
            _step(f"   /status Not Ready "
                  f"({int(now - deadline + timing.ready_timeout)}s"
                  f"/{timing.ready_timeout}s)...")
            last_beat = now
        time.sleep(1)

    _step(f"[4/4] /status {timing.ready_timeout}s Not Ready —— Rediagnosing:")
    facts2 = probe_remote(ssh_alias, env_script, port, log_file, repo_path,
                          probe_device_ids)
    render_findings(
        classify(facts2, port, backend=backend, dsl=effective_dsl,
                 for_start=True),
        facts2.get("LOG_TAIL", ""),
    )
    return 1


def dispatch_stop(alias: str, host_cfg: dict, port: int) -> int:
    """Tear down the tunnel, stop the exact listener, reap its eval trees."""
    ssh_alias = host_cfg.get("ssh_alias") or alias
    tunnel_stop_silent(port, ssh_alias)
    print(f"[op-autoresearch] tore down local tunnel for :{port}")
    if "repo_path" not in host_cfg:
        print(f"[op-autoresearch] remote_worker.hosts.{alias} Missing repo_path",
              file=sys.stderr)
        return 2
    rc = _ssh_dispatch(ssh_alias, _build_remote_stop_cmd(host_cfg, port))
    if rc != 0:
        print(f"[op-autoresearch] remote daemon stop rc={rc}", file=sys.stderr)
        return rc
    print(f"[op-autoresearch] stopped remote daemon and reaped owned eval trees on "
          f"{ssh_alias}:{port}")
    return 0


def dispatch_status(alias: str, host_cfg: dict, port: int, *,
                    backend: Optional[str] = None,
                    dsl: Optional[str] = None) -> int:
    """Curl tunneled ``/status`` + ``/health``.

    On /status failure, identify the local port holder and, for remote
    aliases, run the same SSH/env/NPU probe used by start preflight.
    /health output surfaces ``free`` + ``note`` so 'healthy busy' and
    'healthy idle' are distinguishable."""
    st = _curl_status("127.0.0.1", port)
    if st is None:
        holder = who_holds_port(port)
        if holder is None:
            print(f"Worker 127.0.0.1:{port} Unattainable; live :port Free → Run! `--start`.")
        else:
            print(
                f"Worker 127.0.0.1:{port} Unattainable;:port By PID={holder['pid']} Hold on.\n"
                f"  cmdline: {holder['cmdline'][:120]}\n"
                f"  → Residues tunnel:`{kill_pid_hint(holder['pid'])}` after --start;"
                f"Far daemon Stopped:--stop + --start"
            )
        ssh_alias = host_cfg.get("ssh_alias") or alias
        if ssh_alias != "local":
            cfg = WorkerConfig.load(None)
            effective_backend = backend or cfg.backend
            effective_dsl = dsl or cfg.dsl
            log_file = f"/tmp/op_autoresearch_worker_{port}.log"
            env_script = host_cfg.get("env_script")
            repo_path = host_cfg.get("repo_path")
            facts = probe_remote(ssh_alias, env_script, port, log_file, repo_path)
            render_findings(
                classify(
                    facts, port, backend=effective_backend,
                    dsl=effective_dsl, for_start=False,
                ),
                facts.get("LOG_TAIL", ""),
            )
        return 1

    health = _curl_health("127.0.0.1", port)
    out = dict(st)
    if health is not None:
        out["health"] = {
            "healthy": bool(health.get("healthy")),
            "probed_device": health.get("probed_device"),
            "free": health.get("free"),
            "note": health.get("note"),
            "error": health.get("error"),
        }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if health is not None and not health.get("healthy"):
        print(
            f"\n[op-autoresearch] /status OK but /health Report! degraded —— "
            f"daemon handler Possible blocking. Error:{health.get('error')!r}",
            file=sys.stderr,
        )
        return 1
    return 0


def dispatch_reconnect_tunnel(alias: str, host_cfg: dict, port: int) -> int:
    """Rebuild only the local tunnel; leave remote daemon alone. Use when
    a long batch silently lost its tunnel (server-side SSH reset / network
    drop) but the daemon is still alive. Falls back to --stop+--start if
    the daemon is also gone."""
    ssh_alias = host_cfg.get("ssh_alias") or alias
    tunnel_stop_silent(port, ssh_alias)
    pid = tunnel_start(ssh_alias, port)
    if pid:
        print(f"[op-autoresearch] ssh -L 127.0.0.1:{port} → "
              f"{ssh_alias}:{port} reconnected (tunnel pid={pid})")
    st = _curl_status("127.0.0.1", port)
    if st is None:
        print(
            f"[op-autoresearch] /status It's still not working.daemon Maybe it stopped. — Use it. --stop + --start.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(st, indent=2, ensure_ascii=False))
    return 0
