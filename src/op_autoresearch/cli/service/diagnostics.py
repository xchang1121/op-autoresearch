"""Facts → ``list[Finding]`` classifier + rich.Table renderer.

Pure mapping. Takes raw facts produced by ``remote_probe.probe_remote``
and emits structured findings with severity + remediation suggestion.
The split lets the same probe feed different severity policies — e.g.
triton_ascend DSL flags missing triton as fatal, ascendc_catlass flags
the same fact as warn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Finding:
    severity: str  # "ok" / "info" / "warn" / "fatal"
    check: str
    result: str
    suggest: str


def _ssh_suggest(err: str) -> str:
    """Classify common SSH errors and return a focused remediation hint."""
    low = err.lower()
    if "could not resolve hostname" in low or "name or service not known" in low:
        return "The alias is missing from ~/.ssh/config; check the alias and Host entry."
    if "timed out" in low or "no route to host" in low or "network is unreachable" in low:
        return "Check the VPN, network route, and whether the remote host is online."
    if "connection refused" in low:
        return "The remote sshd is unavailable or the firewall is blocking port 22."
    if "connection closed" in low or "connection reset" in low:
        return "The server closed the connection; inspect remote /var/log/auth.log for authentication, MaxSessions, or firewall errors."
    if "permission denied" in low or "publickey" in low:
        return "Public-key authentication failed; run ssh-copy-id or check IdentityFile and authorized_keys."
    if "host key verification failed" in low:
        return "The host key changed; verify the host, then run ssh-keygen -R <host>."
    return "Check the SSH alias, network, and authentication by running `ssh <alias>` manually."


def classify(facts: dict, port: int, *,
             backend: Optional[str] = None,
             dsl: Optional[str] = None,
             for_start: bool = False) -> List[Finding]:
    """Map probe facts to findings.

    Context flags (all optional; defaults to the conservative "ascend +
    unknown DSL + not-starting" policy):

    - ``backend``: ``"ascend"`` / ``"cuda"`` / ``"cpu"`` / None. When set
      to ``"cuda"`` or ``"cpu"`` we don't flag missing torch_npu / npu-smi
      as fatal — those would just be noise on a non-Ascend worker. None
      treated as ascend (the project's default).
    - ``dsl``: e.g. ``"triton_ascend"``, ``"ascendc_catlass"``, ``"pypto"``.
      Only ``triton_*`` DSLs make missing triton fatal; everyone else
      flags as warn.
    - ``for_start``: True when called from ``dispatch_start`` (i.e. about
      to spawn). Remote :port held becomes fatal then — bind will fail
      anyway. False (default) keeps it warn for diagnostic-only callers."""
    findings: List[Finding] = []

    ssh_err = facts.get("_SSH_ERROR")
    if ssh_err:
        return [Finding("fatal", "ssh", ssh_err[:160], _ssh_suggest(ssh_err))]

    # Resolve policy from context.
    ascendish = backend in (None, "", "ascend")
    cudaish = backend == "cuda"
    cpuish = backend == "cpu"
    needs_triton = (dsl or "").startswith("triton")

    # env_script: distinguish "not configured" (use remote default shell
    # env) from "configured but path missing" (config bug, fatal).
    env_path = facts.get("ENV_PATH") or ""
    env_ok = facts.get("ENV_OK") or "no"
    if not env_path:
        findings.append(Finding(
            "info", "env_script", "Unconfigured",
            "If the remote default shell does not load CANN and torch_npu, set "
            "config.yaml: remote_worker.hosts.<alias>.env_script.",
        ))
    elif env_ok == "yes":
        findings.append(Finding("ok", "env_script", env_path, ""))
    else:
        findings.append(Finding(
            "fatal", "env_script", f"Configured path does not exist: {env_path}",
            "Check config.yaml: remote_worker.hosts.<alias>.env_script.",
        ))

    # torch_npu: ascend backend hard dependency; non-ascend → warn at most.
    torch_result = facts.get("TORCH_NPU") or ""
    if torch_result == "ok":
        findings.append(Finding("ok", "torch_npu", "importable", ""))
    elif ascendish:
        findings.append(Finding(
            "fatal", "torch_npu", torch_result[:120] or "Import failed",
            "Ensure env_script sources CANN set_env.sh and installs torch_npu.",
        ))
    else:
        findings.append(Finding(
            "info", "torch_npu", "not importable", f"Not required for backend={backend}.",
        ))

    # triton: fatal only when target DSL needs it (triton_*).
    triton_result = facts.get("TRITON") or ""
    if triton_result == "ok":
        findings.append(Finding("ok", "triton", "importable", ""))
    else:
        sev = "fatal" if needs_triton else "warn"
        suggest = ("The selected Triton DSL requires the triton package." if needs_triton
                   else "Only Triton DSLs require this package.")
        findings.append(Finding(
            sev, "triton", triton_result[:80] or "Import failed", suggest,
        ))

    # npu-smi: required for ascend backend.
    if facts.get("NPU_SMI") == "ok":
        findings.append(Finding("ok", "npu-smi", "in PATH", ""))
    elif ascendish:
        findings.append(Finding(
            "fatal", "npu-smi", "not in PATH",
            "env_script missing source CANN set_env.sh",
        ))
    else:
        findings.append(Finding(
            "info", "npu-smi", "not in PATH", f"Not required for backend={backend}.",
        ))

    # nvidia-smi: required for CUDA backend.
    if facts.get("NVIDIA_SMI") == "ok":
        findings.append(Finding("ok", "nvidia-smi", "in PATH", ""))
    elif cudaish:
        findings.append(Finding(
            "fatal", "nvidia-smi", "not in PATH",
            "A CUDA worker requires nvidia-smi; check the driver, PATH, and env_script.",
        ))
    else:
        findings.append(Finding(
            "info", "nvidia-smi", "not in PATH", f"Not required for backend={backend}.",
        ))

    # arch: backend-specific canonical token; no hard-coded fallback.
    arch = (facts.get("ARCH") or "").strip()
    if arch:
        findings.append(Finding("ok", "npu arch", arch.lower(), ""))
    elif ascendish:
        findings.append(Finding(
            "warn", "npu arch", "Could not infer from npu-smi",
            "Pass --arch explicitly, for example ascend910b3 or ascend950pr.",
        ))

    cuda_arch = (facts.get("CUDA_ARCH") or "").strip()
    if cuda_arch:
        name = (facts.get("CUDA_NAME") or "").strip()
        result = f"{cuda_arch} ({name})" if name else cuda_arch
        findings.append(Finding("ok", "cuda arch", result, ""))
    elif cudaish:
        findings.append(Finding(
            "warn", "cuda arch", "Could not infer from nvidia-smi",
            "Pass --arch explicitly, for example a100, h100, or rtx4090.",
        ))

    cpu_arch = (facts.get("CPU_ARCH") or "").strip()
    if cpu_arch:
        findings.append(Finding("ok", "cpu arch", cpu_arch, ""))
    elif cpuish:
        findings.append(Finding(
            "warn", "cpu arch", "platform.machine() returned empty",
            "Pass --arch explicitly, for example x86_64 or aarch64.",
        ))

    # device count: backend-specific.
    try:
        n = int(facts.get("DEVICES") or "0")
    except ValueError:
        n = 0
    if n > 0:
        findings.append(Finding("ok", "npu devices", f"{n} visible", ""))
    elif ascendish:
        findings.append(Finding(
            "fatal", "npu devices", "0 visible",
            "Check the driver and run `npu-smi info` manually over SSH.",
        ))

    try:
        cuda_n = int(facts.get("CUDA_DEVICES") or "0")
    except ValueError:
        cuda_n = 0
    if cuda_n > 0:
        findings.append(Finding("ok", "cuda devices", f"{cuda_n} visible", ""))
    elif cudaish:
        findings.append(Finding(
            "fatal", "cuda devices", "0 visible",
            "Check the driver and run `nvidia-smi -L` manually over SSH.",
        ))

    # Use the smaller free-space value from the remote POSIX /tmp and /
    # filesystems. A daemon that hits ENOSPC can flood the terminal with
    # secondary logging errors and hide the original failure. A 500 MB floor
    # leaves room for logs, worker_state.json, and temporary files.
    try:
        free_mb = int(facts.get("DISK_FREE_MB") or "0")
    except ValueError:
        free_mb = 0
    if free_mb >= 500:
        findings.append(Finding("ok", "disk free",
                                f"{free_mb} MB", ""))
    elif free_mb > 0:
        findings.append(Finding(
            "fatal", "disk free", f"only {free_mb} MB",
            "The remote disk is almost full and daemon logging may fail with ENOSPC. Clear /tmp and old logs, then retry.",
        ))
    # A zero value means the remote df probe failed; it is not fatal by itself.

    # remote port owner: blocks --start; only informational for status diag.
    port_pid = (facts.get("PORT_PID") or "").strip()
    if not port_pid:
        findings.append(Finding("ok", f"remote :{port}", "free", ""))
    else:
        sev = "fatal" if for_start else "warn"
        suggest = (
            f"The daemon cannot bind this port. Run `ssh <alias> kill {port_pid}` or choose another port."
            if for_start
            else f"Another process owns the port. Run `ssh <alias> kill {port_pid}` or choose another port."
        )
        findings.append(Finding(
            sev, f"remote :{port}", f"held by PID {port_pid}", suggest,
        ))

    return findings


def has_fatal(findings: Iterable[Finding]) -> bool:
    return any(f.severity == "fatal" for f in findings)


def render_findings(findings: Iterable[Finding], log_tail: str = "") -> None:
    """Print findings table + (optional) log tail to stderr via rich."""
    from rich.console import Console
    from rich.table import Table
    color = {"ok": "green", "info": "cyan", "warn": "yellow", "fatal": "red"}
    sym = {"ok": "✓", "info": "ⓘ", "warn": "⚠", "fatal": "✗"}
    console = Console(stderr=True)
    table = Table(title="Remote diagnostics", show_header=True)
    table.add_column("Check", style="cyan")
    table.add_column("", width=2)
    table.add_column("Result")
    table.add_column("Suggestion", style="dim")
    for fnd in findings:
        c = color.get(fnd.severity, "white")
        s = sym.get(fnd.severity, "?")
        table.add_row(fnd.check, f"[{c}]{s}[/{c}]", fnd.result, fnd.suggest)
    console.print(table)
    if log_tail and log_tail.strip() and not log_tail.strip().startswith("(no log"):
        console.print(f"[dim]daemon log tail:[/dim]\n{log_tail}")
