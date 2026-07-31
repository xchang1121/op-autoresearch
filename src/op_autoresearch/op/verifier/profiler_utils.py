"""
Profiling utilities shared between KernelVerifier and LocalWorker.
Contains methods for running msprof, nsys, and analyzing profiling data.
"""

import os
import re
import json
import logging
import math
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import pandas as pd
from op_autoresearch.core.worker.eval_config import resolve_eval_timeout
from op_autoresearch.utils.process_utils import run_command_capture

logger = logging.getLogger(__name__)


# Canonical per-section schema returned by both the python-script profile
# path here and the msprof/nsys path in LocalWorker. ``per_case_us`` is the
# load-bearing field: a single-element list for static-shape ops keeps the
# downstream consumer iteration uniform with multi-shape ops. ``avg_us`` is
# the arithmetic mean (== per_case_us[0] for the static case) — present so
# legacy callers asking for "the aggregate timing" don't need to redo the
# sum themselves.
#
#   {
#     "avg_us": float,            # mean of per_case_us
#     "per_case_us": [float, ...],# length 1+ ; never empty when present
#     "method": str | None,       # timer name (e.g. "msprof", "loop_timer")
#   }
#
# `run_profile_scripts_and_collect_results` returns
#   {"base": Section | None, "gen": Section | None}
# where ``base is None`` covers the "skipped / no base script / measurement
# failed" cases uniformly. ``gen is None`` means the generation profile
# couldn't be measured (subprocess failed or JSON missing). Callers see one
# None-check, not two sentinel-value branches.


def make_profile_section(avg_us: float,
                         per_case_us: Optional[List[float]] = None,
                         method: Optional[str] = None) -> Dict[str, Any]:
    """Build a canonical profile section. Use this everywhere we synthesize
    a per-shape section from a single aggregate measurement (override
    baseline, msprof/nsys path, etc.) so the schema stays consistent."""
    if per_case_us is None or not per_case_us:
        per_case_us = [float(avg_us)]
    return {
        "avg_us": float(avg_us),
        "per_case_us": [float(t) for t in per_case_us],
        "method": method,
    }


def _finite(x: Any) -> Optional[float]:
    """Coerce to a finite float; None on inf/nan/non-numeric. Used to
    sanitize values read out of profile JSON before they propagate."""
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def read_profile_result_from_json(verify_dir: str,
                                  json_filename: str) -> Optional[Dict[str, Any]]:
    """Read a profile-result JSON written by ``prof_{base,generation}_template_refactored.j2``.

    Returns a canonical section dict (see module docstring) or ``None`` when
    the file is absent / unparsable / inf-only. Templates emit
    ``per_case_us`` (always a list, length 1 for static-shape); we fall back
    to wrapping ``execution_time_us`` so older JSON written by the previous
    template revision still parses (transitional — drop once all task dirs
    have been re-profiled)."""
    json_path = os.path.join(verify_dir, json_filename)
    if not os.path.exists(json_path):
        logger.error(f"profile JSON not found: {json_path}")
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"profile JSON unreadable {json_filename}: {e}")
        return None

    avg = _finite(data.get("avg_time_us")) or _finite(data.get("execution_time_us"))
    if avg is None:
        return None

    raw_per_case = data.get("per_case_us")
    if isinstance(raw_per_case, list) and raw_per_case:
        per_case = [c for c in (_finite(t) for t in raw_per_case) if c is not None]
    else:
        per_case = []
    if not per_case:
        per_case = [avg]
    return {
        "avg_us": avg,
        "per_case_us": per_case,
        "method": data.get("method"),
    }


async def run_profile_scripts_and_collect_results(
    verify_dir: str, op_name: str, run_script, *, task_id: str = "0",
    override_base_section: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Run base + generation profile scripts and collect their canonical
    per-shape sections. Single owner of the base/gen orchestration; the
    caller injects ``run_script(script_name, label) -> Awaitable[bool]`` so
    the subprocess + timeout/kill policy lives with the caller (the worker
    runs each as a killable async subprocess; tests pass a fake).

    Returns ``{"base": Section | None, "gen": Section | None}``. ``base`` is
    ``None`` when the base script is absent (cross-backend / cached) and no
    override was provided, or when its measurement failed. ``gen`` is
    ``None`` when generation measurement failed (subprocess non-zero or JSON
    missing); callers treat that as an infra error.

    ``override_base_section``: canonical Section dict (built via
    :func:`make_profile_section`) used as ``base`` verbatim."""
    base_section: Optional[Dict[str, Any]] = None
    if (override_base_section is not None
            and isinstance(override_base_section.get("avg_us"), (int, float))
            and 0 < override_base_section["avg_us"] < float("inf")):
        base_section = override_base_section
        logger.info(f"[{op_name}: {task_id}] Using Cache baseline: "
                    f"{base_section['avg_us']:.2f} us "
                    f"(per_shape len={len(base_section.get('per_case_us') or [])})")
    elif os.path.exists(os.path.join(verify_dir, f"profile_{op_name}_base.py")):
        if await run_script(f"profile_{op_name}_base.py", "base_profile"):
            base_section = read_profile_result_from_json(
                verify_dir, "base_profile_result.json")
        else:
            logger.error(f"[{op_name}: {task_id}] Benchmark performance script execution failed")
    else:
        logger.info(f"[{op_name}: {task_id}] Base Performance Script does not exist"
                    f"(using caches) baseline Or cross.backendScene) Skip base profile")

    gen_section: Optional[Dict[str, Any]] = None
    if os.path.exists(os.path.join(verify_dir, f"profile_{op_name}_generation.py")):
        if await run_script(f"profile_{op_name}_generation.py", "generation_profile"):
            gen_section = read_profile_result_from_json(
                verify_dir, "generation_profile_result.json")
        else:
            logger.error(f"[{op_name}: {task_id}] Failed to generate code performance scripts")
    else:
        logger.info(f"[{op_name}: {task_id}] Generating code performance scripts doesn't exist."
                    "Skipgeneration profile")

    base_avg = base_section["avg_us"] if base_section else float("inf")
    gen_avg = gen_section["avg_us"] if gen_section else float("inf")
    logger.info(f"[{op_name}: {task_id}] Read profile results: "
                f"base={base_avg:.2f} us, gen={gen_avg:.2f} us "
                f"(base_cases={len(base_section['per_case_us']) if base_section else 0}, "
                f"gen_cases={len(gen_section['per_case_us']) if gen_section else 0})")
    return {"base": base_section, "gen": gen_section}


def run_msprof(script_path: str, op_name: str = "", task_id: str = "0",
               timeout: Optional[int] = None,
               cancel_event=None) -> Tuple[bool, str, Optional[str]]:
    """Run msprofprofiling

    Args:
        Script_path: Python script path
        op_name: operator name (for logs)
        task_id: task ID (for log)
        Timeout: Timeout (sec)

    Returns:
        (success, error_msg, prof_path): Success, error message, prof data path
    """
    timeout = resolve_eval_timeout(timeout)
    try:
        returncode, stdout, stderr, timed_out = run_command_capture(
            ["msprof", f"--application=python {script_path}"],
            timeout=timeout,
            cancel_event=cancel_event,
        )
        if timed_out:
            return False, f"msprof timed out after {timeout} seconds", None
        if returncode != 0:
            return False, stderr or stdout or f"msprof exited with {returncode}", None

        for line in stdout.split('\n'):
            if "[INFO] Process profiling data complete. Data is saved in" in line:
                match = re.search(r"Data is saved in (.+)$", line)
                if match:
                    return True, "", match.group(1).strip()

        return False, "No data saving path found", None
    except Exception as e:
        logger.error(f"[{task_id}:{op_name}] msprofExecute Error: {e}")
        return False, f"Execute Error: {str(e)}", None


def analyze_prof_data(prof_path: str, warmup_times: int, run_times: int, op_name: str = "", task_id: str = "0") -> Tuple[bool, str, float]:
    """Analysis of PROF data

    Args:
        pof_path: pof data directory path
        Warmup_times: number of preheats
        Run_times: Number of times actually running
        op_name: operator name (for logs)
        task_id: task ID (for log)

    Returns:
        (success, error_msg, avg_time_us): Success, error message, average time (microseconds)
    """
    try:
        csv_files = list(Path(prof_path).glob("mindstudio_profiler_output/op_summary_*.csv"))
        if not csv_files:
            return False, "CSV file not found", 0.0

        df = pd.read_csv(csv_files[0])

        # Remove a specific Op
        df_filtered = df[~df["Op Name"].str.contains("aclnnIsClose_IsCloseAiCpu_IsClose|aclnnAll_ReduceAll_ReduceAll",
                                                     regex=True, na=False)]

        total_count = warmup_times + run_times
        op_counts = df_filtered["Op Name"].value_counts()
        valid_ops = op_counts[op_counts == total_count]

        if len(valid_ops) == 0:
            return False, "Op does not match the expected number", float('inf')

        # Check for mismatch Ops
        invalid_ops = op_counts[op_counts != total_count]
        if len(invalid_ops) > 0:
            logger.warning(f"[{task_id}:{op_name}] Found{len(invalid_ops)}individualOpNumber does not match")

        # Calculating average time
        df_valid = df_filtered[df_filtered["Op Name"].isin(valid_ops.index)]
        total_avg_time = 0.0

        for op_name_iter in valid_ops.index:
            op_data = df_valid[df_valid["Op Name"] == op_name_iter]["Task Duration(us)"].tolist()
            if len(op_data) > warmup_times:
                valid_data = op_data[warmup_times:]
                avg_time = sum(valid_data) / len(valid_data)
                total_avg_time += avg_time

        return True, "", total_avg_time

    except Exception as e:
        logger.error(f"[{task_id}:{op_name}] AnalysisprofError while Data: {e}")
        return False, f"Error parsing data: {str(e)}", float('inf')


def run_nsys(script_path: str, op_name: str = "", task_id: str = "0",
             timeout: Optional[int] = None,
             cancel_event=None) -> Tuple[bool, str, Optional[str]]:
    """Run nsysprofiling

    Args:
        Script_path: Python script path
        op_name: operator name (for logs)
        task_id: task ID (for log)
        Timeout: Timeout (sec)

    Returns:
        (success, error_msg, rep_path): Success, error message,nsys reports file path
    """
    timeout = resolve_eval_timeout(timeout)
    try:
        output_name = "nsys_report_" + os.path.basename(script_path).replace(".py", "")
        cmd = ["nsys", "profile", f"--output={output_name}", "python", script_path]
        logger.debug(f"[{task_id}:{op_name}] Running nsys profile: {cmd}")
        returncode, stdout, stderr, timed_out = run_command_capture(
            cmd,
            timeout=timeout,
            cwd=os.path.dirname(script_path),
            cancel_event=cancel_event,
        )
        if timed_out:
            return False, f"nsys timed out after {timeout} seconds", None
        if returncode != 0:
            return False, stderr or stdout or f"nsys exited with {returncode}", None
        report_path = os.path.join(os.path.dirname(script_path), output_name + ".nsys-rep")

        if os.path.exists(report_path):
            return True, "", report_path
        return False, "nsys report file not found", None
    except Exception as e:
        logger.error(f"[{task_id}:{op_name}] nsysExecute Error: {e}")
        return False, f"Execute Error: {str(e)}", None


def analyze_nsys_data(rep_path: str, warmup_times: int, run_times: int,
                      profile_type: str = "", op_name: str = "",
                      task_id: str = "0", cancel_event=None
                      ) -> Tuple[bool, str, float]:
    """Analyze rep files generated by nsys, return average time (us)

    Args:
        Rep_path: nsys report file path
        Warmup_times: number of preheats
        Run_times: Number of times actually running
        Profile_type: profile type identification (for CSV file naming)
        op_name: operator name (for logs)
        task_id: task ID (for log)

    Returns:
        (success, error_msg, avg_time_us): Success, error message, average time (microseconds)
    """
    try:
        dir_plib = Path(rep_path).resolve().parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add the profile_ type identifier to the CSV filename
        type_suffix = f"_{profile_type}" if profile_type else ""
        csv_base = f"nsys_report_{timestamp}{type_suffix}"
        csv_path = dir_plib / csv_base

        # Export csv
        cmd = [
            "nsys",
            "stats",
            "--report",
            "gputrace",
            "--timeunit",
            "us",
            "--format",
            "csv",
            "--output",
            str(csv_path),
            str(rep_path),
        ]
        logger.debug(f"[{task_id}:{op_name}] Running nsys stats: {cmd}")
        returncode, stdout, stderr, timed_out = run_command_capture(
            cmd,
            timeout=resolve_eval_timeout(),
            cancel_event=cancel_event,
        )
        if timed_out:
            return False, "nsys stats timed out", float('inf')
        if returncode != 0:
            return False, stderr or stdout or f"nsys stats exited with {returncode}", float('inf')
        csv_path = dir_plib / f"{csv_base}_gputrace.csv"

        if not os.path.exists(csv_path):
            return False, "No csv file generated", float('inf')

        df = pd.read_csv(csv_path)

        # Compatible with different nsys versions of listing
        name_col = None
        for col in df.columns:
            if col.lower() in ["name", "function name", "kernel name", "Name"]:
                name_col = col
                break
        if not name_col:
            # Underground search for column with name
            for col in df.columns:
                if "name" in col.lower():
                    name_col = col
                    break

        time_col = None
        for col in df.columns:
            if "time (ns)" in col.lower() or "average" in col.lower() or "duration" in col.lower():
                time_col = col
                break

        if not name_col or not time_col:
            return False, "kernel name or time line not found", float('inf')

        total_count = warmup_times + run_times
        op_counts = df[name_col].value_counts()
        valid_ops = op_counts[op_counts == total_count]

        if len(valid_ops) == 0:
            return False, "No kernel found to match the expected number", float('inf')

        df_valid = df[df[name_col].isin(valid_ops.index)]
        total_avg_time = 0.0

        for op_name_iter in valid_ops.index:
            op_data = df_valid[df_valid[name_col] == op_name_iter][time_col].tolist()
            if len(op_data) > warmup_times:
                valid_data = op_data[warmup_times:]
                avg_time = sum(valid_data) / len(valid_data)
                total_avg_time += avg_time  # timeunit us

        return True, "", total_avg_time

    except Exception as e:
        logger.error(f"[{task_id}:{op_name}] AnalysisnsysError while Data: {e}")
        return False, f"AnalysisnsysError while Data: {str(e)}", float('inf')
