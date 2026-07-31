"""
NPU Profiller module.

Provide NPU profiling functionality to support:
• Accurate Time Measurement of Implementation

• L2 Cache Clear (optional)

• automatically filter unrelated warning output

"""

import os
import sys
import contextlib
import re
import shutil
import time
from typing import Callable, Tuple, Optional, Literal
import pandas as pd

# Import L2 Cache Clear Related Functions
from .l2_cache_clear import (
    DslType,
    L2_CACHE_CLEAR_KERNEL_NAME,
    clear_l2_cache,
    get_l2_cache_warnings,
    clear_l2_cache_warnings,
)

try:
    from op_autoresearch.op.utils.triton_autotune_patch import OP_AUTORESEARCH_RESTORE_COPY_KERNEL_NAME
except ImportError:
    OP_AUTORESEARCH_RESTORE_COPY_KERNEL_NAME = "OP_AUTORESEARCH_restore_copy"

# Pre-compile regular expressions to enhance performance
# Filter profiller-related noise output
_FILTER_PATTERNS = re.compile(
    r'('
    r'Please DO NOT tune args|'
    r'Invalid parameter export_type|'
    r'Start parsing profiling data|'
    r'CANN profiling data parsed|'
    r'All profiling data parsed|'
    r'\[WARNING\]|'
    r'\[INFO\]|'
    r'profiler\.py:|'
    # Filter triton compiled associated warning
    r'WARNING:\s*Grid.*physical limit|'
    r'WARNING:\s*Grid.*performance'
    r')'
)

_SYMBOL_PATTERN = re.compile(r'^[\\\|/\-_=+*#~`!@$%^&()\[\]{}.,;:\'"<>?\s]+$')
_DECORATION_PATTERN = re.compile(r'[\\\|\-=/]{3,}')


def suppress_output():
    """
    Creates an output inhibitor context manager to filter specific WARNING/INFO outputs.

    Note: This filter does not filter L2 Cache's warning messages.
    These messages are collected through the l2_cache_clar module and output after the profiler.
    """
    class OutputFilter:
        def __init__(self, original_stream):
            self.original_stream = original_stream
            self.suppress_next_lines = 0

        def write(self, text):
            # If the follow-up line is being suppressed, reduce the counter
            if self.suppress_next_lines > 0:
                self.suppress_next_lines -= 1
                if not text.strip():
                    return

            # Quick Match with Precompiled Regular Expressions
            if _FILTER_PATTERNS.search(text):
                self.suppress_next_lines = 2
                return

            stripped_text = text.strip()

            # Total empty lines
            if not stripped_text:
                return

            # Quick check symbol rows with regular expression
            if len(stripped_text) <= 50 and _SYMBOL_PATTERN.match(stripped_text):
                unique_chars = set(stripped_text.replace(' ', '').replace('\t', ''))
                if len(unique_chars) <= 3:
                    return

            # Check decoration lines using regular expression
            if _DECORATION_PATTERN.search(stripped_text):
                return

            # Other Content Normal Output
            self.original_stream.write(text)

        def flush(self):
            if hasattr(self.original_stream, 'flush'):
                self.original_stream.flush()

        def __getattr__(self, name):
            return getattr(self.original_stream, name)

    @contextlib.contextmanager
    def output_suppressor():
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = OutputFilter(old_stdout)
            sys.stderr = OutputFilter(old_stderr)
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return output_suppressor()


def profiler_npu_core(fn: Callable, warmup: int = 25, active: int = 100,
                      prof_dir_name: Optional[str] = None,
                      clear_l2_cache_flag: bool = False,
                      dsl: DslType = "other",
                      filter_restore_copy: bool = False) -> Tuple[float, str]:
    """
    NPU core function (PyTorch version).

    Args:
        {\\fn: function to profile
        Warmup: warmup times
        Activation: Number of effective measurements
        prof_dir_name: result directory name
        clear_l2_cache_flag: clear L2 size before each iterative
        dsl: DSL type, determine L2 Cache clearance method
             ▪ \"triton_ascend\": Use a special triton Kernel (recommended, accurately filtered)

             ▪ Other: Use tensor.zero_() (fallback, risk of error)


    Returns:
        Tuple [float, st]: (execution time (microseconds), profile result directory path)
    """
    import torch
    import torch_npu

    fn()
    torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    skip_first = 1 + warmup
    wait = 0
    warmup_prof = 0
    repeat = 1
    total = skip_first + (wait + warmup_prof + active) * repeat

    timestamp = int(time.time() * 1000)

    if prof_dir_name is not None:
        profile_path = os.path.join(os.getcwd(), f"{prof_dir_name}_{timestamp}")
    else:
        profile_path = os.path.join(os.getcwd(), f"profile_results_{timestamp}")

    if clear_l2_cache_flag:
        clear_l2_cache(dsl, framework="torch")

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=wait, warmup=warmup_prof, active=active,
                                             repeat=repeat, skip_first=skip_first),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total):
            if clear_l2_cache_flag:
                clear_l2_cache(dsl, framework="torch")
            fn()
            prof.step()
            torch.npu.synchronize()

    exec_time = collect_time(profile_path, active, clear_l2_cache_flag=clear_l2_cache_flag,
                             dsl=dsl, framework="torch", filter_restore_copy=filter_restore_copy)
    return exec_time, profile_path


def profiler_npu_mindspore_core(fn: Callable, warmup: int = 25, active: int = 100,
                                prof_dir_name: Optional[str] = None,
                                clear_l2_cache_flag: bool = False,
                                dsl: DslType = "other",
                                filter_restore_copy: bool = False) -> Tuple[float, str]:
    """
    NPU core function (MindSpore version).

    A key difference with the PyTorch version:
    1. AicoreMetrics (non-AiCMetrics)
    Schedule parameters must use keyword references
    3. Data_simplification default is True and needs to be visible as False
    4. Profile() not supported with _flops / with _modules
    5. Synchronization interface: ms.runtime.synchronize() (non-torch.npu.synchronize())

    Args:
        {\\fn: function to profile
        Warmup: warmup times
        Activation: Number of effective measurements
        prof_dir_name: result directory name
        clear_l2_cache_flag: clear L2 size before each iterative
        dsl: DSL type (MindSpore Edit Zero_() Clear L2 Cache)
        Filter_restore_copy: filtering the condition_copy operation

    Returns:
        Tuple [float, st]: (execution time (microseconds), profile result directory path)
    """
    import mindspore as ms
    from mindspore.profiler import (ProfilerActivity, ProfilerLevel, AicoreMetrics,
                                     _ExperimentalConfig, schedule, profile,
                                     tensorboard_trace_handler)

    fn()
    ms.runtime.synchronize()

    experimental_config = _ExperimentalConfig(
        aic_metrics=AicoreMetrics.PipeUtilization,
        profiler_level=ProfilerLevel.Level0,
        l2_cache=False,
        data_simplification=False
    )
    skip_first = 1 + warmup
    wait = 0
    warmup_prof = 0
    repeat = 1
    total = skip_first + (wait + warmup_prof + active) * repeat

    timestamp = int(time.time() * 1000)

    if prof_dir_name is not None:
        profile_path = os.path.join(os.getcwd(), f"{prof_dir_name}_{timestamp}")
    else:
        profile_path = os.path.join(os.getcwd(), f"profile_results_{timestamp}")

    if clear_l2_cache_flag:
        clear_l2_cache(dsl, framework="mindspore")

    with profile(
        activities=[ProfilerActivity.NPU],
        schedule=schedule(wait=wait, warmup=warmup_prof, active=active,
                          repeat=repeat, skip_first=skip_first),
        on_trace_ready=tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total):
            if clear_l2_cache_flag:
                clear_l2_cache(dsl, framework="mindspore")
            fn()
            prof.step()
            ms.runtime.synchronize()

    exec_time = collect_time(profile_path, active, clear_l2_cache_flag=clear_l2_cache_flag,
                             dsl=dsl, framework="mindspore", filter_restore_copy=filter_restore_copy)
    return exec_time, profile_path


def profiler_npu(fn: Callable, warmup: int = 25, active: int = 100, prof_dir_name: Optional[str] = None,
                 keep_res: bool = False, suppress_warnings: bool = True,
                 clear_l2_cache: bool = False, dsl: DslType = "other",
                 filter_restore_copy: bool = False,
                 framework: str = "torch") -> float:
    """
    NPU programr main function.

    Args:
        {\\fn: function to profile
        Warmup: warmup times
        Activation: Number of effective measurements
        prof_dir_name: result directory name
        keep the outcome document or not
        suppress_warnings: inhibit WARNING/INFO output
        clear_l2_cache: Whether to clear L2 size before each iterative
        dsl: DSL type, determine L2 Cache clearance method
             ▪ \"triton_ascend\": Use a special triton Kernel (recommended, accurately filtered)

             ▪ Other: Use tensor.zero_() (fallback, risk of error)

        Filter_restore_copy: filtering the condition_copy operation
        ramework: framework type (\"toch\" or \"mindspore\"), determine which interface to use

    Returns:
        float: Average execution time (microseconds)
    """
    # --trace / OP_AUTORESEARCH_PROF_KEEP_RES: keep the msprof trace dir (timeline + CSVs).
    keep_res = keep_res or os.environ.get("OP_AUTORESEARCH_PROF_KEEP_RES") == "1"
    clear_l2_cache_warnings()

    core_fn = profiler_npu_mindspore_core if framework == "mindspore" else profiler_npu_core

    if suppress_warnings:
        with suppress_output():
            exec_time, profile_path = core_fn(
                fn, warmup, active, prof_dir_name,
                clear_l2_cache_flag=clear_l2_cache, dsl=dsl,
                filter_restore_copy=filter_restore_copy,
            )
    else:
        exec_time, profile_path = core_fn(
            fn, warmup, active, prof_dir_name,
            clear_l2_cache_flag=clear_l2_cache, dsl=dsl,
            filter_restore_copy=filter_restore_copy,
        )

    warnings_list = get_l2_cache_warnings()
    if warnings_list:
        for warning_msg in warnings_list:
            print(f"[WARN] {warning_msg}", file=sys.__stderr__)

    if not keep_res and os.path.exists(profile_path):
        shutil.rmtree(profile_path)

    return exec_time


def collect_time(base_dir: str, active: int, clear_l2_cache_flag: bool = False,
                 dsl: DslType = "other", framework: str = "torch",
                 filter_restore_copy: bool = False) -> float:
    """
    Collect time information from the results of profiling.

    -Torch: Read op_statistic.csv, press Count % action=0 filter, request Total Time(us)/ action
    - Mindspore (Level0): Read kernel_details.csv, press Step ID to take an active step, ask for Duration(us)/ steps

    Args:
        Base_dir: result directory
        Activation: Number of effective measurements
        clear_l2_cache_flag: L2 Cache clear is enabled
        dsl: DSL type
        ramework: framework type (\"toch\" or \"mindspore\")
        Filter_restore_copy: filtering the condition_copy operation

    Returns:
        float: average execution time (microseconds), returns float (`inf')
    """
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return float('inf')

    target_csv = 'kernel_details.csv' if framework == 'mindspore' else 'op_statistic.csv'

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file != target_csv:
                continue

            target_file = os.path.join(root, file)
            try:
                df = pd.read_csv(target_file)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
                print(f"Failed to read {target_file}: {e}")
                continue

            if clear_l2_cache_flag or filter_restore_copy:
                df = _filter_l2_cache_clear_ops(df, dsl, framework=framework,
                                                filter_restore_copy=filter_restore_copy)

            try:
                if framework == 'mindspore':
                    if 'Duration(us)' not in df.columns:
                        print(f"Missing 'Duration(us)' in {target_file}. Found: {list(df.columns)}")
                        continue
                    if 'Step ID' in df.columns:
                        all_steps = sorted(df['Step ID'].dropna().unique())
                        active_steps = all_steps[-active:] if len(all_steps) > active else all_steps
                        df = df[df['Step ID'].isin(active_steps)]
                    if df.empty:
                        print(f"No valid rows in {target_file}")
                        continue
                    num_steps = len(df['Step ID'].unique()) if 'Step ID' in df.columns else active
                    return df['Duration(us)'].sum() / num_steps

                else:
                    required_columns = ['Count', 'Total Time(us)']
                    if not all(col in df.columns for col in required_columns):
                        print(f"Missing required columns in {target_file}. Found: {list(df.columns)}")
                        continue
                    valid_ops = df[df['Count'] % active == 0]
                    if valid_ops.empty:
                        print(f"No valid ops found in {target_file}")
                        continue
                    total_time = valid_ops['Total Time(us)'].sum()
                    if pd.isna(total_time) or total_time <= 0:
                        print(f"Invalid timing data in {target_file}")
                        continue
                    return total_time / active

            except (KeyError, ValueError, ZeroDivisionError) as e:
                print(f"Error processing timing data in {target_file}: {e}")
                continue

    print(f"No valid timing data ({target_csv}) found in {base_dir}")
    return float('inf')


def _filter_l2_cache_clear_ops(df: pd.DataFrame, dsl: DslType,
                                framework: str = "torch",
                                filter_restore_copy: bool = False) -> pd.DataFrame:
    """
    Filter the OP_AUTORESEARCH framework internal operation from the profiling result.

    The listings of op_statistic.csv (torch) and kernel_details.csv (mindspore) are also supported.

    Excluded operations:
    - L2 Cache Clear Kernel (OP_AUTORESEARCH_122cache_clar / ZerosLike)
    - Copy kernel used by restore_value
    """
    if dsl == "triton_ascend":
        col = None
        if 'OP Type' in df.columns:
            col = 'OP Type'
        elif 'Name' in df.columns:
            col = 'Name'

        if col is not None:
            keep = pd.Series(True, index=df.index)
            keep &= ~df[col].str.contains(
                L2_CACHE_CLEAR_KERNEL_NAME, case=False, na=False, regex=False)
            if filter_restore_copy:
                keep &= ~df[col].str.contains(
                    OP_AUTORESEARCH_RESTORE_COPY_KERNEL_NAME, case=False, na=False, regex=False)
            if framework == "mindspore":
                keep &= ~df[col].str.contains(
                    r'ZerosLike', case=False, na=False, regex=False)
            return df[keep]
        return df

    if 'OP Type' in df.columns:
        return df[~df['OP Type'].str.contains(r'^ZerosLike$', case=False, na=False, regex=True)]
    if 'Type' in df.columns:
        return df[~df['Type'].str.contains(r'^ZerosLike$', case=False, na=False, regex=True)]
    return df
