"""
Baseline Profiller: Premeasure baseline performance

For the volve/adaptive_search scene, separate caseline once before the start,
Avoids double measurements for all tasks. Supports KernelBench, SOL-ExecBench and CANN-Bench bench_ type.
"""

import os
import io
import json
import shutil
import tarfile
import logging
from contextlib import AsyncExitStack
from typing import Optional, Dict, Any

from op_autoresearch.core.worker.eval_config import (
    resolve_eval_timeout,
    resolve_run_times,
    resolve_warmup_times,
)
from op_autoresearch.op.verifier.data_cache import (
    build_baseline_cache_key,
    build_baseline_cache_payload,
    build_sol_problem_cache_identity,
    delete_baseline_result_from_cache,
    extract_baseline_time_us,
    get_baseline_cache_file_path,
    get_verifier_data_cache_key_id,
    load_verifier_data_cache_config,
    read_baseline_result_from_cache,
    verifier_data_cache_lock,
    write_baseline_result_to_cache,
)

logger = logging.getLogger(__name__)


async def profile_baseline_once(
    op_name: str,
    task_desc: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: Optional[int] = None,
    run_times: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Optional[float]:
    """
    Prefile baseline once (measuring framework performance only)

    Automatically selects the baseline program for KernelBech or SOL-ExecBench according to config[\"bench_type\"].

    device is managed by workinger's data_pool Acquire/release.
    Consistency with normal verifiy/file process.

    Args:
        Op_name: operator name
        task_dec: job description (KernelBench: framework code; SOL: Chinese description text)
        dsl: DSL type
        framework: framework
        Back: backend
        Arch: Structure
        config: Configure Dictionary
        Warmup_times: number of preheats
        Run_times: Run number of times
        Timeout: Timeout

    Returns:
        float: baseline time (microseconds), failed to return None
    """
    warmup_times = resolve_warmup_times(warmup_times)
    run_times = resolve_run_times(run_times)
    timeout = resolve_eval_timeout(timeout)
    bench_type = config.get("bench_type", "kernelbench")

    if bench_type == "sol":
        return await _profile_sol_baseline(
            op_name, dsl, framework, backend, arch, config,
            warmup_times, run_times, timeout
        )
    elif bench_type == "cann":
        return await _profile_cann_baseline(
            op_name, dsl, framework, backend, arch, config,
            warmup_times, run_times, timeout
        )
    else:
        return await _profile_kernelbench_baseline(
            op_name, task_desc, dsl, framework, backend, arch, config,
            warmup_times, run_times, timeout
        )


async def _profile_kernelbench_baseline(
    op_name: str,
    task_desc: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: int,
    run_times: int,
    timeout: int
) -> Optional[float]:
    """Baseline program in KernelBech mode (old logic)"""
    worker = None
    cache_cfg = load_verifier_data_cache_config(config)
    cache_key = None
    cache_file = None
    try:
        from op_autoresearch.op.verifier.kernel_verifier import KernelVerifier
        from op_autoresearch.core.worker.manager import get_worker_manager

        if cache_cfg.enabled and cache_cfg.cache_baseline_result:
            cache_key_id = get_verifier_data_cache_key_id(config, "baseline_profile")
            cache_key = build_baseline_cache_key(
                op_name=op_name,
                framework_code=task_desc,
                framework=framework,
                backend=backend,
                arch=arch,
                bench_type="kernelbench",
                warmup_times=warmup_times,
                run_times=run_times,
                dsl=dsl,
                task_id=cache_key_id,
            )
            cache_file = get_baseline_cache_file_path(
                cache_cfg,
                op_name=op_name,
                cache_key=cache_key,
            )
            cached_entry = read_baseline_result_from_cache(
                cache_cfg,
                op_name=op_name,
                cache_key=cache_key,
            )
            cached_time_us = extract_baseline_time_us(cached_entry)
            if cached_time_us is not None:
                logger.info(
                    f"[{op_name}] ✅ Local hit. baseline cache: {cached_time_us:.2f}us, "
                    f"cache_file={cache_file}, cache_key={cache_key}"
                )
                return cached_time_us
            if cached_entry:
                logger.warning(
                    f"[{op_name}] baseline cache Invalid content, remove old caches and remeasure: "
                    f"cache_file={cache_file}, cache_key={cache_key}"
                )
                delete_baseline_result_from_cache(
                    cache_cfg,
                    op_name=op_name,
                    cache_key=cache_key,
                )

        logger.info(f"[{op_name}] 🚀 Start in advance. profile baseline(One test only)...")

        async with AsyncExitStack() as stack:
            if cache_cfg.enabled and cache_cfg.cache_baseline_result and cache_key:
                await stack.enter_async_context(
                    verifier_data_cache_lock(
                        cache_cfg,
                        namespace="baseline",
                        op_name=op_name,
                        cache_key=cache_key,
                    )
                )
                cached_entry = read_baseline_result_from_cache(
                    cache_cfg,
                    op_name=op_name,
                    cache_key=cache_key,
                )
                cached_time_us = extract_baseline_time_us(cached_entry)
                if cached_time_us is not None:
                    logger.info(
                        f"[{op_name}] ✅ Local hit during the waiting period baseline cache: {cached_time_us:.2f}us, "
                        f"cache_file={cache_file}, cache_key={cache_key}"
                    )
                    return cached_time_us
                if cached_entry:
                    logger.warning(
                        f"[{op_name}] baseline cache Invalid content, remove old caches and remeasure: "
                        f"cache_file={cache_file}, cache_key={cache_key}"
                    )
                    delete_baseline_result_from_cache(
                        cache_cfg,
                        op_name=op_name,
                        cache_key=cache_key,
                    )

            worker_manager = get_worker_manager()
            worker = await worker_manager.select(backend=backend, arch=arch)
            if not worker:
                logger.warning(f"[{op_name}] Unable to access workerSkip the advance. profile baseline")
                return None
            stack.push_async_callback(worker_manager.release, worker)

            device_id = await stack.enter_async_context(
                worker.device_lease("baseline_profile"))
            logger.info(f"[{op_name}] Acquired device {device_id} for baseline profile")

            verifier = KernelVerifier(
                op_name=op_name,
                framework_code=task_desc,
                task_id="baseline_profile",
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                config=config,
                worker=worker
            )

            result = await verifier.profile_single_task(
                task_desc=task_desc,
                warmup_times=warmup_times,
                run_times=run_times,
                timeout=timeout,
                device_id=device_id
            )

            if result.get('success', False):
                baseline_time_us = result.get('time_us')
                if baseline_time_us and baseline_time_us > 0 and baseline_time_us < float('inf'):
                    logger.info(f"[{op_name}] ✅ Baseline profile Completed: {baseline_time_us:.2f}us")
                    _save_baseline_profile_scripts(verifier, op_name, task_desc, warmup_times, run_times, device_id)
                    if cache_cfg.enabled and cache_cfg.cache_baseline_result and cache_key:
                        written_path = write_baseline_result_to_cache(
                            cache_cfg,
                            op_name=op_name,
                            cache_key=cache_key,
                            result_data=build_baseline_cache_payload(
                                base_time_us=baseline_time_us,
                                warmup_times=warmup_times,
                                run_times=run_times,
                                method="profile_single_task",
                            ),
                            metadata={
                                "framework": framework,
                                "dsl": dsl,
                                "cache_key_id": cache_key_id,
                                "backend": backend,
                                "arch": arch,
                                "bench_type": "kernelbench",
                            },
                        )
                        if written_path:
                            logger.info(
                                f"[{op_name}] baseline Results written locally cache: "
                                f"cache_file={written_path}, cache_key={cache_key}, "
                                f"cache_dir={cache_cfg.cache_dir}"
                            )
                    return baseline_time_us
                else:
                    logger.warning(f"[{op_name}] Baseline profile The result is invalid: {baseline_time_us}")
            else:
                error_log = result.get('log', 'Unknown error')
                logger.warning(f"[{op_name}] Baseline profile Failed: {error_log}")

            return None
    except TimeoutError as e:
        logger.warning(f"[{op_name}] Wait. baseline cache lock Overtime, skip ahead. profile baseline: {e}")
        return None
    except Exception as e:
        logger.warning(f"[{op_name}] Advance profile baseline Failed: {e}")
        return None

async def _try_read_baseline_cache(
    cache_cfg, op_name: str, cache_key: str, cache_file: str, bench_label: str,
) -> Optional[float]:
    """Try to read baseline time from cache. Returns cached time_us or None."""
    cached_entry = read_baseline_result_from_cache(
        cache_cfg, op_name=op_name, cache_key=cache_key,
    )
    cached_time_us = extract_baseline_time_us(cached_entry)
    if cached_time_us is not None:
        logger.info(
            f"[{op_name}] ✅ Local hit. {bench_label} baseline cache: {cached_time_us:.2f}us, "
            f"cache_file={cache_file}, cache_key={cache_key}"
        )
        return cached_time_us
    if cached_entry:
        logger.warning(
            f"[{op_name}] {bench_label} baseline cache Invalid content, remove old caches and remeasure: "
            f"cache_file={cache_file}, cache_key={cache_key}"
        )
        delete_baseline_result_from_cache(
            cache_cfg, op_name=op_name, cache_key=cache_key,
        )
    return None


def _parse_profile_log_times(output_log: str, op_name: str) -> list:
    """Parse per-item times from profile output log."""
    times_us = []
    for line in output_log.splitlines():
        stripped = line.strip()
        if stripped and op_name in stripped:
            logger.info(f"[{op_name}] {stripped}")
            if "base time:" in stripped and "us" in stripped and "Geometric mean" not in stripped:
                try:
                    time_str = stripped.split("base time:")[1].strip().replace("us", "").strip()
                    times_us.append(float(time_str))
                except (ValueError, IndexError):
                    pass
    return times_us


def _handle_profile_result(
    result, op_name, profile_dir, times_us,
    warmup_times, run_times, backend, framework, dsl,
    cache_cfg, cache_key, cache_key_id, arch,
    bench_type, bench_label, cache_method, times_label,
) -> Optional[float]:
    """Handle profile result: validate, save, cache. Returns baseline_time_us or None."""
    if not result.get('success', False):
        error_log = result.get('log', 'Unknown error')
        logger.warning(f"[{op_name}] {bench_label} Baseline profile Failed: {error_log}")
        return None

    baseline_time_us = result.get('time_us')
    if not baseline_time_us or baseline_time_us <= 0 or baseline_time_us >= float('inf'):
        logger.warning(f"[{op_name}] {bench_label} Baseline profile The result is invalid: {baseline_time_us}")
        return None

    logger.info(f"[{op_name}] ✅ {bench_label} Baseline profile Completed (geometric average): {baseline_time_us:.2f}us")
    _save_baseline_result_json(
        profile_dir, op_name, baseline_time_us,
        times_us, warmup_times, run_times, backend,
        bench_type, times_label,
    )
    if cache_cfg.enabled and cache_cfg.cache_baseline_result and cache_key:
        write_baseline_result_to_cache(
            cache_cfg,
            op_name=op_name,
            cache_key=cache_key,
            result_data=build_baseline_cache_payload(
                base_time_us=baseline_time_us,
                warmup_times=warmup_times,
                run_times=run_times,
                method=cache_method,
                extra={
                    f"{times_label}_count": len(times_us) if times_us else 0,
                    f"{times_label}_times_us": times_us or [],
                },
            ),
            metadata={
                "framework": framework,
                "dsl": dsl,
                "cache_key_id": cache_key_id,
                "backend": backend,
                "arch": arch,
                "bench_type": bench_type,
            },
        )
    logger.info(f"[{op_name}] {bench_label} Baseline profile Script and result saved to: {profile_dir}")
    return baseline_time_us


def _build_cache_key(cache_cfg, op_name, cache_framework_code, framework, backend, arch, bench_type, warmup_times, run_times, dsl, cache_key_id):
    """Build baseline cache key and file path. Returns (cache_key, cache_file) or (None, None)."""
    if not cache_cfg.enabled or not cache_cfg.cache_baseline_result:
        return None, None
    try:
        cache_key = build_baseline_cache_key(
            op_name=op_name,
            framework_code=cache_framework_code,
            framework=framework,
            backend=backend,
            arch=arch,
            bench_type=bench_type,
            warmup_times=warmup_times,
            run_times=run_times,
            dsl=dsl,
            task_id=cache_key_id,
        )
        cache_file = get_baseline_cache_file_path(cache_cfg, op_name=op_name, cache_key=cache_key)
        return cache_key, cache_file
    except Exception as exc:
        logger.info(f"[{op_name}] baseline cache key Build failed, Skip cache: {exc}")
        return None, None


async def _run_cached_baseline_profile(
    op_name: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: int,
    run_times: int,
    timeout: int,
    bench_type: str,
    bench_label: str,
    cache_framework_code: str,
    prepare_fn,
    cache_method: str,
    times_label: str,
) -> Optional[float]:
    """Common framework for SOL/CANN baseline profiling with cache support.

    Args:
        prepare_fn: async callable(worker, verifier, profile_dir, warmup_times, run_times, timeout)
                     that builds and executes the profile, returning result dict.
    """
    cache_cfg = load_verifier_data_cache_config(config)
    cache_key_id = get_verifier_data_cache_key_id(config, "baseline_profile")
    try:
        from op_autoresearch.op.verifier.kernel_verifier import KernelVerifier
        from op_autoresearch.core.worker.manager import get_worker_manager

        cache_key, cache_file = _build_cache_key(
            cache_cfg, op_name, cache_framework_code, framework, backend, arch,
            bench_type, warmup_times, run_times, dsl, cache_key_id,
        )
        if cache_key:
            cached = await _try_read_baseline_cache(
                cache_cfg, op_name, cache_key, cache_file, bench_label,
            )
            if cached is not None:
                return cached

        logger.info(f"[{op_name}] 🚀 Start in advance. {bench_label} baseline profile(One test only)...")

        async with AsyncExitStack() as stack:
            if cache_cfg.enabled and cache_cfg.cache_baseline_result and cache_key:
                await stack.enter_async_context(
                    verifier_data_cache_lock(
                        cache_cfg, namespace="baseline", op_name=op_name, cache_key=cache_key,
                    )
                )
                cached = await _try_read_baseline_cache(
                    cache_cfg, op_name, cache_key, cache_file, bench_label,
                )
                if cached is not None:
                    return cached

            worker_manager = get_worker_manager()
            worker = await worker_manager.select(backend=backend, arch=arch)
            if not worker:
                logger.warning(f"[{op_name}] Unable to access workerSkip the advance. {bench_label} baseline profile")
                return None
            stack.push_async_callback(worker_manager.release, worker)
            device_id = await stack.enter_async_context(
                worker.device_lease("baseline_profile"))

            verifier = KernelVerifier(
                op_name=op_name,
                framework_code="",
                task_id="baseline_profile",
                framework=framework,
                dsl=dsl,
                backend=backend,
                arch=arch,
                config=config,
                bench_type=bench_type,
                worker=worker,
            )

            profile_dir = os.path.join(
                os.path.expanduser(verifier.log_dir),
                f"{op_name}_profile_single_baseline_profile",
            )
            os.makedirs(profile_dir, exist_ok=True)

            result = await prepare_fn(
                worker, verifier, profile_dir, device_id,
                warmup_times, run_times, timeout)

            times_us = _parse_profile_log_times(result.get('log', ''), op_name)

            return _handle_profile_result(
                result, op_name, profile_dir, times_us,
                warmup_times, run_times, backend, framework, dsl,
                cache_cfg, cache_key, cache_key_id, arch,
                bench_type, bench_label, cache_method, times_label,
            )

    except TimeoutError as e:
        logger.warning(f"[{op_name}] Wait. {bench_label} baseline cache lock Overtime, skip ahead. profile baseline: {e}")
        return None
    except Exception as e:
        logger.warning(f"[{op_name}] {bench_label} baseline profile Failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


async def _prepare_sol_profile(worker, verifier, profile_dir, device_id,
                               warmup_times, run_times, timeout):
    """Prepare and execute SOL baseline profile project."""
    from op_autoresearch.op.verifier.sol_verifier import PROF_SOL_BASE_TEMPLATE_PATH
    from op_autoresearch.op.verifier.adapters.factory import get_framework_adapter, get_backend_adapter
    from op_autoresearch import get_project_root
    from jinja2 import Template

    config = verifier.config
    sol_problem_dir = config.get("sol_problem_dir")
    if not sol_problem_dir:
        raise ValueError("Config ['sol_problem_dir'] not configured")
    sol_problem_dir = os.path.expandvars(os.path.expanduser(str(sol_problem_dir)))
    if not os.path.isdir(sol_problem_dir):
        raise FileNotFoundError(f"SOL case Directory does not exist: {sol_problem_dir}")

    for file_name in ["definition.json", "workload.jsonl", "reference.py"]:
        src = os.path.join(sol_problem_dir, file_name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing required SOL file: {src}")
        shutil.copy2(src, os.path.join(profile_dir, file_name))

    sol_correctness_src = os.path.join(
        get_project_root(), "op", "resources", "utils", "sol_correctness.py",
    )
    shutil.copy2(sol_correctness_src, os.path.join(profile_dir, "sol_correctness.py"))

    framework_adapter = get_framework_adapter(verifier.framework)
    backend_adapter = get_backend_adapter(verifier.backend)
    backend_adapter.setup_environment(device_id, verifier.arch)
    device_setup_code = verifier._prepare_code_lines(
        framework_adapter.get_device_setup_code(
            verifier.backend, verifier.arch, device_id),
    )
    sol_execbench_src_dir = os.path.abspath(
        os.path.join(get_project_root(), "..", "..", "thirdparty", "sol-execbench", "src"),
    )

    with open(PROF_SOL_BASE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        base_template = Template(f.read())

    base_script = base_template.render(
        op_name=verifier.op_name,
        backend=verifier.backend,
        arch=verifier.arch,
        device_id=device_id,
        warmup_times=warmup_times,
        run_times=run_times,
        device_setup_code=device_setup_code,
        sol_execbench_src_dir=sol_execbench_src_dir,
    )

    wrapper = base_script + """

import shutil as _shutil
if os.path.exists("base_profile_result.json"):
    _shutil.copy2("base_profile_result.json", "profile_single_result.json")
"""
    script_path = os.path.join(profile_dir, f"profile_single_{verifier.op_name}.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(wrapper)

    package_data = _pack_directory(profile_dir)
    profile_settings = {
        'warmup_times': warmup_times,
        'run_times': run_times,
        'timeout': timeout,
    }
    return await worker.profile_single_task(
        package_data, "baseline_profile_profile_single", verifier.op_name, profile_settings,
    )


async def _prepare_cann_profile(worker, verifier, profile_dir, device_id,
                                warmup_times, run_times, timeout):
    """Prepare and execute CANN baseline profile project."""
    from op_autoresearch.op.cann_correctness import CANN_BENCH_SRC_DIR, stage_core_into
    from op_autoresearch.op.cann_correctness.verifier import PROF_CANN_BASE_TEMPLATE_PATH
    from op_autoresearch.op.verifier.adapters.factory import get_framework_adapter, get_backend_adapter
    from jinja2 import Template
    import yaml

    config = verifier.config
    cann_problem_dir = config.get("cann_problem_dir")
    if not cann_problem_dir:
        raise ValueError("Config ['cann_problem_dir'] not configured")
    cann_problem_dir = os.path.expandvars(os.path.expanduser(str(cann_problem_dir)))
    if not os.path.isdir(cann_problem_dir):
        raise FileNotFoundError(f"CANN case Directory does not exist: {cann_problem_dir}")

    for file_name in ["proto.yaml", "golden.py", "cases.yaml"]:
        src = os.path.join(cann_problem_dir, file_name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing required CANN file: {src}")
        shutil.copy2(src, os.path.join(profile_dir, file_name))

    desc_src = os.path.join(cann_problem_dir, "desc.md")
    if os.path.exists(desc_src):
        shutil.copy2(desc_src, os.path.join(profile_dir, "desc.md"))

    stage_core_into(profile_dir)

    framework_adapter = get_framework_adapter(verifier.framework)
    backend_adapter = get_backend_adapter(verifier.backend)
    backend_adapter.setup_environment(device_id, verifier.arch)
    device_setup_code = verifier._prepare_code_lines(
        framework_adapter.get_device_setup_code(
            verifier.backend, verifier.arch, device_id),
    )

    proto_path = os.path.join(profile_dir, "proto.yaml")
    with open(proto_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)
    schema = proto.get("operator", {}).get("schema", "")

    cann_bench_src_dir = CANN_BENCH_SRC_DIR

    with open(PROF_CANN_BASE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        base_template = Template(f.read())

    base_script = base_template.render(
        op_name=verifier.op_name,
        backend=verifier.backend,
        arch=verifier.arch,
        dsl=verifier.dsl,
        device_id=device_id,
        warmup_times=warmup_times,
        run_times=run_times,
        device_setup_code=device_setup_code,
        schema=schema,
        cann_bench_src_dir=cann_bench_src_dir,
    )

    wrapper = base_script + """

import shutil as _shutil
if os.path.exists("base_profile_result.json"):
    _shutil.copy2("base_profile_result.json", "profile_single_result.json")
"""
    script_path = os.path.join(profile_dir, f"profile_single_{verifier.op_name}.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(wrapper)

    package_data = _pack_directory(profile_dir)
    profile_settings = {
        'warmup_times': warmup_times,
        'run_times': run_times,
        'timeout': timeout,
    }
    return await worker.profile_single_task(
        package_data, "baseline_profile_profile_single", verifier.op_name, profile_settings,
    )


async def _profile_sol_baseline(
    op_name: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: int,
    run_times: int,
    timeout: int,
) -> Optional[float]:
    """SOL-ExecBench baseline profiling."""
    sol_problem_dir_for_cache = config.get("sol_problem_dir", "")
    if not sol_problem_dir_for_cache:
        logger.warning(f"[{op_name}] config['sol_problem_dir'] Unconfigured, Skip Forward SOL baseline profile")
        return None
    sol_cache_identity = build_sol_problem_cache_identity(sol_problem_dir_for_cache)
    return await _run_cached_baseline_profile(
        op_name, dsl, framework, backend, arch, config,
        warmup_times, run_times, timeout,
        bench_type="sol",
        bench_label="SOL",
        cache_framework_code=sol_cache_identity,
        prepare_fn=_prepare_sol_profile,
        cache_method="sol_profile_single_task",
        times_label="workload",
    )


async def _profile_cann_baseline(
    op_name: str,
    dsl: str,
    framework: str,
    backend: str,
    arch: str,
    config: Dict[str, Any],
    warmup_times: int,
    run_times: int,
    timeout: int,
) -> Optional[float]:
    """CANN-Bench baseline profiling."""
    return await _run_cached_baseline_profile(
        op_name, dsl, framework, backend, arch, config,
        warmup_times, run_times, timeout,
        bench_type="cann",
        bench_label="CANN",
        cache_framework_code=config.get("cann_problem_dir", ""),
        prepare_fn=_prepare_cann_profile,
        cache_method="cann_profile_single_task",
        times_label="case",
    )


def _save_baseline_result_json(
    profile_dir: str,
    op_name: str,
    baseline_time_us: float,
    times_us: list,
    warmup_times: int,
    run_times: int,
    backend: str,
    bench_type: str,
    times_label: str,
) -> None:
    """Write the result as base_profile_result.json to base_dir"""
    try:
        method_prefix = f"{bench_type}_base"
        method = f"{method_prefix}_profiler_npu" if backend == "ascend" else f"{method_prefix}_loop_timer"
        if bench_type == "sol":
            method = "sol_base_profiler_npu" if backend == "ascend" else "sol_base_do_bench"
        result_data = {
            "execution_time_ms": baseline_time_us / 1000.0,
            "execution_time_us": baseline_time_us,
            "avg_time_us": baseline_time_us,
            "warmup_times": warmup_times,
            "run_times": run_times,
            f"{times_label}_count": len(times_us) if times_us else 0,
            f"{times_label}_times_us": times_us or [],
            "method": method,
            "bench_type": bench_type,
        }
        result_file = os.path.join(profile_dir, "base_profile_result.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
        logger.info(f"[{op_name}] base_profile_result.json Written: {result_file}")
    except Exception as e:
        logger.warning(f"[{op_name}] Writing base_profile_result.json Failed: {e}")


def _pack_directory(dir_path: str) -> bytes:
    """Pack directory as tar byte"""
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w') as tar_file:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dir_path)
                tar_file.add(file_path, arcname=arcname)
    return tar_buffer.getvalue()


def _save_baseline_profile_scripts(verifier, op_name: str, task_desc: str,
                                   warmup_times: int, run_times: int,
                                   device_id: int = 0) -> None:
    """
    Save KernelBench case profile script to log directory
    """
    try:
        baseline_dir = os.path.join(
            os.path.expanduser(verifier.log_dir),
            f"{op_name}_baseline_profile"
        )
        os.makedirs(baseline_dir, exist_ok=True)

        framework_file = verifier._materialize_framework_bundle(
            baseline_dir, task_desc)

        script_file = os.path.join(baseline_dir, f"profile_baseline_{op_name}.py")
        verifier.gen_profile_single_task_file(script_file, device_id=device_id,
                                              warmup_times=warmup_times,
                                              run_times=run_times)

        logger.info(f"[{op_name}] Baseline profile Script saved to: {baseline_dir}")

    except Exception as e:
        logger.warning(f"[{op_name}] Save baseline profile Script failed: {e}")


def set_baseline_in_config(config: Dict[str, Any], baseline_time_us: float) -> None:
    """
    Sets the cache baseon time to config

    Args:
        config: Configure Dictionary
        Baseline_time_us: baseline time (microseconds)
    """
    # Set only if baseline_time_us is valid
    if baseline_time_us is None or baseline_time_us <= 0 or baseline_time_us >= float('inf'):
        return

    if 'profile_settings' not in config:
        config['profile_settings'] = {}

    from op_autoresearch.op.verifier.profiler_utils import make_profile_section
    config['profile_settings']['override_base_section'] = make_profile_section(
        baseline_time_us, method="override")
    config['profile_settings']['skip_base_profile'] = True
