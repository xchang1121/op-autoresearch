import os

_collected_config_timings = {}
_current_framework = "torch"


def set_framework(framework: str):
    """Sets the current framework type to influence the behaviour of op_autoresearch_restore_copy / benchmarker."""
    global _current_framework
    _current_framework = framework
    if framework == "mindspore":
        os.environ["TRITON_BACKEND"] = "mindspore"


def get_framework() -> str:
    return _current_framework

# ============================================================================
# OP_AUTORESEARCH_restore_copy Triton kernel
# Reference l2_cache_clear.py design: Using kernel, with the prefix of OP_AUTORESEARCH,
# Easy to filter by name in profiller 's op_statistic.csv.
# ============================================================================

OP_AUTORESEARCH_RESTORE_COPY_KERNEL_NAME = "OP_AUTORESEARCH_restore_copy"

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass

if _TRITON_AVAILABLE:
    @triton.jit
    def OP_AUTORESEARCH_restore_copy(
        dst_ptr, src_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr,
    ):
        """
        Restore_value is dedicated to copy kernel.

        kernel name with OP_AUTORESEARCH_ prefix, displayed as OP_AUTORESEARCH_restore_copy,
        Precisely filtered without deleting the TensorMove equivalent in the user code.
        """
        pid = tl.program_id(0)
        num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
        for block_idx in range(pid, num_blocks, CORE_NUM):
            block_start = block_idx * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            data = tl.load(src_ptr + offsets, mask=mask)
            tl.store(dst_ptr + offsets, data, mask=mask)


def _get_core_nums(vec_default=40, cube_default=20):
    if get_framework() == "mindspore":
        import mindspore as ms
        limits = ms.runtime.get_device_limit(0)
        vec = limits.get("vector_core_num", vec_default)
        cube = limits.get("cube_core_num", cube_default)
        return (vec, cube)
    vec, cube = vec_default, cube_default
    try:
        import torch
        import triton
        device = torch.npu.current_device()
        properties = triton.runtime.driver.active.utils.get_device_properties(device)
        vec = properties.get("num_vectorcore", vec_default)
        cube = properties.get("num_aicore", cube_default)
    except Exception:
        pass
    return (vec, cube)


def op_autoresearch_restore_copy_torch(dst, src):
    """An OP_AUTORESEARCH_restore_copy kernel is used to execute tensor copy (PyTorch version)."""
    import torch
    n = dst.numel()
    dst_flat = dst.view(-1)
    src_flat = src.view(-1)
    core_num, _ = _get_core_nums()
    BLOCK_SIZE = 1024
    grid = (core_num,)
    OP_AUTORESEARCH_restore_copy[grid](dst_flat, src_flat, n,
                           BLOCK_SIZE=BLOCK_SIZE, CORE_NUM=core_num)
    torch.npu.synchronize()


def op_autoresearch_restore_copy_mindspore(dst, src):
    """An OP_AUTORESEARCH_restore_copy Kernel is used to execute tensor copy (MindSpore version)."""
    import mindspore as ms
    n = dst.numel()
    dst_flat = dst.view(-1)
    src_flat = src.view(-1)
    core_num, _ = _get_core_nums()
    BLOCK_SIZE = 1024
    grid = (core_num,)
    OP_AUTORESEARCH_restore_copy[grid](dst_flat, src_flat, n,
                           BLOCK_SIZE=BLOCK_SIZE, CORE_NUM=core_num)
    ms.runtime.synchronize()


def op_autoresearch_restore_copy(dst, src):
    """Use OP_AUTORESEARCH_restore_copykel to execute tensor copy instead of tensor.copy_()."""
    if get_framework() == "mindspore":
        op_autoresearch_restore_copy_mindspore(dst, src)
    else:
        op_autoresearch_restore_copy_torch(dst, src)


def _restore_saved_tensors(saved, args):
    """Restore saved output tensors back to the live kernel arguments."""
    for idx, saved_val in saved.items():
        op_autoresearch_restore_copy(args[idx], saved_val)


def _wrap_kernel_call_with_restore(kernel_call, restore_info):
    """Wrap benchmark calls with Triton-like pre/post restore semantics."""
    if restore_info is None:
        return kernel_call

    saved = restore_info['saved']
    args = restore_info['args']

    def wrapped_call():
        _restore_saved_tensors(saved, args)
        try:
            return kernel_call()
        finally:
            # Leave every benchmark iteration with the original output state
            # so a later config cannot inherit stale values from an earlier one.
            _restore_saved_tensors(saved, args)

    return wrapped_call


# ============================================================================
# _Bench Patch: Disable primary copy of _value.
# Let kernel_call contain only pure kernel, restore to benchmarker with a name kernel.
# ============================================================================

_restore_info = None


def _patch_autotuner_bench(autotuner_module):
    """Patch Autotuner. _Bench, take over under conditione_value."""
    original_bench = getattr(autotuner_module.Autotuner, '_bench', None)
    if original_bench is None:
        return
    if getattr(original_bench, '_op_autoresearch_bench_patched', False):
        return

    _noop = lambda *a, **kw: None

    def patched_bench(self, *args, config, **meta):
        global _restore_info

        if not (_TRITON_AVAILABLE and hasattr(self, 'restore_value') and self.restore_value):
            _restore_info = None
            return original_bench(self, *args, config=config, **meta)

        saved = {}
        for name in self.restore_value:
            idx = self.fn.arg_names.index(name)
            saved[idx] = args[idx].clone()
        _restore_info = {'saved': saved, 'args': list(args)}

        orig_rv = self.restore_value
        orig_ph = getattr(self, 'pre_hook', None)
        orig_posth = getattr(self, 'post_hook', None)
        self.restore_value = None
        self.pre_hook = _noop
        self.post_hook = _noop

        try:
            result = original_bench(self, *args, config=config, **meta)
        finally:
            self.restore_value = orig_rv
            self.pre_hook = orig_ph
            self.post_hook = orig_posth
            _restore_info = None

        return result

    patched_bench._op_autoresearch_bench_patched = True
    autotuner_module.Autotuner._bench = patched_bench


# ============================================================================
# Bottom realization parameters that need filtering
# ============================================================================

_FILTERED_CONFIG_PARAMS = {
    'num_warps',
    'num_ctas',
    'num_stages',
    'num_buffers_warp_spec',
    'num_consumer_groups',
    'reg_dec_producer',
    'reg_inc_consumer',
    'maxnreg'
}


def _filter_config_string(config_str: str) -> str:
    """Filter configuration string, remove bottom realization parameters"""
    params = []
    for param in config_str.split(','):
        param = param.strip()
        if not param:
            continue
        if ':' in param:
            param_name = param.split(':', 1)[0].strip()
        elif '=' in param:
            param_name = param.split('=', 1)[0].strip()
        else:
            params.append(param)
            continue
        if param_name not in _FILTERED_CONFIG_PARAMS:
            params.append(param)
    return ', '.join(params)


def patch_triton_autotuner():
    """Dynamic patch, add configuration information collection + _bench conditione_value to take over."""
    try:
        import triton.runtime.autotuner as autotuner_module
    except ImportError:
        return True

    try:
        import triton.runtime.autotiling_tuner as autotiling_module
    except ImportError:
        autotiling_module = None

    if not hasattr(autotuner_module, 'Autotuner'):
        return True

    original_autotuner_run = getattr(autotuner_module.Autotuner, 'run', None)
    if original_autotuner_run is None:
        return True
    if getattr(original_autotuner_run, '_op_autoresearch_run_patched', False):
        return True

    original_autotiling_run = None
    if autotiling_module and hasattr(autotiling_module, 'AutoTilingTuner'):
        original_autotiling_run = getattr(autotiling_module.AutoTilingTuner, 'run', None)

    # Patch _bench take over restore_value
    _patch_autotuner_bench(autotuner_module)

    def _process_config_timings(self):
        if not (hasattr(self, 'best_config') and
                hasattr(self, 'configs_timings') and
                self.configs_timings and
                isinstance(self.configs_timings, dict)):
            return

        func_name = "unknown_function"
        try:
            if hasattr(self, 'base_fn') and hasattr(self.base_fn, '__name__'):
                func_name = self.base_fn.__name__
            elif hasattr(self, 'fn') and hasattr(self.fn, '__name__'):
                func_name = self.fn.__name__
        except (AttributeError, TypeError):
            pass

        try:
            sorted_timings = sorted(self.configs_timings.items(), key=lambda x: x[1])
            config_data = []
            for i, (config, timing) in enumerate(sorted_timings):
                try:
                    is_best = config == self.best_config
                    timing_value = timing[0] if isinstance(timing, list) else timing
                    timing_us = timing_value
                    config_str = _filter_config_string(str(config))
                    config_data.append({
                        "config": config_str,
                        "timing_us": float(timing_us),
                        "is_best": is_best,
                        "rank": i + 1
                    })
                except (TypeError, ValueError, AttributeError):
                    continue

            if config_data:
                global _collected_config_timings
                if func_name not in _collected_config_timings:
                    _collected_config_timings[func_name] = config_data

                    if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
                        print(f"All config timings for {func_name}:")
                        for i, (config, timing) in enumerate(sorted_timings):
                            try:
                                status = " (BEST)" if config == self.best_config else ""
                                timing_value = timing[0] if isinstance(timing, list) else timing
                                timing_us = timing_value
                                config_str = _filter_config_string(str(config))
                                print(f"  Config {i+1}: {config_str} -> {timing_us:.4f}us{status}")
                            except (TypeError, ValueError, AttributeError):
                                continue

        except (TypeError, ValueError, AttributeError):
            pass

    def patched_autotuner_run(self, *args, **kwargs):
        result = original_autotuner_run(self, *args, **kwargs)
        try:
            _process_config_timings(self)
        except Exception:
            pass
        return result

    def patched_autotiling_run(self, *args, **kwargs):
        result = original_autotiling_run(self, *args, **kwargs)
        try:
            _process_config_timings(self)
        except Exception:
            pass
        return result

    try:
        patched_autotuner_run._op_autoresearch_run_patched = True
        autotuner_module.Autotuner.run = patched_autotuner_run
    except (AttributeError, TypeError):
        pass

    if original_autotiling_run is not None:
        try:
            patched_autotiling_run._op_autoresearch_run_patched = True
            autotiling_module.AutoTilingTuner.run = patched_autotiling_run
        except (AttributeError, TypeError):
            pass

    return True


def get_collected_config_timings():
    global _collected_config_timings
    return _collected_config_timings.copy()


def clear_collected_config_timings():
    global _collected_config_timings
    _collected_config_timings = {}


def patch_driver_benchmarker():
    """Patches driver.active.get_benchmarker(), allowing autotune to use profler_npu.

    When _restore_info is not empty (i.e. _bench disables the original restore_value),
    Benchmarker uses OP_AUTORESEARCH_restore_copy kernel_call,
    Profiler filters accurately by kernel name and does not miss the user 's TensorMove operation.
    """
    try:
        from triton.runtime import driver

        if hasattr(driver.active.get_benchmarker, '_op_autoresearch_patched'):
            return True

        original_get_benchmarker = driver.active.get_benchmarker

        def patched_get_benchmarker():
            def custom_benchmarker(kernel_call, quantiles=(0.5, 0.2, 0.8)):
                fn_to_profile = _wrap_kernel_call_with_restore(kernel_call, _restore_info)

                try:
                    from op_autoresearch.op.verifier.profiler import profiler_npu

                    time_us = profiler_npu(
                        fn_to_profile,
                        warmup=5,
                        active=30,
                        suppress_warnings=True,
                        clear_l2_cache=True,
                        dsl="triton_ascend",
                        filter_restore_copy=(_restore_info is not None),
                        framework=get_framework(),
                    )
                    return [time_us] * 3

                except ImportError:
                    original_benchmarker = original_get_benchmarker()
                    return original_benchmarker(fn_to_profile, quantiles)

            return custom_benchmarker

        driver.active.get_benchmarker = patched_get_benchmarker
        driver.active.get_benchmarker._op_autoresearch_patched = True
        return True

    except ImportError:
        return False
    except Exception as e:
        print(f"Warning: Failed to patch driver benchmarker: {e}")
        return False


def apply_triton_patches():
    """Apply all triton patches"""
    success1 = patch_triton_autotuner()
    success2 = patch_driver_benchmarker()
    return success1 or success2


if __name__ != "__main__":
    apply_triton_patches()

if __name__ == "__main__":
    print("Testing Triton patches...")
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    success1 = patch_triton_autotuner()
    success2 = patch_driver_benchmarker()

    if success1:
        print("Autotuner patch applied successfully!")
    if success2:
        print("Driver benchmarker patch applied successfully!")

    if not any([success1, success2]):
        print("Failed to apply patches")
