"""Triton Ascend DSL adapter - Supports ModelNew (KernelBench) format."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterTritonAscend(DSLAdapter):
    """Adapter for Triton Ascend DSL."""

    profile_via_python_script = True
    impl_func_name_template = "ModelNew"
    profiler_dsl = "triton_ascend"
    supports_autotune_configs = True
    emits_autotune_artifacts = True

    def get_import_statements(self, framework: str) -> str:
        """Return Triton Ascend import statements."""
        code = ""
        if framework == "mindspore":
            code += """import os
os.environ["TRITON_BACKEND"] = "mindspore"
try:
    from op_autoresearch.op.utils.triton_autotune_patch import set_framework
    set_framework("mindspore")
except ImportError:
    pass
"""
        code += """try:
    from op_autoresearch.op.utils.triton_autotune_patch import apply_triton_patches
    apply_triton_patches()
except ImportError:
    pass
"""
        if framework == "numpy":
            code += "import numpy as np\n"
        code += "import triton\nimport triton.language as tl\n"
        return code

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import.

        Use the ModelNew class format (KernelBench style) in a uniform way.
        """
        return f"from {op_name}_triton_ascend_impl import ModelNew\n"

    def create_impl_module(self, framework: str,
                          framework_adapter: Any,
                          init_params_var: str = "init_params",
                          device_var: str = "device") -> str:
        """Generates the code that creates impl_model (examples only once).

        Args:
            framework: Framework name (torch, mindspore, numpy)
            framework_adapter: Framework adapter instance
            init_params_var: Variable name for init_params (default: "init_params")
            device_var: Variable name for device (default: "device")

        Returns:
            str: Code string to create impl_model
        """
        code = f"impl_model = ModelNew(*{init_params_var})\n"
        if framework == "torch":
            code += f"impl_model = impl_model.to({device_var})\n"

        return code

    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str,
                  data_dir: Optional[str] = None,
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call Triton Ascend implementation function.

        Call an impl_model (can be called several times).
        """
        return f"impl_output = impl_model(*{inputs})\n"

    def benchmark_impl(self, impl_func_name: str, inputs: str,
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      clear_l2_cache: bool = True,
                      framework: str = "torch") -> str:
        """Return code string to benchmark Triton Ascend implementation.

        Performance tests are performed using impl_model, which has already been sampled.

        Args:
            impl_func_name: achieve function name
            inputs: Enter variable name
            Warmup: warmup times
            Runs: Effective run times
            Back: backend type
            Op_name: operator name
            Case_idx: case index
            ramework_model: framework Model Variable Name (optional)
            framework_adapter: framework adapter (optional)
            Data_id: deviceID (optional)
            clear_l2_cache: Whether to clear L2 Cache (default True) before each iterative
            ramework: framework type (\"toch\" or \"mindspore\")
        """
        framework_arg = f', framework="{framework}"' if framework == "mindspore" else ""
        set_framework_code = ""
        if framework == "mindspore":
            set_framework_code = """        import os
        os.environ["TRITON_BACKEND"] = "mindspore"
        try:
            from op_autoresearch.op.utils.triton_autotune_patch import set_framework
            set_framework("mindspore")
        except ImportError:
            pass
"""
        code = f"""{set_framework_code}        try:
            from op_autoresearch.op.verifier.profiler import profiler_npu
            from op_autoresearch.op.utils.triton_autotune_patch import get_collected_config_timings, clear_collected_config_timings
            # Clear pre-configuration information
            clear_collected_config_timings()
            patch_imported = True
        except ImportError:
            get_collected_config_timings = lambda: {{}}
            clear_collected_config_timings = lambda: None
            patch_imported = False

        # Clear Cache to Ensure Re-entryautotune
        if hasattr(impl_model, 'cache'):
            impl_model.cache.clear()

        # Triggerautotune
        impl_model(*{inputs})

        # Get collected configuration information
        config_timings = get_collected_config_timings()

        # SaveautotuneCan not open message
        if config_timings:
            autotune_filename = f"autotune_info_case_{case_idx}.json"
            try:
                with open(autotune_filename, 'w') as f:
                    json.dump(config_timings, f, indent=2, ensure_ascii=False)
                print(f"[{op_name}] Autotune info saved to {{autotune_filename}}")
            except Exception as e:
                print(f"[{op_name}] Warning: Failed to save autotune info: {{e}}")

        # Final performance test.
        def triton_benchmark_fn():
            result = impl_model(*{inputs})
            return result

        if backend == "ascend" and patch_imported:
            # Use triton_ascend It's for personal use. L2 cache Clear Method
            # Pass. OP_AUTORESEARCH_l2cache_clear kernel Cleared, available profiler Medium Precision Filter
            execution_time_us = profiler_npu(
                triton_benchmark_fn,
                warmup={warmup},
                active={runs},
                prof_dir_name=f"prof_generation_output_case_{{case_idx}}",
                keep_res=False,
                suppress_warnings=True,
                clear_l2_cache={clear_l2_cache},
                dsl="triton_ascend"{framework_arg}
            )
            execution_time_ms = execution_time_us / 1000
            method = "profiler_npu"
        else:
            # GPUEnvironment or patch import failed: use standarddo_bench
            import triton.testing
            execution_time_ms = triton.testing.do_bench(
                triton_benchmark_fn,
                warmup={warmup},
                rep={runs},
                return_mode="min"
            )
            method = "triton_do_bench"
"""
        return code

    def get_special_setup_code(self, framework: str = "torch") -> str:
        """Return special setup code for triton_ascend."""
        code = ""
        if framework == "mindspore":
            code += """import os
os.environ["TRITON_BACKEND"] = "mindspore"
try:
    from op_autoresearch.op.utils.triton_autotune_patch import set_framework
    set_framework("mindspore")
except ImportError:
    pass
"""
        code += """try:
    from op_autoresearch.op.utils.triton_autotune_patch import apply_triton_patches
    apply_triton_patches()
except ImportError:
    pass
"""
        return code

