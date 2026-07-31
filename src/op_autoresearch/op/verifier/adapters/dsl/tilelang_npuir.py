"""TileLang NPUIR DSL adapter."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterTilelangNpuir(DSLAdapter):
    """Adapter for TileLang NPUIR DSL."""

    profile_via_python_script = True

    def get_import_statements(self, framework: str) -> str:
        """Return TileLang NPUIR import statements."""
        code = """import tilelang
tilelang.cache.clear_cache()
try:
    from op_autoresearch.op.utils.tilelang_compile_patch import apply_tilelang_patches
    apply_tilelang_patches()
except ImportError:
    pass
"""
        if framework == "torch":
            code += "import torch\nimport torch_npu\nimport tilelang.language as T\n"
        return code

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import."""
        return f"from {op_name}_tilelang_npuir_impl import {impl_func_name}\n"

    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str,
                  data_dir: Optional[str] = None,
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call TileLang NPUIR implementation function."""
        return f"impl_output = {impl_func_name}(*{inputs})\n"

    def benchmark_impl(self, impl_func_name: str, inputs: str,
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      clear_l2_cache: bool = True,
                      framework: str = "torch") -> str:
        """Return code string to benchmark TileLang NPUIR implementation.

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
        if backend == "ascend":
            framework_arg = f', framework="{framework}"' if framework == "mindspore" else ""
            # Performance test with profiler_npu to support L2 Cache cleanup
            code = f"""        # dsl:tilelang_npuir
        try:
            from op_autoresearch.op.verifier.profiler import profiler_npu
            patch_imported = True
        except ImportError:
            patch_imported = False

        def tilelang_benchmark_fn():
            return {framework_model}(*{inputs})

        if patch_imported:
            execution_time_us = profiler_npu(
                tilelang_benchmark_fn,
                warmup={warmup},
                active={runs},
                prof_dir_name=f"prof_generation_output_case_{{case_idx}}",
                keep_res=False,
                suppress_warnings=True,
                clear_l2_cache={clear_l2_cache},
                dsl="other"{framework_arg}
            )
            execution_time_ms = execution_time_us / 1000
            method = "profiler_npu"
        else:
            import time
            start_time = time.time()
            for _ in range({warmup + runs}):
                _ = tilelang_benchmark_fn()
                torch.npu.synchronize()
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000 / {warmup + runs}
            method = "traditional_timing"
"""
        else:
            # Non ascend backend, when using fax statistics
            sync_code = "torch.npu.synchronize()" if backend == "ascend" else ""
            code = f"""        # dsl:tilelang_npuir
        import time
        start_time = time.time()
        for _ in range({warmup + runs}):
            framework_output = {framework_model}(*{inputs})
            {sync_code}
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000 / {warmup + runs}  # Convert to milliseconds
        method = "traditional_timing"
"""
        return code

    def get_special_setup_code(self, framework: str = "torch") -> str:
        """Return special setup code for tilelang_npuir."""
        return """import tilelang
tilelang.cache.clear_cache()
try:
    from op_autoresearch.op.utils.tilelang_compile_patch import apply_tilelang_patches
    apply_tilelang_patches()
except ImportError:
    pass
"""

