"""C++ DSL adapter - Support ModelNew (KernelBench) format."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterCpp(DSLAdapter):
    """Adapter for C++ DSL."""

    static_check_via_python_ast = False  # C++ inline string, no Python AST

    def get_import_statements(self, framework: str) -> str:
        """Return C++ import statements."""
        return "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.cpp_extension import load_inline\n"

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import.

        Use the ModelNew class format (KernelBench style) in a uniform way.
        """
        return f"from {op_name}_cpp_impl import ModelNew\n"

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
        """Return code string to call C++ implementation function.

        Call an impl_model (can be called several times).
        """
        return f"impl_output = impl_model(*{inputs})\n"

    def benchmark_impl(self, impl_func_name: str, inputs: str,
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      framework: str = "torch") -> str:
        """Return code string to benchmark C++ implementation.

        Performance tests are performed using impl_model, which has already been sampled.
        """
        code = f"""        # CPU
        import time
        def cpp_benchmark_fn():
            return impl_model(*{inputs})
        # Implementation warmup
        for _ in range({warmup}):
            _ = cpp_benchmark_fn()
        # Time rep Numbers
        start_t = time.perf_counter()
        for _ in range({runs}):
            _ = cpp_benchmark_fn()
        end_t = time.perf_counter()
        execution_time_ms = (end_t - start_t) * 1000.0 / max({runs}, 1)
        method = "cpu_loop_timer"
"""
        return code



