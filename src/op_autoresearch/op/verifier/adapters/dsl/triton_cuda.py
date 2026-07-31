"""Triton CUDA DSL adapter - Support ModelNew (KernelBench) format."""

from typing import Any, Optional, Tuple

from .base import DSLAdapter


class DSLAdapterTritonCuda(DSLAdapter):
    """Adapter for Triton CUDA DSL."""

    profile_via_python_script = True
    impl_func_name_template = "ModelNew"
    supports_autotune_configs = True

    def get_import_statements(self, framework: str) -> str:
        """Return Triton import statements."""
        if framework == "mindspore":
            return "import torch\nimport triton\nimport triton.language as tl\n"
        elif framework == "torch":
            return "import triton\nimport triton.language as tl\n"
        elif framework == "numpy":
            return "import numpy as np\nimport triton\nimport triton.language as tl\n"
        else:
            return "import triton\nimport triton.language as tl\n"

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import.

        Use the ModelNew class format (KernelBench style) in a uniform way.
        """
        return f"from {op_name}_triton_cuda_impl import ModelNew\n"

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
        """Return code string to call Triton CUDA implementation function.

        Call an impl_model (can be called several times).
        """
        return f"impl_output = impl_model(*{inputs})\n"

    def benchmark_impl(self, impl_func_name: str, inputs: str,
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      framework: str = "torch") -> str:
        """Return code string to benchmark Triton CUDA implementation.

        Performance tests are performed using impl_model, which has already been sampled.
        """
        code = f"""        # Final performance test.
        def triton_benchmark_fn():
            result = impl_model(*{inputs})
            return result

        import triton.testing
        execution_time_ms = triton.testing.do_bench(
            triton_benchmark_fn,
            warmup={warmup},
            rep={runs},
            return_mode="median"
        )
        method = "triton_do_bench"
"""
        return code

