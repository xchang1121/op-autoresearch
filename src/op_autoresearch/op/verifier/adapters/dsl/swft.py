"""SWFT DSL adapter."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterSwft(DSLAdapter):
    """Adapter for SWFT DSL."""

    def get_import_statements(self, framework: str) -> str:
        """Return SWFT import statements."""
        return "from swft.core import *\nfrom swft.api import *\nimport numpy as np\n"

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import."""
        return f"from {op_name}_swft_impl import ModelNew\n"

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
        return code

    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str,
                  data_dir: Optional[str] = None,
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call SWFT implementation function.

        SWFT requires binary I/O, so we need to generate bin files first.
        """
        if data_dir is None:
            data_dir = "os.path.dirname(__file__)"
        if framework_output is None:
            framework_output = "framework_output"

        code = f"""        # RunSWFTAchieved
        data_dir = os.path.dirname(__file__)

        # Generate binary data files
        gen_binary_data({inputs}, {framework_output}, data_dir)

        # RunSWFTAchieved
        impl_model(*{inputs})

        # LoadSWFTOutput
        impl_output = load_binary_data(data_dir, {framework_output})
"""
        return code

    needs_binary_io = True
    static_check_via_python_ast = False  # swft-format src, not stdlib Python

    def benchmark_impl(self, impl_func_name: str, inputs: str,
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      framework: str = "torch") -> str:
        """Return code string to benchmark SWFT implementation."""
        if framework_model is None:
            framework_model = "framework_model"
        if device_id is None:
            device_id = 0

        code = f"""        # RunSWFTAchieved
        data_dir = os.path.dirname(__file__)

        # Generate binary data files
        framework_output = {framework_model}(*{inputs})
        gen_binary_data({inputs}, framework_output, data_dir)

        # RunSWFTAchieved
        import time
        start_time = time.time()
        for _ in range({warmup + runs}):
            impl_model(*{inputs})
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000 / {warmup + runs}  # Convert to milliseconds
        method = "traditional_timing"
"""
        return code


