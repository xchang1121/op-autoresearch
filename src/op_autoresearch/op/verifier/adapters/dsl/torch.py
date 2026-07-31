"""
PyTorch DSL adapter - for Kernel → PyTorch conversion scene

Supports the ModelNew (KernelBench) format, where the resulting code is pure PyTorch realization (without any custom Kernel).
The resulting PyTorch code will be compared with the original Kernel output (Triton/CUDA C/ Other).

Usage:
    The target is pure PyTorch.
    - input framework (Triton/CUDA C code is running through PyTonch)
"""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterTorch(DSLAdapter):
    """Adapter for PyTorch DSL (Kernel → PyTorch conversion, supports Triton/CUDA C/etc.)."""

    impl_func_name_template = "ModelNew"

    def get_import_statements(self, framework: str) -> str:
        """Return PyTorch import statements.

        Note: There is no need for Import Triton because the resulting code is pure PyTorch.
        """
        if framework == "torch":
            return "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
        elif framework == "mindspore":
            # MindSpore also uses torch for validation (if needed)
            return "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
        elif framework == "numpy":
            return "import torch\nimport torch.nn as nn\nimport numpy as np\n"
        else:
            return "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import.

        Use the ModelNew class format (KernelBench style) in a uniform way.
        Note: Use _impl suffix to avoid conflict with file_framework
        """
        return f"from {op_name}_torch_impl import ModelNew\n"

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
        code += "impl_model.eval()\n"

        return code

    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str,
                  data_dir: Optional[str] = None,
                  framework_output: Optional[str] = None) -> str:
        """Return code string to call PyTorch implementation function.

        Call an impl_model (can be called several times).
        """
        return f"impl_output = impl_model(*{inputs})\n"

    def benchmark_impl(self, impl_func_name: str, inputs: str,
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      framework: str = "torch") -> str:
        """Return code string to benchmark PyTorch implementation.

        Performance tests are performed using impl_model, which has already been sampled.
        """
        # Select Synchronization Method by Backend
        if backend == "cuda":
            sync_code = "torch.cuda.synchronize()"
        elif backend == "ascend":
            sync_code = "torch.npu.synchronize()"
        else:
            sync_code = "pass  # CPU, no sync needed"

        code = f"""        # PyTorch Nutrient performance test
        import time

        def torch_benchmark_fn():
            result = impl_model(*{inputs})
            return result

        # Preheat
        for _ in range({warmup}):
            _ = torch_benchmark_fn()
            {sync_code}

        # Time
        start_time = time.time()
        for _ in range({runs}):
            _ = torch_benchmark_fn()
            {sync_code}
        end_time = time.time()

        execution_time_ms = (end_time - start_time) * 1000 / {runs}
        method = "pytorch_loop_timer"
"""
        return code

    def get_special_setup_code(self, framework: str = "torch") -> str:
        """Return special setup code (not needed for PyTorch)."""
        return ""
