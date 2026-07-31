"""PyPTO DSL adapter - Support ModelNew (KernelBench) format."""

from typing import Any, Optional
import re

from .base import DSLAdapter


class DSLAdapterPypto(DSLAdapter):
    """Adapter for PyPTO DSL.

    PyPTO is a new language used to generate NPU operator, using @pypto.jit decorations and slice syntax.
    Unlike Triton, PyPTO uses tensor [start:end] syntax instead of tl.load/store.
    """

    profile_via_python_script = True

    def get_import_statements(self, framework: str) -> str:
        """Return PyPTO import statements."""
        code = ""
        if framework == "torch":
            code += "import torch\n"
            code += "import pypto\n"
        elif framework == "mindspore":
            code += "import torch\n"
            code += "import pypto\n"
        elif framework == "numpy":
            code += "import numpy as np\n"
            code += "import torch\n"
            code += "import pypto\n"
        else:
            code += "import pypto\n"
        code += """import os
"""
        return code

    def get_runtime_env_override_code(
        self,
        pypto_run_mode: Optional[int] = None,
        pypto_runtime_debug_mode: Optional[int] = None,
    ) -> str:
        """Return code to inject per-task PyPTO runtime env overrides."""
        lines = []
        if pypto_run_mode is not None:
            lines.append(f'os.environ["OP_AUTORESEARCH_PYPTO_RUN_MODE"] = "{pypto_run_mode}"')
            lines.append(
                'print(f"[INFO] Task override: OP_AUTORESEARCH_PYPTO_RUN_MODE={os.environ[\'OP_AUTORESEARCH_PYPTO_RUN_MODE\']}")'
            )
        if pypto_runtime_debug_mode is not None:
            lines.append(
                f'os.environ["OP_AUTORESEARCH_PYPTO_RUNTIME_DEBUG_MODE"] = "{pypto_runtime_debug_mode}"'
            )
            lines.append(
                'print(f"[INFO] Task override: OP_AUTORESEARCH_PYPTO_RUNTIME_DEBUG_MODE={os.environ[\'OP_AUTORESEARCH_PYPTO_RUNTIME_DEBUG_MODE\']}")'
            )
        return "\n".join(lines) + ("\n" if lines else "")

    def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
        """Return implementation function import.

        Use the ModelNew class format (KernelBench style) in a uniform way.
        """
        module_name = re.sub(r"\W", "_", op_name)
        if not module_name or module_name[0].isdigit():
            module_name = f"op_{module_name}"
        return (
            "import importlib.util\n"
            "import os\n"
            f"_impl_module_name = '{module_name}_pypto_impl'\n"
            f"_impl_module_path = os.path.join(os.path.dirname(__file__), '{op_name}_pypto_impl.py')\n"
            "_impl_spec = importlib.util.spec_from_file_location(_impl_module_name, _impl_module_path)\n"
            "_impl_module = importlib.util.module_from_spec(_impl_spec)\n"
            "_impl_spec.loader.exec_module(_impl_module)\n"
            "ModelNew = _impl_module.ModelNew\n"
        )

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
        """Return code string to call PyPTO implementation function.

        Call ``impl_model`` already sampled.
        """
        return (
            f"impl_output = impl_model(*{inputs})\n"
        )

    def benchmark_impl(self, impl_func_name: str, inputs: str,
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      framework: str = "torch") -> str:
        """Return code string to benchmark PyPTO implementation.

        Performance tests are performed using impl_model, which has already been sampled.
        PyPTO runs on NPU, using NPU programr for performance testing.
        """
        code = f"""        import os
        import json
        import sys
        import subprocess
        from pathlib import Path

        # Define Performance Test Functions
        def pypto_benchmark_fn():
            result = impl_model(*{inputs})
            return result

        def _calc_trace_span_us(trace_path):
            try:
                from op_autoresearch import get_project_root
                script = (Path(get_project_root()) / "op" / "tools" /
                          "calc_trace_span.py")
                if script.exists():
                    out = subprocess.check_output(
                        [sys.executable, str(script), str(trace_path)],
                        text=True
                    )
                    return float(out.strip())
            except Exception:
                pass
            with open(trace_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            events = [e for e in data.get("traceEvents", []) if e.get("ph") == "X"]
            if not events:
                raise RuntimeError("Not found ph==X Events")
            min_ts = min(float(e.get("ts", 0) or 0) for e in events)
            max_end = max(
                float(e.get("ts", 0) or 0) + float(e.get("dur", 0) or 0)
                for e in events
            )
            return max_end - min_ts

        def _find_latest_swimlane(base_dir):
            \"\"\"Find merged_swimlane.json from profiler output directory.\"\"\"
            import glob
            if not os.path.isdir(base_dir):
                raise FileNotFoundError(f"Profiler output dir not found: {{base_dir}}")

            patterns = [
                os.path.join(base_dir, "output_*", "merged_swimlane.json"),
                os.path.join(base_dir, "merged_swimlane.json"),
                os.path.join(base_dir, "**", "merged_swimlane.json"),
            ]
            candidates = []
            for pattern in patterns:
                candidates.extend(glob.glob(pattern, recursive=("**" in pattern)))
            candidates = sorted(set(candidates), key=os.path.getmtime, reverse=True)
            if candidates:
                return candidates[0]
            try:
                entries = sorted(os.listdir(base_dir))
            except Exception:
                entries = []
            raise FileNotFoundError(
                f"merged_swimlane.json not found under {{base_dir}}. "
                f"Searched patterns: {{patterns}}. "
                f"Top-level entries: {{entries[:30]}}"
            )

        if backend == "ascend":
            # persistent The output directory and log status must be reset at each time in the scene;
            # Otherwise... pypto Could be reused from last cache output_* subdirectories, which result in files not found for secondary running.
            output_dir = os.path.abspath(f"prof_generation_output_case{case_idx}")
            os.environ["TILE_FWK_OUTPUT_DIR"] = output_dir
            os.makedirs(output_dir, exist_ok=True)
            try:
                if hasattr(pypto, "pypto_impl") and hasattr(pypto.pypto_impl, "ResetLog"):
                    pypto.pypto_impl.ResetLog("")
            except Exception as e:
                print(f"[WARN] pypto ResetLog failed: {{e}}")
            # PyPTO profile I don't need it. warmup
            pypto_benchmark_fn()
            trace_path = _find_latest_swimlane(output_dir)
            print(f"[INFO] PyPTO trace path: {{trace_path}}")
            execution_time_us = _calc_trace_span_us(trace_path)
            execution_time_ms = execution_time_us / 1000
            method = "trace_span"
        else:
            # Simple timer (nil) warmup)
            import time
            times = []
            for _ in range({runs}):
                start = time.perf_counter()
                pypto_benchmark_fn()
                end = time.perf_counter()
                times.append((end - start) * 1000)
            execution_time_ms = min(times) if times else 0.0
            method = "simple_timing"
"""
        return code

