"""TileLang-Ascend DSL adapter - ModelNew (KernelBench)."""

from typing import Any, Optional

from .base import DSLAdapter


class DSLAdapterTilelangAscend(DSLAdapter):
    """Adapter for TileLang-Ascend DSL."""

    profile_via_python_script = True
    impl_func_name_template = "ModelNew"

    def get_import_statements(self, framework: str) -> str:
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
        return f"from {op_name}_tilelang_ascend_impl import ModelNew\n"

    def create_impl_module(self, framework: str,
                          framework_adapter: Any,
                          init_params_var: str = "init_params",
                          device_var: str = "device") -> str:
        code = f"impl_model = ModelNew(*{init_params_var})\n"
        if framework == "torch":
            code += f"impl_model = impl_model.to({device_var})\n"
        return code

    def call_impl(self, impl_func_name: str, inputs: str, device_id: int,
                  framework_adapter: Any, op_name: str,
                  data_dir: Optional[str] = None,
                  framework_output: Optional[str] = None) -> str:
        return f"impl_output = impl_model(*{inputs})\n"

    def benchmark_impl(self, impl_func_name: str, inputs: str,
                      warmup: int, runs: int, backend: str, op_name: str,
                      case_idx: int = 0, framework_model: Optional[str] = None,
                      framework_adapter: Optional[Any] = None,
                      device_id: Optional[int] = None,
                      clear_l2_cache: bool = True,
                      framework: str = "torch") -> str:
        """Return code string to benchmark TileLang Ascend implementation.

        Alignment of Tillang.profiller.do_bench:
        - Time with torch.npu. Event high accuracy
        - Support L2 Cache Clear (256MB Buffer + Zero_)
        - Autowarmup and multiple measurements to take the minimum
        """
        if framework == "torch":
            code = f"""
        import torch

        def tilelang_benchmark_fn():
            return impl_model(*{inputs})

        # Let's do it once and make sure it's done.
        tilelang_benchmark_fn()
        torch.npu.synchronize()

        # L2 cache Clear buffer( Alignment) tilelang.profiler.do_bench)
        cache = None
        if {clear_l2_cache}:
            cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="npu")

        # Use torch.npu.Event HighaccuracyTime
        start_event = [torch.npu.Event(enable_timing=True) for _ in range({runs})]
        end_event = [torch.npu.Event(enable_timing=True) for _ in range({runs})]

        # warmup
        for _ in range({warmup}):
            if cache is not None:
                cache.zero_()
            tilelang_benchmark_fn()
        torch.npu.synchronize()

        # timing
        for i in range({runs}):
            if cache is not None:
                cache.zero_()
            start_event[i].record()
            tilelang_benchmark_fn()
            end_event[i].record()

        torch.npu.synchronize()
        times = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
            dtype=torch.float,
        )
        execution_time_ms = torch.mean(times).item()
        method = "tilelang_event_timing"
"""
        else:
            raise ValueError(
                f"TileLang Ascend currently only supports framework='torch', "
                f"got framework='{framework}'"
            )
        return code

    def get_special_setup_code(self, framework: str = "torch") -> str:
        return """import tilelang
tilelang.cache.clear_cache()
try:
    from op_autoresearch.op.utils.tilelang_compile_patch import apply_tilelang_patches
    apply_tilelang_patches()
except ImportError:
    pass
"""
