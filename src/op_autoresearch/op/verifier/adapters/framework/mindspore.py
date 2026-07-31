"""MindSpore framework adapter."""

import os
from typing import Any, Optional
import mindspore as ms
from mindspore.common import np_dtype
import numpy as np

from op_autoresearch.op.utils.config_utils import check_backend_arch
from .base import FrameworkAdapter


class FrameworkAdapterMindSpore(FrameworkAdapter):
    """Adapter for MindSpore framework."""

    def get_import_statements(self) -> str:
        """Return MindSpore import statements."""
        return "import mindspore as ms\nfrom mindspore.common import np_dtype\n"

    def get_framework_import(
        self,
        op_name: str,
        is_dynamic_shape: bool,
        inputs_factory_name: Optional[str] = None,
        module_name: Optional[str] = None,
    ) -> str:
        local = "get_inputs_dyn_list" if is_dynamic_shape else "get_inputs"
        factory = inputs_factory_name or local
        module = module_name or f"{op_name}_mindspore"
        return (f"from {module} import Model as FrameworkModel, "
                f"get_init_inputs, {factory} as {local}\n")

    def setup_device(self, backend: str, arch: str, device_id: int) -> Any:
        """Setup MindSpore device."""
        os.environ['DEVICE_ID'] = str(device_id)
        if backend == "ascend":
            check_backend_arch(backend, arch)
            ms.set_device("Ascend", device_id)
            return "Ascend"
        elif backend == "cpu":
            ms.set_device("CPU")
            return "CPU"
        else:
            raise ValueError(f"MindSporeUnsupportedbackend: {backend}")

    def process_input(self, x: Any, device: Any) -> Any:
        """Process input (MindSpore doesn't need device movement)."""
        return x

    def convert_to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert MindSpore tensor to numpy."""
        if isinstance(tensor, ms.Tensor):
            return tensor.flatten().asnumpy()
        return tensor.flatten() if hasattr(tensor, 'flatten') else tensor

    def get_limit(self, dtype: Any) -> float:
        """Get precision rtol for dtype (backward compatibility)."""
        if dtype == ms.float32:
            return 1.22e-4
        elif dtype == ms.float16:
            return 9.77e-4
        elif dtype == ms.bfloat16:
            return 7.81e-3
        else:
            return 1.22e-4

    def save_tensor(self, tensor: Any, bin_path: str) -> None:
        """Save MindSpore tensor to binary file."""
        tensor_np = tensor.asnumpy()
        uint8_view = tensor_np.view(np.uint8)
        with open(bin_path, 'wb') as f:
            f.write(uint8_view.tobytes())

    def load_tensor(self, bin_path: str, reference_tensor: Any) -> Any:
        """Load MindSpore tensor from binary file."""
        with open(bin_path, 'rb') as f:
            data = f.read()
            uint8_array = np.frombuffer(data, dtype=np.uint8)
            numpy_dtype = self.get_dtype_mapping().get(reference_tensor.dtype)
            if numpy_dtype is None:
                raise ValueError(f"Unsupporteddata type: {reference_tensor.dtype}")
            numpy_tensor = uint8_array.view(numpy_dtype).reshape(reference_tensor.shape)
            return ms.Tensor(numpy_tensor, dtype=reference_tensor.dtype)

    def set_seed(self, backend: Optional[str] = None) -> None:
        """Set random seed."""
        ms.manual_seed(0)

    def move_model_to_device(self, model: Any, device: Any) -> Any:
        """Move model to device (MindSpore doesn't need explicit move)."""
        return model

    def get_tensor_type(self) -> type:
        """Get MindSpore tensor type."""
        return ms.Tensor

    def get_tensor_type_name(self) -> str:
        """Get MindSpore tensor type name as string (full path)."""
        return "ms.Tensor"

    def get_dtype_mapping(self) -> dict:
        """Get MindSpore to NumPy dtype mapping."""
        return {
            ms.float32: np.float32,
            ms.float16: np.float16,
            ms.bfloat16: np_dtype.bfloat16,
            ms.int8: np.int8,
            ms.int16: np.int16,
            ms.int32: np.int32,
            ms.int64: np.int64,
            ms.uint8: np.uint8,
            ms.uint16: np.uint16,
            ms.uint32: np.uint32,
            ms.uint64: np.uint64,
            ms.bool_: np.bool_,
        }

    def _get_save_tensor_code(self, tensor_type: str) -> str:
        """Get save_tensor function code for MindSpore."""
        return """def save_tensor(tensor: TensorType, bin_path: str):
    \"\"\"willMindSporetensorSave as Binary File\"\"\"
    tensor_np = tensor.asnumpy()
    uint8_view = tensor_np.view(np.uint8)
    with open(bin_path, 'wb') as f:
        f.write(uint8_view.tobytes())

"""

    def _get_load_tensor_code(self, tensor_type: str) -> str:
        """Get load_tensor function code for MindSpore."""
        return """def load_tensor(bin_path: str, expect_tensor: TensorType) -> TensorType:
    \"\"\"Load from binary fileMindSporetensor\"\"\"
    with open(bin_path, 'rb') as f:
        data = f.read()
        uint8_array = np.frombuffer(data, dtype=np.uint8)
        numpy_dtype = MS_TO_NP_DTYPE_MAP.get(expect_tensor.dtype)
        if numpy_dtype is None:
            raise ValueError(f"Unsupporteddata type: {expect_tensor.dtype}")
        numpy_tensor = uint8_array.view(numpy_dtype).reshape(expect_tensor.shape)
        return ms.Tensor(numpy_tensor, dtype=expect_tensor.dtype)

"""

    def _get_gen_binary_data_code(self, tensor_type: str, op_name: str) -> str:
        """Get gen_binary_data function code."""
        return f"""def gen_binary_data(inputs, outputs, data_dir):
    \"\"\"Generate binary data files

    Args:
        inputs: InputtensorList
        outputs: OutputtensorList or individualtensor
        data_dir: Data Save Directory
    \"\"\"
    import os
    os.makedirs(data_dir, exist_ok=True)

    # Create Input Output Directory
    input_dir = os.path.join(data_dir, "{op_name}", "input")
    output_dir = os.path.join(data_dir, "{op_name}", "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Save input data
    for i, input_tensor in enumerate(inputs):
        if isinstance(input_tensor, TensorType):
            bin_path = os.path.join(input_dir, f"input{{i}}.bin")
            save_tensor(input_tensor, bin_path)

    # Processing output data
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]  # Will be SingletensorConvert to List

    # SavegoldenOutput
    for i, output_tensor in enumerate(outputs):
        if isinstance(output_tensor, TensorType):
            golden_path = os.path.join(output_dir, f"output{{i}}_golden.bin")
            save_tensor(output_tensor, golden_path)

"""

    def _get_load_binary_data_code(self, tensor_type: str, op_name: str) -> str:
        """Get load_binary_data function code."""
        return f"""def load_binary_data(data_dir, reference_outputs):
    \"\"\"Load binary data files and convert them totensor

    Args:
        data_dir: Data Directory
        reference_outputs: Reference OutputtensorList or individualtensor, for determiningdata typeandshape

    Returns:
        LoadedtensorList
    \"\"\"
    import os
    if not isinstance(reference_outputs, (list, tuple)):
        reference_outputs = [reference_outputs]

    output_dir = os.path.join(data_dir, "{op_name}", "output")
    loaded_outputs = []
    i = 0
    while True:
        output_path = os.path.join(output_dir, f"output{{i}}_actual.bin")
        if not os.path.exists(output_path):
            break
        if i >= len(reference_outputs):
            raise RuntimeError(f"Number of Output Files({{i+1}})More than the reference output({{len(reference_outputs)}})")
        loaded_outputs.append(load_tensor(output_path, reference_outputs[i]))
        i += 1

    if not loaded_outputs:
        raise RuntimeError("No output file found, Usually because of inputdata type.... and the original task inputdata typeDo not match")

    return loaded_outputs

"""

    def get_device_setup_code(self, backend: str, arch: str, device_id: int) -> str:
        """Get device setup code for MindSpore."""
        code = f"""    os.environ['DEVICE_ID'] = str({device_id})
"""
        if backend == "ascend":
            check_backend_arch(backend, arch)
            code += f"""    ms.set_device("Ascend", {device_id})
    device = "Ascend"
"""
        elif backend == "cpu":
            code += """    ms.set_device("CPU")
    device = "CPU"
"""
        return code

    def get_process_input_code(self, backend: str, dsl: str) -> str:
        """Get process_input function code for MindSpore."""
        return """    def process_input(x):
        \"\"\"Processing input data\"\"\"
        return x
"""

    def get_set_seed_code(self, backend: str) -> str:
        """Get set seed code for MindSpore.

        Note: Returns code without indentation, template will handle indentation.
        """
        return """ms.manual_seed(0)
"""

    def get_compare_code(self) -> str:
        """Get compare function code using layered tolerance (hard-coded, no config)."""
        return '''def _get_tolerance(data_type):
    """Hard-coded tolerance table aligned with CANN / NPUKernelBench.

    Returns (rtol, atol, outlier_rtol, outlier_atol, outlier_ratio).
    - strict_tol  = atol       + rtol       * |ref|
    - relaxed_tol = outlier_atol + outlier_rtol * |ref|
    - outlier_atol = 10 * atol  (aligned with PyTorch MatMul FP32 atol=1e-4)
    - outlier_rtol = 10 * rtol  (aligned with CANN MARE threshold = 10 * MERE threshold)
    """
    if data_type == ms.float32:
        return (1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001)
    elif data_type == ms.float16:
        return (9.77e-4, 1e-3, 9.77e-3, 1e-2, 0.005)
    elif data_type == ms.bfloat16:
        return (7.81e-3, 1e-2, 7.81e-2, 1e-1, 0.01)
    else:
        return (1.22e-4, 1e-5, 1.22e-3, 1e-4, 0.001)

def _merge_consecutive(values):
    """Merge sorted integer list into consecutive ranges.

    [0,1,2,5,6,7,10] -> [(0,3), (5,8), (10,11)]
    """
    if not values:
        return []
    ranges = []
    start = values[0]
    end = values[0]
    for v in values[1:]:
        if v == end + 1:
            end = v
        else:
            ranges.append((start, end + 1))
            start = v
            end = v
    ranges.append((start, end + 1))
    return ranges

def _format_dim(values, dim_size):
    """Format per-dimension error distribution with auto-merge."""
    ranges = _merge_consecutive(values)
    total_error = len(values)
    coverage = total_error / dim_size * 100

    if len(ranges) == 1:
        lo, hi = ranges[0]
        if hi - lo == dim_size:
            return f"[:]" + f"  ({total_error}/{dim_size} = {coverage:.1f}%)"
        return f"[{lo}:{hi}]" + f"  ({total_error}/{dim_size} = {coverage:.1f}%)"

    if len(ranges) <= 5:
        parts = [f"[{lo}:{hi}]" for lo, hi in ranges]
        return ", ".join(parts) + f"  ({total_error}/{dim_size} = {coverage:.1f}%)"

    first3 = [f"[{lo}:{hi}]" for lo, hi in ranges[:3]]
    return ", ".join(first3) + f", ... ({len(ranges)} ranges, {total_error}/{dim_size} = {coverage:.1f}%)"

def _format_error_locations(error_mask, shape):
    """Format per-dimension error distribution without materializing all coords."""
    if len(shape) == 0:
        return "Error location: scalar output"

    lines = ["Error location per dimension ([start:end]=error index range, count/size=coverage):"]
    non_singleton_dims = []
    full_coverage_dims = []
    singleton_dims = []

    for d, dim_size in enumerate(shape):
        if dim_size == 1:
            singleton_dims.append(d)
            continue

        reduce_axes = tuple(i for i in range(len(shape)) if i != d)
        dim_mask = np.any(error_mask, axis=reduce_axes) if reduce_axes else error_mask
        unique_vals = np.where(dim_mask)[0].tolist()
        non_singleton_dims.append(d)
        if len(unique_vals) == dim_size:
            full_coverage_dims.append(d)
        lines.append(f"  dim{d}: {_format_dim(unique_vals, dim_size)}")

    if not non_singleton_dims:
        lines.append("  note: All output dimensions are individual, with reference to the following sample values.")
    elif len(non_singleton_dims) == 1:
        lines.append("  note: There is only one non-single output dimension, and there is less additional information in the index of the relative sample by dimension.")
    elif len(full_coverage_dims) == len(non_singleton_dims):
        lines.append("  note: Error overwrite all non-single dimensions, check global formulae, add,dtype,store or buffer Covering, not only local boundaries mask.")

    if singleton_dims:
        lines.append(f"  note: Single dimensions {singleton_dims} They have been omitted as they provide less information on location.")

    return "\\n".join(lines)

def _coord_from_flat(flat_idx, shape):
    """Convert a flattened index to an ND coordinate tuple."""
    idx = int(flat_idx)
    coord = []
    for dim_size in reversed(shape):
        coord.append(idx % dim_size)
        idx //= dim_size
    return tuple(reversed(coord))

def _format_coord(coord):
    if len(coord) == 0:
        return "[scalar]"
    if len(coord) == 1:
        return f"[{coord[0]}]"
    return str(list(coord))

def compare(fw_out, impl_out, data_type):
    """Compare framework output and implementation output using layered tolerance."""
    fw_np = fw_out.asnumpy()
    impl_np = impl_out.asnumpy() if isinstance(impl_out, ms.Tensor) else np.asarray(impl_out, dtype=fw_np.dtype)

    size = fw_np.size

    if fw_np.shape != impl_np.shape:
        raise AssertionError(f"Validation Failed, OutputshapeInconsistencies: framework={fw_np.shape}, impl={impl_np.shape}")

    fw_nan_mask = np.isnan(fw_np)
    impl_nan_mask = np.isnan(impl_np)
    if not np.array_equal(fw_nan_mask, impl_nan_mask):
        fw_nan_count = np.sum(fw_nan_mask)
        impl_nan_count = np.sum(impl_nan_mask)
        raise AssertionError(f"Validation failed.NaNLocation does not match: Framework={fw_nan_count}/{size}, Implementation={impl_nan_count}/{size}")
    if np.sum(fw_nan_mask) > 0:
        nan_count = np.sum(fw_nan_mask)
        print(f"DetectedNaNValue: {nan_count}/{size} (We\'re in position. Continue to verify.)")

    fw_inf_mask = np.isinf(fw_np)
    impl_inf_mask = np.isinf(impl_np)
    if not np.array_equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = np.sum(fw_inf_mask)
        impl_inf_count = np.sum(impl_inf_mask)
        raise AssertionError(f"Validation failed.InfLocation does not match: Framework={fw_inf_count}/{size}, Implementation={impl_inf_count}/{size}")
    if np.sum(fw_inf_mask) > 0:
        inf_sign_match = np.array_equal(
            np.sign(fw_np[fw_inf_mask]),
            np.sign(impl_np[impl_inf_mask])
        )
        if not inf_sign_match:
            raise AssertionError(f"Validation failed.InfThe symbol does not match")

    finite_mask = np.isfinite(fw_np) & np.isfinite(impl_np)
    finite_count = np.sum(finite_mask)
    if finite_count == 0:
        print(f"Warning: All the values.InfSkipaccuracyInspection")
        return

    fw_finite = fw_np[finite_mask]
    impl_finite = impl_np[finite_mask]

    if fw_finite.dtype == bool or impl_finite.dtype == bool:
        if not np.array_equal(fw_finite, impl_finite):
            raise AssertionError(f"Validation failed. Boolean values do not match: dtype={data_type}")
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.astype(fw_finite.dtype)

    rtol, atol, outlier_rtol, outlier_atol, outlier_ratio = _get_tolerance(data_type)

    abs_diff = np.abs(fw_finite - impl_finite)
    abs_ref = np.abs(fw_finite)
    strict_tol = atol + rtol * abs_ref
    relaxed_tol = outlier_atol + outlier_rtol * abs_ref

    strict_pass = abs_diff <= strict_tol
    relaxed_pass = abs_diff <= relaxed_tol

    hard_fail = int(np.sum(~relaxed_pass))
    outlier = int(np.sum((~strict_pass) & relaxed_pass))
    total = fw_finite.size
    cap = int(total * outlier_ratio)

    mere = float(np.mean(abs_diff / (abs_ref + atol)))
    mare = float(np.max(abs_diff / (abs_ref + atol)))
    print(f"[precision] dtype={data_type} total={total} strict={int(np.sum(strict_pass))} outlier={outlier}/{cap} hard={hard_fail} mere={mere:.6e} mare={mare:.6e}")

    if hard_fail > 0:
        hard_fail_mask = np.zeros(fw_np.shape, dtype=bool)
        hard_fail_mask[finite_mask] = ~relaxed_pass
        sample_flat_indices = np.where(hard_fail_mask.reshape(-1))[0][:5]
        error_msg = f"Validation failed. Existence {hard_fail} One element above the relaxing threshold(hard_fail)\\n"
        error_msg += f"rtol={rtol:.6e} atol={atol:.6e} outlier_rtol={outlier_rtol:.6e} outlier_atol={outlier_atol:.6e} outlier_ratio={outlier_ratio}\\n"
        error_msg += f"mere={mere:.6e} mare={mare:.6e}\\n"
        error_msg += _format_error_locations(hard_fail_mask, fw_np.shape) + "\\n"
        for flat_idx in sample_flat_indices.tolist():
            coord = _coord_from_flat(flat_idx, fw_np.shape)
            ref_value = np.float32(fw_np[coord])
            impl_value = np.float32(impl_np[coord])
            sample_abs_diff = float(np.abs(ref_value - impl_value))
            sample_relaxed_tol = float(outlier_atol + outlier_rtol * np.abs(ref_value))
            error_msg += f"  Location{_format_coord(coord)}: ref={ref_value:.6e} impl={impl_value:.6e} abs_diff={sample_abs_diff:.6e} relaxed_tol={sample_relaxed_tol:.6e}\\n"
        raise AssertionError(error_msg)

    if outlier > cap:
        outlier_mask = np.zeros(fw_np.shape, dtype=bool)
        outlier_mask[finite_mask] = (~strict_pass) & relaxed_pass
        sample_flat_indices = np.where(outlier_mask.reshape(-1))[0][:5]
        error_msg = f"Validation failed, excess factor ratio exceeded allowed value: outlier={outlier} / cap={cap}\\n"
        error_msg += f"rtol={rtol:.6e} atol={atol:.6e} outlier_rtol={outlier_rtol:.6e} outlier_atol={outlier_atol:.6e} outlier_ratio={outlier_ratio}\\n"
        error_msg += f"mere={mere:.6e} mare={mare:.6e}\\n"
        error_msg += _format_error_locations(outlier_mask, fw_np.shape) + "\\n"
        for flat_idx in sample_flat_indices.tolist():
            coord = _coord_from_flat(flat_idx, fw_np.shape)
            ref_value = np.float32(fw_np[coord])
            impl_value = np.float32(impl_np[coord])
            sample_abs_diff = float(np.abs(ref_value - impl_value))
            sample_strict_tol = float(atol + rtol * np.abs(ref_value))
            sample_relaxed_tol = float(outlier_atol + outlier_rtol * np.abs(ref_value))
            error_msg += f"  Location{_format_coord(coord)}: ref={ref_value:.6e} impl={impl_value:.6e} abs_diff={sample_abs_diff:.6e} strict_tol={sample_strict_tol:.6e} relaxed_tol={sample_relaxed_tol:.6e}\\n"
        raise AssertionError(error_msg)

'''

    def get_compare_outputs_code(self) -> str:
        """Get code for comparing framework output and impl output."""
        return '''            data_type = framework_output[i].dtype
            compare(fw_out, impl_out, data_type)
'''
