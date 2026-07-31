import os
import re
import shutil
import logging
import json
import textwrap
from datetime import datetime
from typing import Optional, Literal, Tuple, Dict, Any, List, Union
from jinja2 import Template
from pathlib import Path

from op_autoresearch import get_project_root
from op_autoresearch.op.utils.config_utils import normalize_dsl
from op_autoresearch.op.utils.task_layout import REF_FILE_DEFAULT
from op_autoresearch.op.verifier.adapters.factory import (
    get_framework_adapter, get_dsl_adapter, get_backend_adapter
)
from op_autoresearch.op.verifier.profiler_utils import make_profile_section
from op_autoresearch.op.verifier.data_cache import (
    build_baseline_cache_key,
    build_baseline_cache_payload,
    build_reference_cache_key,
    build_sol_problem_cache_identity,
    delete_baseline_result_from_cache,
    delete_reference_data_from_cache,
    extract_baseline_time_us,
    get_baseline_cache_file_path,
    get_reference_cache_file_path,
    get_verifier_data_cache_key_id,
    load_verifier_data_cache_config,
    read_baseline_result_from_cache,
    read_reference_data_from_cache,
    verifier_data_cache_lock,
    write_baseline_result_to_cache,
    write_reference_data_to_cache,
)
from op_autoresearch.core.worker.interface import WorkerInterface, empty_profile_result
from op_autoresearch.core.worker.eval_config import (
    resolve_eval_timeout,
    resolve_reference_timeout,
    resolve_run_times,
    resolve_warmup_times,
)
import tarfile
import io
import ast

# Template Path
TEMPLATE_PATH = os.path.join(get_project_root(), "op", "resources", "templates",
                             "kernel_verify_template_refactored.j2")
PROFILE_BASE_TEMPLATE_PATH = os.path.join(get_project_root(), "op", "resources",
                                          "templates", "prof_base_template_refactored.j2")
PROFILE_GENERATION_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "prof_generation_template_refactored.j2")
PROFILE_SINGLE_TASK_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "prof_single_task_template.j2")
# Path to generating CMakeLists.txt and running scripts

# Type definition
FrameworkType = Literal["torch", "mindspore", "numpy"]
ImplType = Literal["triton_cuda", "triton_ascend", "triton-russia", "swft",
                   "cuda_c", "cpp", "tilelang_npuir", "tilelang_cuda", "ascendc",
                   "ascendc_catlass", "torch"]
BackendType = Literal["cuda", "ascend", "cpu"]
ArchType = str

logger = logging.getLogger(__name__)


def _get_framework_sync_code(framework: FrameworkType,
                             backend: BackendType) -> str:
    """Return the synchronization call emitted into verifier scripts."""
    if framework == "torch":
        if backend == "cuda":
            return "torch.cuda.synchronize()"
        if backend == "ascend":
            return "torch.npu.synchronize()"
    if framework == "mindspore" and backend == "ascend":
        return "ms.runtime.synchronize()"
    return ""


def sync_artifacts_to_directory(artifacts: Dict[str, str], target_dir: str, task_id: str = "0") -> None:
    """
    Synchronises the artifices to the destination directory.

    Args:
        Artiffacts: returns an artiffacts dictionary from Worker in {relative_path: file_content}
                   For example: \"autotune_info_case_0.json,\" \"subdir/result.jsonl\": \"...\"
        target_dir: destination directory path (usually verify_dir)
        task_id: task ID (for log)
    """
    if not artifacts:
        return

    logger.info(f"[{task_id}] Syncing {len(artifacts)} artifact files to {target_dir}")

    target_root = os.path.realpath(target_dir)
    for rel_path, content in artifacts.items():
        if not isinstance(rel_path, str) or not rel_path:
            logger.warning(f"[{task_id}] Ignoring invalid artifact path: {rel_path!r}")
            continue
        full_path = os.path.realpath(os.path.join(target_root, rel_path))
        try:
            contained = os.path.commonpath(
                (target_root, full_path)) == target_root
        except ValueError:
            contained = False
        if not contained:
            logger.warning(
                f"[{task_id}] Ignoring artifact path outside verify dir: "
                f"{rel_path!r}")
            continue

        # Make sure the directory exists.
        dir_path = os.path.dirname(full_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"[{task_id}] Created directory: {dir_path}")

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.debug(f"[{task_id}] Synced artifact: {rel_path}")
        except Exception as e:
            logger.warning(f"[{task_id}] Failed to sync artifact {rel_path}: {e}")


class KernelVerifier:
    def __init__(self,
                 op_name: str,
                 framework_code: str,
                 task_id: str = "0",
                 framework: FrameworkType = "torch",
                 dsl: ImplType = "triton_cuda",
                 backend: BackendType = "cuda",
                 arch: ArchType = "a100",
                 impl_func_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 worker: Optional[WorkerInterface] = None,
                 bench_type: Literal["kernelbench", "sol", "cann"] = "kernelbench"):
        """
        InitializeKernelverifier.

        Args:
            op_name (str): operatorName
            framework_code (str): frameworkAchieve code(s)PyTorch,MindSporeorNumPy)
            log_dir (str): Debug Info Directory
            task_id (str, optional): TasksID, to generate a unique directory name
            framework (FrameworkType): In-depth learningframework, the optional value includes "torch", "mindspore", "numpy"
            dsl (ImplType): Type of achievement, optional values include "triton_cuda", "triton_ascend", "triton-russia", "swft"
            backend (BackendType): Calculatedevicebackend, the optional value includes "cuda", "ascend"
            arch (ArchType): Hardware architecture, by backend-specific Validation level determination
            impl_func_name (str, optional): Fulfilling function name, default toop_name_dsl_framework
            worker (WorkerInterface, optional): WorkerExample for validation missions
        """
        self.op_name = op_name
        self.framework_code = framework_code
        self.framework = framework
        # Normalize DSL (automatic conversion of Triton to Triton_cuda or Triton_ascend)
        self.dsl = normalize_dsl(dsl, backend)
        self.backend = backend.lower()
        self.arch = arch.lower()
        self.task_id = task_id
        self.bench_type = bench_type
        # Can not get message: %s %s
        if config:
            self.config = config
            self.log_dir = config.get("log_dir")
        else:
            raise ValueError("config is required for KernelVerifier")
        self.config["bench_type"] = bench_type
        # Arch is canonical config too: per-DSL adapters (e.g. catlass)
        # read it in prepare_config / special_setup. Stash here so adapter
        # hooks don't need a separate plumbing path.
        self.config["arch"] = self.arch

        # Cache the DSL adapter once: per-instance state set by
        # prepare_config (catlass stashes arch / catlass_root for the
        # subsequent get_special_setup_code call) would be lost across
        # re-instantiations.
        self.dsl_adapter = get_dsl_adapter(self.dsl)

        aux_files = self.config.get("framework_aux_files") or {}
        factory_names = self.config.get("framework_factory_names") or {}
        if not isinstance(aux_files, dict):
            raise TypeError(
                "It must be Dict [str, str. bytes], "
                f"Actually... {type(aux_files).__name__}",
            )
        if not isinstance(factory_names, dict):
            raise TypeError(
                "It must be Dict [str, Any],"
                f"Actually... {type(factory_names).__name__}",
            )
        self.framework_aux_files: Dict[str, Union[str, bytes]] = aux_files
        self.framework_factory_names: Dict[str, Any] = factory_names
        self.framework_module_name = (
            self.config.get("framework_module_name")
            or f"{self.op_name}_{self.framework}"
        )
        self.framework_filename = (
            self.config.get("framework_filename")
            or f"{self.framework_module_name}.py"
        )
        # impl_func_name: per-DSL convention. Caller-pinned wins; else the
        # adapter's ``impl_func_name_template`` (e.g. "ModelNew" for
        # class-style, "{op_name}_kernel" for AscendC). Default template
        # on the base class is "{op_name}_{dsl}_{framework}".
        self.impl_func_name = impl_func_name or self.dsl_adapter.impl_func_name_template.format(
            op_name=op_name, dsl=dsl, framework=framework,
        )

        # Validate backend /arch combination - single source config_utils.check_backend_arch.
        # There was a second hard-coded list of new cards that were easily omitted.
        from op_autoresearch.op.utils.config_utils import check_backend_arch
        check_backend_arch(self.backend, self.arch)

        # Save the example of the worker (available in runtime Dynamic Settings)
        self.worker = worker

    def _materialize_framework_bundle(self, target_dir: str,
                                      framework_code: str,
                                      target_filename: Optional[str] = None
                                      ) -> str:
        """Write the framework reference module and its sidecars to
        ``target_dir`` as a single unit.

        ``target_filename`` is the .py basename to land at. Sidecars in
        ``self.framework_aux_files`` are written verbatim by their declared
        task-relative names. Callers that want
        ``reference.py/reference.json`` should set the framework filename and
        import module to ``reference`` instead of relying on implicit sidecar
        renames.

        Returns the absolute path of the .py file written.
        """
        target_filename = target_filename or self.framework_filename

        py_path = os.path.join(target_dir, target_filename)
        try:
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(framework_code)
            logger.debug(f"[{self.op_name}] framework File written: {py_path}")
        except Exception as e:
            logger.error(f"[{self.op_name}] framework File writing failed: "
                         f"{py_path}, Error: {e}")
            raise

        if not self.framework_aux_files:
            return py_path

        for rel_name, content in self.framework_aux_files.items():
            if (
                os.path.isabs(rel_name)
                or rel_name.startswith("..")
                or ".." in rel_name.split(os.sep)
                or ".." in rel_name.split("/")
            ):
                logger.warning(
                    f"[{self.op_name}] Skip Illegal sidecar Path: {rel_name!r}"
                )
                continue
            aux_path = os.path.join(target_dir, rel_name)
            os.makedirs(os.path.dirname(aux_path) or target_dir, exist_ok=True)
            try:
                if isinstance(content, bytes):
                    with open(aux_path, "wb") as f:
                        f.write(content)
                else:
                    with open(aux_path, "w", encoding="utf-8") as f:
                        f.write(content)
                logger.debug(
                    f"[{self.op_name}] sidecar File written: {aux_path}"
                )
            except Exception as e:
                logger.error(
                    f"[{self.op_name}] sidecar File writing failed: {aux_path}, Error: {e}"
                )
                raise

        return py_path

    def check_task_desc_static(self, code: str) -> Tuple[bool, str]:
        """
        Static check whether the tag_desc code is in line with the code

        Args:
            code: tag_dec code string

        Returns:
            Tuple [bool, st]: (Whether passed, error message)
        """
        try:
            tree = ast.parse(code)

            has_model_class = False
            has_get_inputs = False
            has_get_init_inputs = False

            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == 'Model':
                    has_model_class = True
                elif isinstance(node, ast.FunctionDef):
                    if node.name == 'get_inputs':
                        has_get_inputs = True
                    elif node.name == 'get_init_inputs':
                        has_get_init_inputs = True

            missing = []
            if not has_model_class:
                missing.append("class Model")
            if not has_get_inputs:
                missing.append("function get_inputs")
            if not has_get_init_inputs:
                missing.append("function get_init_inputs")

            if missing:
                return False, f"Missing required components in task_desc: {', '.join(missing)}"

            return True, ""

        except SyntaxError as e:
            return False, f"Syntax error in task_desc: {e}"
        except Exception as e:
            return False, f"Static check failed: {e}"

    async def check_task_desc_runtime(self, task_desc: str, timeout: int = 60) -> Tuple[bool, str]:
        """
        runtime Checks whether the task_desc code is correctly executed

        Args:
            name_dec: name_dec
            Timeout: Timeout

        Returns:
            Tuple [bool, st]: (Whether passed, error message)
        """
        # 1. Creation of a provisional certification directory
        check_dir = os.path.join(os.path.expanduser(self.log_dir), f"{self.op_name}_check_desc_{self.task_id}")
        os.makedirs(check_dir, exist_ok=True)

        try:
            # Resolve device_id from worker's DevicePool (same as gen_verify_project)
            device_id = 0
            if self.worker:
                from op_autoresearch.core.worker.local_worker import LocalWorker
                if isinstance(self.worker, LocalWorker) and self.worker.device_pool:
                    device_id = self.worker.device_pool.device_list[0]

            # Writing task_dec to REF_FILE_DFAULT+Sync sidecar
            ref_file = self._materialize_framework_bundle(
                check_dir, task_desc, target_filename=REF_FILE_DEFAULT)

            # 3.Generate authentication scriptsverify_{op_name}.py
            if self.framework == "mindspore":
                verify_script_content = f"""
import mindspore as ms
import sys
import os

sys.path.append(os.getcwd())
os.environ['DEVICE_ID'] = str({device_id})

def run_check():
    print("Starting reference check...")
    try:
        try:
            from reference import Model, get_inputs, get_init_inputs
        except ImportError as e:
            print(f"Import failed: {{e}}")
            return False

        print("Successfully imported Model and helper functions.")

        ms.set_context(device_target="Ascend", device_id={device_id})
        print(f"Using device: Ascend:{device_id}")

        try:
            init_inputs = get_init_inputs()
            model = Model(*init_inputs)
        except Exception as e:
            print(f"Model instantiation failed: {{e}}")
            return False

        try:
            inputs = get_inputs()
        except Exception as e:
            print(f"get_inputs failed: {{e}}")
            return False

        try:
            output = model(*inputs)
            print("Forward pass successful.")
        except Exception as e:
            print(f"Forward pass failed: {{e}}")
            return False

        return True

    except Exception as e:
        print(f"Unexpected error: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_check()
    if success:
        print("REFERENCE_CHECK_SUCCESS")
        sys.exit(0)
    else:
        print("REFERENCE_CHECK_FAILED")
        sys.exit(1)
"""
            else:
                verify_script_content = f"""
import torch
import sys
import os

sys.path.append(os.getcwd())

def _deep_to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [_deep_to(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_to(x, device) for x in obj)
    return obj

def run_check():
    print("Starting reference check...")
    try:
        try:
            from reference import Model, get_inputs, get_init_inputs
        except ImportError as e:
            print(f"Import failed: {{e}}")
            return False

        print("Successfully imported Model and helper functions.")

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device({device_id})
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            device = "npu"
            torch.npu.set_device({device_id})

        print(f"Using device: {{device}}:{device_id}")

        try:
            init_inputs = get_init_inputs()
            model = Model(*init_inputs)
            if device != "cpu":
                model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Model instantiation failed: {{e}}")
            return False

        if device != "cpu":
            torch.set_default_device(device)
        try:
            inputs = get_inputs()
            if device != "cpu":
                inputs = _deep_to(inputs, device)
        except Exception as e:
            print(f"get_inputs failed: {{e}}")
            return False
        finally:
            if device != "cpu":
                torch.set_default_device("cpu")

        try:
            output = model(*inputs)
            print("Forward pass successful.")
        except Exception as e:
            print(f"Forward pass failed: {{e}}")
            return False

        return True

    except Exception as e:
        print(f"Unexpected error: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_check()
    if success:
        print("REFERENCE_CHECK_SUCCESS")
        sys.exit(0)
    else:
        print("REFERENCE_CHECK_FAILED")
        sys.exit(1)
"""
            verify_file = os.path.join(check_dir, f"verify_{self.op_name}.py")
            with open(verify_file, "w", encoding="utf-8") as f:
                f.write(verify_script_content)

            # Packing catalogues
            package_data = self._pack_directory(check_dir)

            # 5. Use Worker for execution
            if not self.worker:
                raise RuntimeError("Worker not set for runtime check")

            # Note: We don't need to be visible here because reference check usually does a simple forward pass
            # In the case of remote worker, it is automatically distributed; in the case of local worker, it usually does not require a specific device lock (unless OOM)
            # But for safety's sake, the caller should have taken care of it.

            success, log, _ = await self.worker.verify(package_data, f"{self.task_id}_check", self.op_name, timeout)

            if success and "REFERENCE_CHECK_SUCCESS" in log:
                return True, ""
            else:
                return False, f"Runtime check failed:\n{log}"

        except Exception as e:
            return False, f"Runtime check exception: {str(e)}"
        finally:
            # Clear Temporary Directory
            shutil.rmtree(check_dir, ignore_errors=True)

    async def generate_reference_data(
        self,
        task_desc: str,
        timeout: Optional[int] = None,
        save_inputs: bool = False,
        device_id: Optional[int] = None,
    ) -> Tuple[bool, str, bytes]:
        """
        Execute task_dec and generate reference data on GPU

        For the CUDA-to-Asend conversion scenario: execute the Triton-CUDA code on the GPU Worker.
        Saves the output as reference data for NPU Worker to verify the correctness of the converted code.

        Args:
            task_dec: tag_dec code string (Triton-CUDA code)
            Timeout: Timeout
            Save_inputs: Whether to save both inputs and init_inputs to reference data.
                         When True, the authentication end is completely de-dependent from the source platform (not requiring the effect vehicle work code).
            Data_id: Specifies the device ID (optional) that will be used to execute the reference data generation.

        Returns:
            Tuple [bol, st, bytes]: (successful, log, reference databytes)
            - When successful bytes is.pt file content
            Bytes is empty when it fails, b''
        """
        timeout = resolve_reference_timeout(timeout)
        # 1. Creation of temporary directories
        ref_dir = os.path.join(os.path.expanduser(self.log_dir), f"{self.op_name}_gen_ref_{self.task_id}")
        os.makedirs(ref_dir, exist_ok=True)

        try:
            # Writing task_dec to REF_FILE_DFAULT+Sync sidecar
            ref_file = self._materialize_framework_bundle(
                ref_dir, task_desc, target_filename=REF_FILE_DEFAULT)

            # 3. Generate reference data scripts
            save_inputs_flag = "True" if save_inputs else "False"
            backend_name = self.backend
            target_device_id = 0 if device_id is None or device_id < 0 else int(device_id)
            gen_ref_script = f'''
import torch
import sys
import os

sys.path.append(os.getcwd())

def _deep_to(obj, device):
    """Recursively move tensors to device, handling nested list/tuple."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [_deep_to(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_to(x, device) for x in obj)
    return obj

def _deep_clone(obj):
    """Recursively clone tensors, handling nested list/tuple."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, list):
        return [_deep_clone(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_clone(x) for x in obj)
    return obj

def _deep_cpu(obj):
    """Recursively move tensors to CPU, handling nested list/tuple."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, list):
        return [_deep_cpu(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_cpu(x) for x in obj)
    return obj

def generate_reference():
    print("Starting reference data generation...")
    save_inputs = {save_inputs_flag}
    try:
        try:
            from reference import Model, get_inputs, get_init_inputs
        except ImportError as e:
            print(f"Import failed: {{e}}")
            return False

        print("Successfully imported Model and helper functions.")

        backend = "{backend_name}"
        device_id = {target_device_id}
        device = "cpu"
        if backend == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            if not torch.cuda.is_available():
                print("CUDA backend requested but CUDA is not available")
                return False
            device = "cuda"
            torch.cuda.set_device(0)
        elif backend == "ascend":
            os.environ["DEVICE_ID"] = str(device_id)
            try:
                import torch_npu  # noqa: F401
            except ImportError as e:
                print(f"torch_npu import failed: {{e}}")
                return False
            if not hasattr(torch, "npu") or not torch.npu.is_available():
                print("Ascend backend requested but torch.npu is not available")
                return False
            device = "npu"
            torch.npu.set_device(device_id)
        elif backend == "cpu":
            device = "cpu"
        else:
            print(f"Unsupported backend for reference generation: {{backend}}")
            return False

        print(f"Using device: {{device}}")

        torch.manual_seed(0)
        print("[INFO] Random seed: 0")

        try:
            init_inputs = get_init_inputs()
            model = Model(*init_inputs)
            if device != "cpu":
                model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Model instantiation failed: {{e}}")
            return False

        if device != "cpu":
            torch.set_default_device(device)

        torch.manual_seed(0)
        try:
            inputs = get_inputs()
            if device != "cpu":
                inputs = _deep_to(inputs, device)
        except Exception as e:
            print(f"get_inputs failed: {{e}}")
            return False
        finally:
            if device != "cpu":
                torch.set_default_device("cpu")

        inputs_snapshot = None
        if save_inputs:
            inputs_snapshot = _deep_clone(inputs)

        try:
            with torch.no_grad():
                outputs = model(*inputs)
            print("Forward pass successful.")
        except Exception as e:
            print(f"Forward pass failed: {{e}}")
            return False

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        outputs_cpu = _deep_cpu(outputs)

        ref_data = {{
            'op_name': '{self.op_name}',
            'seed': 0,
            'outputs': outputs_cpu,
            'output_shapes': [x.shape if isinstance(x, torch.Tensor) else None for x in outputs_cpu],
            'output_dtypes': [str(x.dtype) if isinstance(x, torch.Tensor) else None for x in outputs_cpu],
        }}

        if save_inputs:
            inputs_cpu = _deep_cpu(inputs_snapshot)
            ref_data['save_inputs'] = True
            ref_data['inputs'] = inputs_cpu
            ref_data['input_shapes'] = [x.shape if isinstance(x, torch.Tensor) else None for x in inputs_cpu]
            ref_data['input_dtypes'] = [str(x.dtype) if isinstance(x, torch.Tensor) else None for x in inputs_cpu]
            ref_data['init_inputs'] = init_inputs
            print(f"[INFO] save_inputs=True, saving inputs ({{len(inputs_cpu)}}) and init_inputs ({{len(init_inputs)}})")

        ref_file = os.path.join(os.getcwd(), "{self.op_name}_reference.pt")
        torch.save(ref_data, ref_file)
        print(f"[INFO] Reference data saved to: {{ref_file}}")
        print(f"[INFO] Output count: {{len(outputs_cpu)}}")
        for i, out in enumerate(outputs_cpu):
            if isinstance(out, torch.Tensor):
                print(f"  Output[{{i}}]: shape={{out.shape}}, dtype={{out.dtype}}")

        return True

    except Exception as e:
        print(f"Unexpected error: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_reference()
    if success:
        print("REFERENCE_GENERATION_SUCCESS")
        sys.exit(0)
    else:
        print("REFERENCE_GENERATION_FAILED")
        sys.exit(1)
'''
            script_file = os.path.join(ref_dir, f"verify_{self.op_name}.py")
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(gen_ref_script)

            # Packing catalogues
            package_data = self._pack_directory(ref_dir)

            # 5. Use Worker.generate_reference for execution
            if not self.worker:
                raise RuntimeError("Worker not set for reference generation")

            # Directly call the starter'sgenerate_reference method
            # This method will execute scripts and return bytes of.pt files
            success, log, ref_bytes = await self.worker.generate_reference(
                package_data, f"{self.task_id}_gen_ref", self.op_name, timeout
            )

            if not success:
                return False, f"Reference generation failed:\n{log}", b''

            return True, log, ref_bytes

        except Exception as e:
            return False, f"Reference generation exception: {str(e)}", b''
        finally:
            # Clear Temporary Directory
            shutil.rmtree(ref_dir, ignore_errors=True)

    async def profile_single_task(self, task_desc: str,
                                  warmup_times: Optional[int] = None,
                                  run_times: Optional[int] = None,
                                  timeout: Optional[int] = None,
                                  device_id: int = 0) -> Dict[str, Any]:
        """
        Performance test for performing a single task (measuring only the performance of the task_desc without comparison of base vs generation)

        This function is used to measure the performance of a certain section of the code (which contains the Model class) separately, creating a directory and producing a profile script on a temporary basis.

        Args:
            name_dec: code string containing Model, get_inputs, get_init_inputs
            Warmup_times: number of preheats
            Run_times: Number of times actually running
            Timeout: Timeout
            Device_id: deviceID

        Returns:
            Dict[str, Any]: Paragraph containing time_us, access, log
        """
        warmup_times = resolve_warmup_times(warmup_times)
        run_times = resolve_run_times(run_times)
        timeout = resolve_eval_timeout(timeout)
        # 1. Creation of temporary directories
        profile_dir = os.path.join(os.path.expanduser(self.log_dir),
                                   f"{self.op_name}_profile_single_{self.task_id}")
        os.makedirs(profile_dir, exist_ok=True)

        try:
            # Drop code +sidecar to drop the disc (buddle internal decision. py name +
            # Sidecar changed name with stem.
            framework_file = self._materialize_framework_bundle(
                profile_dir, task_desc)

            # 3. Testing scripts using templates to generate performance
            script_file = os.path.join(profile_dir, f"profile_single_{self.op_name}.py")
            self.gen_profile_single_task_file(script_file, device_id, warmup_times, run_times)

            # Packing catalogues
            package_data = self._pack_directory(profile_dir)

            # 5. Execute using Worker.profile_single_task
            if not self.worker:
                raise RuntimeError("Worker not set for profile_single_task")

            profile_settings = {
                'warmup_times': warmup_times,
                'run_times': run_times,
                'timeout': timeout
            }

            result = await self.worker.profile_single_task(
                package_data, f"{self.task_id}_profile_single", self.op_name, profile_settings
            )

            return result

        except Exception as e:
            logger.error(f"[{self.op_name}] profile_single_task exception: {e}", exc_info=True)
            return {'time_us': float('inf'), 'success': False, 'log': f"Profile single task exception: {str(e)}"}
        finally:
            # Clear Temporary Directory
            shutil.rmtree(profile_dir, ignore_errors=True)

    def gen_profile_single_task_file(self, profile_file: str, device_id: int,
                                     warmup_times: int, run_times: int):
        """Use templates to generate single task performance to test scripts"""
        logger.info(f"[{self.op_name}] Start generating single task performance test files")

        # Load Template From File
        try:
            with open(PROFILE_SINGLE_TASK_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                template = Template(f.read())
            logger.debug(f"[{self.op_name}] Loaded single task performance test template successfully")
        except Exception as e:
            logger.error(f"[{self.op_name}] Template loading failed: {e}")
            raise

        # Test for dynamic Shape
        is_dynamic_shape = self._detect_dynamic_shape()

        # Getadaps
        try:
            framework_adapter = get_framework_adapter(self.framework)
            backend_adapter = get_backend_adapter(self.backend)
        except Exception as e:
            logger.error(f"[{self.op_name}] AdaptersInitialization failed: {e}")
            raise

        # Generate Snippets with an adapter
        try:
            framework_imports = framework_adapter.get_import_statements()

            # ``get_inputs`` / ``get_inputs_dyn_list`` will automatically use ``import ... as ...``
            framework_model_import = framework_adapter.get_framework_import(
                self.op_name,
                is_dynamic_shape,
                inputs_factory_name=self._resolve_dyn_factory(),
                module_name=self.framework_module_name,
            )
            logger.debug(
                f"[{self.op_name}] Framework model import Generate successfully "
                f"(Length: {len(framework_model_import)})"
            )

            # Generate device settings code
            backend_adapter.setup_environment(device_id, self.arch)
            device_setup_code = framework_adapter.get_device_setup_code(self.backend, self.arch, device_id)

            # Generate input processing code
            process_input_code = framework_adapter.get_process_input_code(self.backend, self.dsl)

            # Generate set_seed code
            set_seed_code = framework_adapter.get_set_seed_code(self.backend)

            # Fetch TensorType Name
            tensor_type_name = framework_adapter.get_tensor_type_name()

            # Generate benchmark code (measuring performance of famework_model using base mode)
            benchmark_code = self._generate_base_benchmark_code(framework_adapter, None,
                                                                warmup_times, run_times)
        except Exception as e:
            logger.error(f"[{self.op_name}] Snippet generation failed: {e}", exc_info=True)
            raise

        # Render Template
        try:
            rendered_code = template.render(
                op_name=self.op_name,
                framework=self.framework,
                backend=self.backend,
                arch=self.arch,
                device_id=device_id,
                is_dynamic_shape=is_dynamic_shape,
                warmup_times=warmup_times,
                run_times=run_times,
                framework_imports=self._prepare_code_lines(framework_imports),
                framework_model_import=self._prepare_code_lines(framework_model_import),
                device_setup_code=self._prepare_code_lines(device_setup_code),
                process_input_code=self._prepare_code_lines(process_input_code),
                set_seed_code=self._prepare_code_lines(set_seed_code),
                tensor_type_name=tensor_type_name,
                benchmark_code=self._prepare_code_lines(benchmark_code),
            )
            logger.info(f"[{self.op_name}] Template rendering succeeded")
        except Exception as e:
            logger.error(f"[{self.op_name}] Template render failed: {e}", exc_info=True)
            raise

        # Writing files
        try:
            with open(profile_file, "w", encoding="utf-8") as f:
                f.write(rendered_code)
            logger.info(f"[{self.op_name}] Script for single task performance test written: {profile_file}")
        except Exception as e:
            logger.error(f"[{self.op_name}] Script writing failed: {e}")
            raise

    def _verify_impl_artifacts_ready(self, verify_dir: str) -> bool:
        """Return True when generated verify/profile artifacts exist.

        bench_type variants (sol / cann) override the default ``framework
        file + <op>_<dsl>_impl.py`` shape. Per-DSL artifact shape (e.g.
        catlass needs kernel.py + CMakeLists.txt) is delegated to the
        adapter via ``expected_artifacts``."""
        # bench_type variants are not per-DSL — handle at this layer.
        impl_file = os.path.join(verify_dir, f"{self.op_name}_{self.dsl}_impl.py")
        if self.bench_type == "sol":
            return (os.path.isfile(os.path.join(verify_dir, "definition.json"))
                    and os.path.isfile(impl_file))
        if self.bench_type == "cann":
            return (os.path.isfile(os.path.join(verify_dir, "proto.yaml"))
                    and os.path.isfile(impl_file))
        artifacts = self.dsl_adapter.expected_artifacts(
            verify_dir, self.op_name, self.framework, self.bench_type, self.dsl,
        )
        return all(os.path.isfile(p) for p in artifacts)

    def _create_verify_dir(self, step_counter) -> str:
        """Create authentication directory and return directory path"""
        expanded_log_dir = os.path.expanduser(self.log_dir)
        unique_dir = f"Iteration{self.task_id}_Step{step_counter}_verify"

        target_dir = os.path.join(expanded_log_dir, self.op_name, unique_dir)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    # Accepted multi-shape factory names (any one triggers dynamic mode).
    # ``get_inputs_dyn_list`` — legacy (209 internal benchmark refs);
    # ``get_input_groups`` — NPUKernelBench + WA new convention. The
    # framework adapter aliases whichever the ref defines back to
    # ``get_inputs_dyn_list`` (the template's internal local name).
    _DYN_FACTORY_NAMES = ("get_inputs_dyn_list", "get_input_groups")

    def _resolve_dyn_factory(self) -> Optional[str]:
        """Name of the ref's multi-shape factory, or None for single-shape.
        Explicit ``framework_factory_names.inputs_factory`` wins; else AST-
        scan the ref source for one of :attr:`_DYN_FACTORY_NAMES`."""
        explicit = (self.framework_factory_names or {}).get("inputs_factory")
        if explicit:
            return explicit
        code = self.framework_code or ""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            for n in self._DYN_FACTORY_NAMES:
                if n in code:
                    return n
            return None
        for node in tree.body:
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in self._DYN_FACTORY_NAMES):
                return node.name
        return None

    def _detect_dynamic_shape(self) -> bool:
        """True iff the ref module exposes a multi-shape factory (auto-
        detected) or ``framework_factory_names.is_dynamic_shape=True``
        is explicitly declared."""
        declared = (self.framework_factory_names or {}).get("is_dynamic_shape")
        if isinstance(declared, bool):
            return declared
        return self._resolve_dyn_factory() is not None

    @staticmethod
    def _prepare_code_lines(code_snippet: Any) -> List[str]:
        """Regularizes multi-line Snippets into a line-by-line list to control indentation when templates are rendered."""
        if not code_snippet:
            return []
        if isinstance(code_snippet, (list, tuple)):
            lines: List[str] = []
            for snippet in code_snippet:
                lines.extend(KernelVerifier._prepare_code_lines(snippet))
            return lines
        if isinstance(code_snippet, str):
            normalized = textwrap.dedent(code_snippet).strip("\n")
            if not normalized:
                return []
            return normalized.split("\n")
        raise TypeError(f"Unsupported code snippet type: {type(code_snippet)}")

    def _get_data_cache_config(self):
        return load_verifier_data_cache_config(self.config)

    def _get_data_cache_key_id(self) -> str:
        return get_verifier_data_cache_key_id(self.config, self.task_id)

    def _get_reference_cache_key(self) -> str:
        return build_reference_cache_key(
            op_name=self.op_name,
            framework_code=self.framework_code,
            framework=self.framework,
            backend=self.backend,
            arch=self.arch,
            bench_type=self.bench_type,
            task_id=self._get_data_cache_key_id(),
        )

    def _get_baseline_cache_source(self) -> Optional[str]:
        if self.bench_type == "cann":
            try:
                cann_problem_dir = self.config.get("cann_problem_dir")
                if not cann_problem_dir:
                    logger.info(
                        f"[{self.op_name}] config['cann_problem_dir'] Unconfigured, Skip CANN baseline cache"
                    )
                    return None
                return cann_problem_dir
            except Exception as exc:
                logger.info(
                    f"[{self.op_name}] CANN baseline cache key Build failed, Skip: {exc}"
                )
                return None
        if self.bench_type != "sol":
            return self.framework_code
        try:
            sol_problem_dir = self.config.get("sol_problem_dir")
            if not sol_problem_dir:
                logger.info(
                    f"[{self.op_name}] config['sol_problem_dir'] Unconfigured, Skip SOL baseline cache"
                )
                return None
            return build_sol_problem_cache_identity(sol_problem_dir)
        except Exception as exc:
            logger.info(
                f"[{self.op_name}] SOL baseline cache key Build failed, Skip baseline cache: {exc}"
            )
            return None

    def _get_baseline_cache_key(self, warmup_times: int, run_times: int) -> Optional[str]:
        cache_source = self._get_baseline_cache_source()
        if cache_source is None:
            return None
        return build_baseline_cache_key(
            op_name=self.op_name,
            framework_code=cache_source,
            framework=self.framework,
            backend=self.backend,
            arch=self.arch,
            bench_type=self.bench_type,
            warmup_times=warmup_times,
            run_times=run_times,
            dsl=self.dsl,
            task_id=self._get_data_cache_key_id(),
        )

    def _load_cached_torch_reference_payload(self, reference_data: bytes) -> Optional[Any]:
        try:
            import torch

            return torch.load(io.BytesIO(reference_data), map_location="cpu", weights_only=True)
        except TypeError as exc:
            if "weights_only" in str(exc):
                logger.warning(
                    f"[{self.op_name}] Current PyTorch Not supported weights_only=True,"
                    "Disable this reference data case to avoid unsafe backsequencing"
                )
            else:
                logger.warning(
                    f"[{self.op_name}] Verifier Data Cache reference data Unable to parse, ready to regenerate.: {exc}"
                )
            return None
        except Exception as exc:
            logger.warning(
                f"[{self.op_name}] Verifier Data Cache reference data Unable to parse, ready to regenerate.: {exc}"
            )
            return None

    def _is_valid_cached_reference_data(self, reference_data: bytes) -> bool:
        if not reference_data:
            return False
        if self.framework != "torch":
            return True
        payload = self._load_cached_torch_reference_payload(reference_data)
        if payload is None:
            return False

        if not isinstance(payload, dict):
            logger.warning(f"[{self.op_name}] Verifier Data Cache reference data Invalid format, ready for regeneration")
            return False
        if "outputs" not in payload:
            logger.warning(f"[{self.op_name}] Verifier Data Cache reference data Missing outputsReady to regenerate.")
            return False
        if not payload.get("save_inputs") or payload.get("inputs") is None:
            logger.warning(
                f"[{self.op_name}] Verifier Data Cache reference data Missing Reusable inputsReady to regenerate."
            )
            return False
        return True

    def _clear_managed_reference_data(self, reason: str = "") -> None:
        if not self.config.get("_data_cache_reference_key"):
            return
        self.config.pop("reference_data", None)
        self.config.pop("use_reference_data", None)
        self.config.pop("use_reference_inputs", None)
        self.config.pop("_data_cache_reference_key", None)
        if reason:
            logger.info(f"[{self.op_name}] Clear Local Data Cache Injecting. reference data: {reason}")

    async def _prepare_cached_reference_data(self, device_id: int) -> Optional[bytes]:
        if self.bench_type != "kernelbench":
            self._clear_managed_reference_data("Bench_ type does not support current reference data size")
            return None

        if self._detect_dynamic_shape():
            self._clear_managed_reference_data("Dynamic Shape Do not repeat static reference data")
            logger.info(f"[{self.op_name}] Dynamics detected shapeSkip reference data cache")
            return None

        cache_cfg = self._get_data_cache_config()
        cache_key = self._get_reference_cache_key()
        cache_file = get_reference_cache_file_path(
            cache_cfg,
            op_name=self.op_name,
            cache_key=cache_key,
        )

        if self.config.get("use_reference_data") and self.config.get("reference_data"):
            managed_key = self.config.get("_data_cache_reference_key")
            if managed_key:
                if not cache_cfg.enabled or not cache_cfg.cache_reference_data:
                    self._clear_managed_reference_data("Data_cache Closed")
                elif managed_key == cache_key:
                    logger.info(f"[{self.op_name}] Reuse Current verifier Injected reference data")
                    return None
                else:
                    self._clear_managed_reference_data("Cache key changed")
            else:
                logger.info(
                    f"[{self.op_name}] By Caller reference_dataSkip Local Data Cache Question"
                )
                return None

        if not cache_cfg.enabled or not cache_cfg.cache_reference_data:
            return None

        try:
            async with verifier_data_cache_lock(
                cache_cfg,
                namespace="reference",
                op_name=self.op_name,
                cache_key=cache_key,
            ):
                cached_reference = read_reference_data_from_cache(
                    cache_cfg,
                    op_name=self.op_name,
                    cache_key=cache_key,
                )
                if cached_reference:
                    if not self._is_valid_cached_reference_data(cached_reference):
                        delete_reference_data_from_cache(
                            cache_cfg,
                            op_name=self.op_name,
                            cache_key=cache_key,
                        )
                    else:
                        logger.info(
                            f"[{self.op_name}] Verifier Data Cache Hit:reference data, "
                            f"cache_file={cache_file}, cache_key={cache_key}"
                        )
                        return cached_reference

                if cached_reference:
                    logger.info(
                        f"[{self.op_name}] Verifier Data Cache reference data Expired, regenerated: "
                        f"cache_file={cache_file}, cache_key={cache_key}"
                    )
                else:
                    logger.info(
                        f"[{self.op_name}] Verifier Data Cache Uncut:reference data, "
                        f"cache_file={cache_file}, cache_key={cache_key}"
                    )

                if not self.worker:
                    logger.info(f"[{self.op_name}] Current None workerSkip reference data Backfill")
                    return None

                logger.info(f"[{self.op_name}] Start Generating reference data")
                reference_timeout = resolve_reference_timeout(
                    self.config.get("reference_data_timeout",
                                    self.config.get("verify_timeout"))
                )
                try:
                    success, log, reference_bytes = await self.generate_reference_data(
                        self.framework_code,
                        timeout=reference_timeout,
                        save_inputs=True,
                        device_id=device_id,
                    )
                except Exception as exc:
                    logger.warning(f"[{self.op_name}] Generate reference data Failed, back to real-time validation: {exc}")
                    return None

                if not success or not reference_bytes:
                    logger.warning(
                        f"[{self.op_name}] reference data Generate failed, back to real-time validation: {(log or '')[:500]}"
                    )
                    return None

                written_path = write_reference_data_to_cache(
                    cache_cfg,
                    op_name=self.op_name,
                    cache_key=cache_key,
                    reference_data=reference_bytes,
                    metadata={
                        "framework": self.framework,
                        "task_id": self.task_id,
                        "cache_key_id": self._get_data_cache_key_id(),
                        "backend": self.backend,
                        "arch": self.arch,
                        "bench_type": self.bench_type,
                        "save_inputs": True,
                    },
                )
                if written_path:
                    logger.info(
                        f"[{self.op_name}] reference data Written Verifier Data Cache: "
                        f"cache_file={written_path}, cache_key={cache_key}, "
                        f"cache_dir={cache_cfg.cache_dir}"
                    )
                return reference_bytes
        except TimeoutError as exc:
            logger.warning(
                f"[{self.op_name}] Access Verifier Data Cache reference lock Timeout, back to real time validation.: {exc}"
            )
            return None

    def _apply_cached_reference_data(self, reference_data: bytes, cache_key: Optional[str] = None) -> None:
        if not reference_data:
            return
        self.config["use_reference_data"] = True
        self.config["use_reference_inputs"] = True
        self.config["reference_data"] = reference_data
        self.config["_data_cache_reference_key"] = cache_key or self._get_reference_cache_key()

    def _get_cached_baseline_time_us(self, warmup_times: int, run_times: int) -> Optional[float]:
        if self.bench_type not in {"kernelbench", "sol", "cann"}:
            return None

        cache_cfg = self._get_data_cache_config()
        if not cache_cfg.enabled or not cache_cfg.cache_baseline_result:
            return None

        cache_key = self._get_baseline_cache_key(warmup_times, run_times)
        if not cache_key:
            return None
        cache_file = get_baseline_cache_file_path(
            cache_cfg,
            op_name=self.op_name,
            cache_key=cache_key,
        )
        cache_entry = read_baseline_result_from_cache(
            cache_cfg,
            op_name=self.op_name,
            cache_key=cache_key,
        )
        baseline_time_us = extract_baseline_time_us(cache_entry)
        if baseline_time_us is not None:
            logger.info(
                f"[{self.op_name}] Verifier Data Cache Hit:baseline={baseline_time_us:.2f} us, "
                f"cache_file={cache_file}, cache_key={cache_key}"
            )
        elif cache_entry:
            logger.warning(
                f"[{self.op_name}] Verifier Data Cache baseline Failed to delete old cache: "
                f"cache_file={cache_file}, cache_key={cache_key}"
            )
            delete_baseline_result_from_cache(
                cache_cfg,
                op_name=self.op_name,
                cache_key=cache_key,
            )
        return baseline_time_us

    def _store_baseline_result_in_data_cache(
        self,
        *,
        base_time_us: Optional[float],
        warmup_times: int,
        run_times: int,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        if self.bench_type not in {"kernelbench", "sol", "cann"} or base_time_us is None:
            return
        if base_time_us <= 0 or base_time_us >= float("inf"):
            return

        cache_cfg = self._get_data_cache_config()
        if not cache_cfg.enabled or not cache_cfg.cache_baseline_result:
            return
        cache_key = self._get_baseline_cache_key(warmup_times, run_times)
        if not cache_key:
            return

        payload: Dict[str, Any]
        raw_json = (artifacts or {}).get("base_profile_result.json")
        if raw_json:
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                payload = build_baseline_cache_payload(
                    base_time_us=base_time_us,
                    warmup_times=warmup_times,
                    run_times=run_times,
                )
        else:
            payload = build_baseline_cache_payload(
                base_time_us=base_time_us,
                warmup_times=warmup_times,
                run_times=run_times,
            )

        written_path = write_baseline_result_to_cache(
            cache_cfg,
            op_name=self.op_name,
            cache_key=cache_key,
            result_data=payload,
            metadata={
                "framework": self.framework,
                "task_id": self.task_id,
                "cache_key_id": self._get_data_cache_key_id(),
                "dsl": self.dsl,
                "backend": self.backend,
                "arch": self.arch,
                "bench_type": self.bench_type,
            },
        )
        if written_path:
            logger.info(
                f"[{self.op_name}] baseline Results written Verifier Data Cache: "
                f"cache_file={written_path}, cache_key={cache_key}, "
                f"cache_dir={cache_cfg.cache_dir}"
            )

    def gen_verify_project(self, impl_code: str, verify_dir: str, device_id: int = 0):
        """Generate authentication project files to a specified directory"""
        if self.bench_type == "sol":
            from op_autoresearch.op.verifier.sol_verifier import generate_sol_verify_project
            return generate_sol_verify_project(self, impl_code, verify_dir, device_id)
        if self.bench_type == "cann":
            from op_autoresearch.op.cann_correctness import generate_cann_verify_project
            return generate_cann_verify_project(self, impl_code, verify_dir, device_id)

        logger.info(f"[{self.op_name}] Start generating validation items, directories: {verify_dir}, device_id={device_id}")

        # ==========Processingreference dataMode==========
        use_reference_data = self.config.get('use_reference_data', False)
        reference_file = None

        if use_reference_data:
            reference_data_bytes = self.config.get('reference_data')
            if reference_data_bytes:
                # Write reference data to the authentication directory
                # Note: Use relative path (file name only) in RemoteWorker
                # When scripts are packaged and sent to a remote server, you can find the reference data file correctly from the current working directory
                reference_file_name = f"{self.op_name}_reference.pt"
                reference_file_abs = os.path.join(verify_dir, reference_file_name)
                try:
                    with open(reference_file_abs, 'wb') as f:
                        f.write(reference_data_bytes)
                    logger.info(
                        f"[{self.op_name}] reference dataWritten: "
                        f"{reference_file_abs} ({len(reference_data_bytes)} bytes)"
                    )
                    # A relative path (file name only) is passed to the template, and the script is searched from cwd when executing
                    reference_file = reference_file_name
                except Exception as e:
                    logger.error(f"[{self.op_name}] reference dataWriting Failed: {e}")
                    use_reference_data = False
                    reference_file = None
            else:
                logger.warning(f"[{self.op_name}] use_reference_data=True But he didn't. reference_data")
                use_reference_data = False

        # Use_reference_inputs relies on use_reference_data and requests.pt to include inputs
        use_reference_inputs = self.config.get('use_reference_inputs', False) and use_reference_data

        # framework code +sidecar sets the disc together (bundle internal decision. py name +
        # Sidecar changed name with stem.
        framework_file = self._materialize_framework_bundle(
            verify_dir, self.framework_code)

        # Write realization file: each DSL's own adapter decides schema (default)
        # ``<op>_<dsl>_impl.py``, catlass write kernel.py + Torture catlass_op tree,
        # Ascendc render CMakeLists + Write Tiling/kernel/pybind11 three cpps.
        self.dsl_adapter.materialize_impl(
            impl_code=impl_code,
            verify_dir=verify_dir,
            op_name=self.op_name,
            framework=self.framework,
            dsl_name=self.dsl,
            task_info=None,
            config=self.config,
        )

        # cannbench precision path: stage the package's MERE/MARE comparator into
        # verify_dir as cann_correctness.py (the name the generated script imports).
        if getattr(self.dsl_adapter, "uses_cannbench_precision", False):
            from op_autoresearch.op.cann_correctness import CORE_PY_PATH
            shutil.copy2(CORE_PY_PATH, os.path.join(verify_dir, "cann_correctness.py"))

        # Generate authentication scripts
        verify_file = os.path.join(verify_dir, f"verify_{self.op_name}.py")

        # Load Template From File
        logger.info(f"[{self.op_name}] Start generating validation items, using templates: {os.path.basename(TEMPLATE_PATH)}")
        try:
            with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
                template = Template(f.read())
            logger.debug(f"[{self.op_name}] Template file loaded successfully: {TEMPLATE_PATH}")
        except Exception as e:
            logger.error(f"[{self.op_name}] Failed to load template file: {TEMPLATE_PATH}, Error: {e}")
            raise

        # Test for dynamic Shape
        is_dynamic_shape = self._detect_dynamic_shape()
        logger.info(f"[{self.op_name}] DetectedshapeType: {'Dynamics' if is_dynamic_shape else 'Static'}")

        # Getadaps
        logger.debug(f"[{self.op_name}] Initializeadapters: framework={self.framework}, dsl={self.dsl}, backend={self.backend}")
        try:
            framework_adapter = get_framework_adapter(self.framework)
            dsl_adapter = get_dsl_adapter(self.dsl)
            backend_adapter = get_backend_adapter(self.backend)
            logger.debug(f"[{self.op_name}] AdaptersInitialization succeeded")
        except Exception as e:
            logger.error(f"[{self.op_name}] AdaptersInitialization failed: {e}")
            raise

        # Generate a code string using an adapter
        logger.debug(f"[{self.op_name}] Start generating Snippets...")
        try:
            framework_imports = framework_adapter.get_import_statements()
            logger.debug(f"[{self.op_name}] Framework importsGenerate successfully (Length: {len(framework_imports)})")

            framework_model_import = framework_adapter.get_framework_import(
                self.op_name,
                is_dynamic_shape,
                inputs_factory_name=self._resolve_dyn_factory(),
                module_name=self.framework_module_name,
            )
            logger.debug(f"[{self.op_name}] Framework model importGenerate successfully (Length: {len(framework_model_import)})")

            dsl_imports = dsl_adapter.get_import_statements(self.framework)
            logger.debug(f"[{self.op_name}] DSL importsGenerate successfully (Length: {len(dsl_imports)})")
            # get_runtime_env_override_code defaults to "" on the ABC;
            # only the pypto adapter emits a non-empty body. No dsl check.
            dsl_imports += dsl_adapter.get_runtime_env_override_code(
                pypto_run_mode=self.config.get("pypto_run_mode"),
                pypto_runtime_debug_mode=0,
            )

            dsl_impl_import = dsl_adapter.get_impl_import(self.op_name, self.impl_func_name)
            logger.debug(f"[{self.op_name}] DSL impl importGenerate successfully (Length: {len(dsl_impl_import)})")

            dsl_adapter.prepare_config(self.config, task_info=None)
            special_setup_code = dsl_adapter.get_special_setup_code(framework=self.framework)
            logger.debug(f"[{self.op_name}] Special setup codeGenerate successfully (Length: {len(special_setup_code)})")

            # Generate device settings code
            backend_adapter.setup_environment(device_id, self.arch)
            logger.debug(f"[{self.op_name}] BackendEnvironment Settings Completed: device_id={device_id}, arch={self.arch}")

            device_setup_code = framework_adapter.get_device_setup_code(self.backend, self.arch, device_id)
            logger.debug(f"[{self.op_name}] Device setup codeGenerate successfully (Length: {len(device_setup_code)})")

            # Generate input processing code
            process_input_code = framework_adapter.get_process_input_code(self.backend, self.dsl)
            logger.debug(f"[{self.op_name}] Process input codeGenerate successfully (Length: {len(process_input_code)})")

            # Generate code for creating impl_mode (DSL for ModelNew-type format)
            create_impl_code = dsl_adapter.create_impl_module(self.framework, framework_adapter)
            logger.debug(f"[{self.op_name}] Create impl module codeGenerate successfully (Length: {len(create_impl_code)})")

            # Generate call realization code
            call_impl_code = dsl_adapter.call_impl(
                self.impl_func_name, "inputs_for_impl", device_id,
                framework_adapter, self.op_name, "data_dir", "framework_output"
            )
            logger.debug(f"[{self.op_name}] Call impl codeGenerate successfully (Length: {len(call_impl_code)})")

            # Generate set_seed code
            set_seed_code = framework_adapter.get_set_seed_code(self.backend)
            logger.debug(f"[{self.op_name}] Set seed codeGenerate successfully (Length: {len(set_seed_code)})")

            # Generate binary I/O functions (if required)
            binary_io_functions = ""
            needs_binary_io = dsl_adapter.needs_binary_io
            if needs_binary_io:
                binary_io_functions = framework_adapter.get_binary_io_functions(self.op_name)
                logger.info(f"[{self.op_name}] Binary I/OFunction Generation Success (Length: {len(binary_io_functions)})")
            else:
                logger.debug(f"[{self.op_name}] I don't need it.Binary I/OFunctions")

            # Fetch TensorType Name (full path)
            tensor_type_name = framework_adapter.get_tensor_type_name()
            logger.debug(f"[{self.op_name}] TensorTypeName: {tensor_type_name}")

            # Generate compare function code (generated by FrameworkAdapter, using framework primary)
            compare_code = framework_adapter.get_compare_code()
            logger.debug(f"[{self.op_name}] Compare codeGenerate successfully (Length: {len(compare_code)})")

            # cannbench precision path (uses_cannbench_precision boolean): pull the
            # reference-call + compare snippets from the cann_correctness package.
            # Otherwise run the base model and the framework's generic compare.
            if getattr(self.dsl_adapter, "uses_cannbench_precision", False) \
                    and self.framework == "torch":
                from op_autoresearch.op import cann_correctness as _cann
                reference_call_code = _cann.reference_call_snippet()
                compare_outputs_code = _cann.compare_snippet()
            else:
                reference_call_code = "framework_output = framework_model(*inputs_for_framework)"
                compare_outputs_code = framework_adapter.get_compare_outputs_code()
            logger.debug(f"[{self.op_name}] Compare outputs codeGenerate successfully (Length: {len(compare_outputs_code)})")

            reference_sync_code = _get_framework_sync_code(
                self.framework, self.backend)
        except Exception as e:
            logger.error(f"[{self.op_name}] Snippet generation failed: {e}", exc_info=True)
            raise

        # Use template variables
        logger.debug(f"[{self.op_name}] Start rendering templates...")
        try:
            rendered_code = template.render(
                op_name=self.op_name,
                framework=self.framework,
                dsl=self.dsl,
                device_id=device_id,
                impl_func_name=self.impl_func_name,
                backend=self.backend,
                arch=self.arch,
                is_dynamic_shape=is_dynamic_shape,
                timeout=resolve_eval_timeout(self.config.get('verify_timeout')),
                # reference data mode (for conversion across backend)
                use_reference_data=use_reference_data,
                use_reference_inputs=use_reference_inputs,
                reference_file=reference_file,
                # Codes generated by Adapter
                framework_imports=self._prepare_code_lines(framework_imports),
                framework_model_import=self._prepare_code_lines(framework_model_import),
                dsl_imports=self._prepare_code_lines(dsl_imports),
                dsl_impl_import=self._prepare_code_lines(dsl_impl_import),
                special_setup_code=self._prepare_code_lines(special_setup_code),
                device_setup_code=self._prepare_code_lines(device_setup_code),
                process_input_code=self._prepare_code_lines(process_input_code),
                create_impl_code=self._prepare_code_lines(create_impl_code),
                call_impl_code=self._prepare_code_lines(call_impl_code),
                set_seed_code=self._prepare_code_lines(set_seed_code),
                reference_sync_code=self._prepare_code_lines(reference_sync_code),
                binary_io_functions=self._prepare_code_lines(binary_io_functions),
                needs_binary_io=needs_binary_io,
                tensor_type_name=tensor_type_name,
                compare_code=self._prepare_code_lines(compare_code),
                compare_outputs_code=self._prepare_code_lines(compare_outputs_code),
                reference_call_code=self._prepare_code_lines(reference_call_code),
            )
            logger.info(f"[{self.op_name}] Template rendering successful, code length after rendering: {len(rendered_code)} Character")
        except Exception as e:
            logger.error(f"[{self.op_name}] Template render failed: {e}", exc_info=True)
            raise

        # Writing files
        try:
            with open(verify_file, "w", encoding="utf-8") as f:
                f.write(rendered_code)
            logger.info(f"[{self.op_name}] Authentication script written: {verify_file}")
        except Exception as e:
            logger.error(f"[{self.op_name}] Authentication of script writing failed: {verify_file}, Error: {e}")
            raise

    def _pack_directory(self, dir_path: str) -> bytes:
        """Pack directory as tar bytes"""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w') as tar_file:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, dir_path)
                    tar_file.add(file_path, arcname=arcname)
        return tar_buffer.getvalue()

    async def run_verify(self, verify_dir: str,
                         timeout: Optional[int] = None,
                         device_id: int = 0):
        """
        Run validation scripts

        Note: device management (acquire/release) is the responsibility of the caller (verifier.run())
        This method is only responsible for implementing the scripts that have been produced.

        Args:
            Verify_dir: Authentication Directory
            Timeout: Timeout (seconds), default 5 minutes (transmit to template for each calculation)
            Data_id: deviceID (for logs and compatibility only, actual device is set in scripts)
        """
        timeout = resolve_eval_timeout(timeout)
        verify_script = os.path.join(verify_dir, f"verify_{self.op_name}.py")
        logger.info(f"[{self.op_name}] Prepare to run validation scripts: {verify_script}, timeout={timeout}sec")

        try:
            # Call Worker for authentication
            if not self.worker:
                # Check if device_id is -1 (means RemoteWorker, device managed by remote server)
                if device_id == -1:
                    raise RuntimeError(
                        f"[{self.op_name}] Worker not set and device_id=-1 (RemoteWorker mode). "
                        "Worker must be provided by Task or WorkerManager for RemoteWorker."
                    )
                # If there is no workingr, create Localworker (for testing scenario) based on device_id
                import warnings
                warnings.warn(
                    f"⚠️  [DEPRECATED] KernelVerifier Automatically Create LocalWorker It's the old bottom logic, only for testing.\n"
                    f"Suggested new formulation:\n"
                    f"  1. Register before calling Worker to WorkerManager(line code):\n"
                    f"     from op_autoresearch.core.worker.manager import register_local_worker\n"
                    f"     \n"
                    f"     await register_local_worker([{device_id}], backend='{self.backend}', arch='{self.arch}')\n"
                    f"  2. Task It's automatic from WorkerManager Access worker\n"
                    f"Example:examples/run_torch_npu_triton_single.py",
                    DeprecationWarning,
                    stacklevel=2
                )
                logger.warning(f"⚠️  [{self.op_name}] Worker not set, creating temporary LocalWorker (deprecated)")

                from op_autoresearch.core.worker.local_worker import LocalWorker
                from op_autoresearch.core.async_pool.device_pool import DevicePool
                logger.info(f"[{self.op_name}] Worker not set, creating LocalWorker with device [{device_id}]")
                device_pool = DevicePool([device_id])
                self.worker = LocalWorker(device_pool=device_pool, backend=self.backend)

            from op_autoresearch.core.worker.local_worker import LocalWorker
            if isinstance(self.worker, LocalWorker):
                if not hasattr(self.worker, 'device_pool') or self.worker.device_pool is None:
                    raise RuntimeError(
                        f"[{self.op_name}] LocalWorker must have device_pool. "
                        "This should be provided by Task when creating _private_worker."
                    )
                package_data = verify_dir
            else:
                logger.info(f"[{self.op_name}] Packing verify project")
                package_data = self._pack_directory(verify_dir)

            # Worker.verefy() just execute scripts without managing data
            # Device has set it up when it's producing the script.
            # The timeout we give to the worker needs to be a little bigger because the script has precise timeout controls inside.
            # The timeout here is to prevent the whole process from being hung by the death lock of the script.
            worker_timeout = timeout + 30
            logger.info(f"[{self.op_name}] Dispatching verify project to worker")
            success, log, artifacts = await self.worker.verify(package_data, self.task_id, self.op_name, worker_timeout)
            logger.info(f"[{self.op_name}] Worker verify returned")

            # Sync Artifacts to Verify_dir (for RemoteWorker)
            if artifacts:
                sync_artifacts_to_directory(artifacts, verify_dir, self.task_id)

            if success:
                logger.info(f"[{self.op_name}] Validate successful execution")
            else:
                # Full log is returned and written to the fail report; don't dump
                # it here (CANN toolchain warnings alone can be 100+ noise lines).
                logger.error(f"[{self.op_name}] Validation of execution failed (full log view) fail report)")
            return success, log

        except Exception as e:
            logger.error(f"[{self.op_name}] Validation of performance anomaly: {e}", exc_info=True)
            return False, str(e)

    def gen_profile_project(self, verify_dir: str, device_id: int = 0,
                            warmup_times: Optional[int] = None,
                            run_times: Optional[int] = None,
                            skip_base: bool = False):
        """Generate project files to a specified directory

        Args:
            Verify_dir: Authentication Directory
            Device_id: deviceID
            Warmup_times: number of preheats
            Run_times: Run number of times
            sskip_base: Skip base profile (rue under backend)
        """
        warmup_times = resolve_warmup_times(warmup_times)
        run_times = resolve_run_times(run_times)
        if self.bench_type == "sol":
            from op_autoresearch.op.verifier.sol_verifier import generate_sol_profile_project
            return generate_sol_profile_project(
                self, verify_dir, device_id, warmup_times, run_times, skip_base
            )
        if self.bench_type == "cann":
            from op_autoresearch.op.cann_correctness import generate_cann_profile_project
            return generate_cann_profile_project(
                self, verify_dir, device_id, warmup_times, run_times, skip_base
            )

        profile_generation_enabled = getattr(
            self, "_profile_generation_enabled", True)

        # Generate a baseline performance test script (if not skip)
        if not skip_base:
            profile_file = os.path.join(verify_dir, f"profile_{self.op_name}_base.py")
            self.gen_profile_file_from_template(PROFILE_BASE_TEMPLATE_PATH, profile_file,
                                                device_id, warmup_times, run_times)
        else:
            logger.info(f"[{self.op_name}] Skip base profile Generate (using caches) baseline Or cross.backendThe scene)")

        # Generate performance testing scripts
        if profile_generation_enabled:
            profile_file = os.path.join(verify_dir, f"profile_{self.op_name}_generation.py")
            self.gen_profile_file_from_template(PROFILE_GENERATION_TEMPLATE_PATH,
                                                profile_file, device_id, warmup_times, run_times)
        else:
            logger.info(f"[{self.op_name}] Skip generation profile Generate (Previous round) verify Not adopted)")

    def gen_profile_file_from_template(self, template_path: str, profile_file: str, device_id: int, warmup_times: int, run_times: int):
        """Generate profile files from templates"""
        template_name = os.path.basename(template_path)
        logger.info(f"[{self.op_name}] Start generating performance test files, using templates: {template_name}")

        # Load Template From File
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template = Template(f.read())
            logger.debug(f"[{self.op_name}] Capability test template file loaded successfully: {template_path}")
        except Exception as e:
            logger.error(f"[{self.op_name}] Failed to load performance test template file: {template_path}, Error: {e}")
            raise

        # Test for dynamic Shape
        is_dynamic_shape = self._detect_dynamic_shape()
        logger.debug(f"[{self.op_name}] Performance testshapeType: {'Dynamics' if is_dynamic_shape else 'Static'}")

        is_base_template = "base" in template_path.lower()
        logger.debug(f"[{self.op_name}] Performance test template type: {'base' if is_base_template else 'generation'}")

        # Getadaps
        try:
            framework_adapter = get_framework_adapter(self.framework)
            dsl_adapter = get_dsl_adapter(self.dsl)
            backend_adapter = get_backend_adapter(self.backend)
            logger.debug(f"[{self.op_name}] Performance testAdaptersInitialization succeeded")
        except Exception as e:
            logger.error(f"[{self.op_name}] Performance testAdaptersInitialization failed: {e}")
            raise

        # Generate a code string using an adapter
        logger.debug(f"[{self.op_name}] Start generating performance test Snippets...")
        try:
            framework_imports = framework_adapter.get_import_statements()
            framework_model_import = framework_adapter.get_framework_import(
                self.op_name,
                is_dynamic_shape,
                inputs_factory_name=self._resolve_dyn_factory(),
                module_name=self.framework_module_name,
            )

            # Generate device settings code
            backend_adapter.setup_environment(device_id, self.arch)
            device_setup_code = framework_adapter.get_device_setup_code(self.backend, self.arch, device_id)

            # Generate input processing code
            process_input_code = framework_adapter.get_process_input_code(self.backend, self.dsl)

            # Generate set_seed code
            set_seed_code = framework_adapter.get_set_seed_code(self.backend)

            # Base profile must stay framework-only. A broken generated DSL
            # project should not prevent measuring the reference baseline.
            dsl_imports = ""
            dsl_impl_import = ""
            special_setup_code = ""
            create_impl_code = ""
            binary_io_functions = ""
            needs_binary_io = False

            # Fetch TensorType Name (full path)
            tensor_type_name = framework_adapter.get_tensor_type_name()

            # Generate benchmark code
            if is_base_template:
                # Base template: benchmark vehicle model
                benchmark_code = self._generate_base_benchmark_code(
                    framework_adapter, dsl_adapter,
                    warmup_times, run_times,
                    clear_l2_cache=dsl_adapter.benchmark_requires_l2_clear,
                )
                logger.debug(f"[{self.op_name}] Base benchmarkCode Generation Success (Length: {len(benchmark_code)})")
            else:
                dsl_imports = dsl_adapter.get_import_statements(self.framework)
                # get_runtime_env_override_code defaults to "" on the ABC;
                # only the pypto adapter emits a non-empty body. No dsl check.
                dsl_imports += dsl_adapter.get_runtime_env_override_code(
                    pypto_run_mode=self.config.get("pypto_run_mode"),
                    pypto_runtime_debug_mode=1,
                )
                dsl_impl_import = dsl_adapter.get_impl_import(
                    self.op_name, self.impl_func_name)
                dsl_adapter.prepare_config(self.config, task_info=None)
                special_setup_code = dsl_adapter.get_special_setup_code(
                    framework=self.framework)

                # Generate code for creating impl_mode (DSL for ModelNew-type format)
                create_impl_code = dsl_adapter.create_impl_module(
                    self.framework, framework_adapter)
                logger.debug(
                    f"[{self.op_name}] Performance testCreate impl module codeGenerate successfully "
                    f"(Length: {len(create_impl_code)})")

                # Generate binary I/O functions (if required)
                needs_binary_io = dsl_adapter.needs_binary_io
                if needs_binary_io:
                    binary_io_functions = framework_adapter.get_binary_io_functions(
                        self.op_name)
                    logger.info(f"[{self.op_name}] Performance testBinary I/OFunction Generation Success")

                # General template: benchmark application
                benchmark_code = dsl_adapter.benchmark_impl(
                    self.impl_func_name, "inputs", warmup_times, run_times,
                    self.backend, self.op_name, case_idx=0,
                    framework_model="framework_model" if needs_binary_io else None,
                    framework_adapter=framework_adapter if needs_binary_io else None,
                    device_id=device_id if needs_binary_io else None,
                    framework=self.framework
                )
                logger.debug(f"[{self.op_name}] Generation benchmarkCode Generation Success (Length: {len(benchmark_code)})")
        except Exception as e:
            logger.error(f"[{self.op_name}] Performance test code segment generation failed: {e}", exc_info=True)
            raise

        # Use template variables
        logger.debug(f"[{self.op_name}] Start rendering performance test templates...")
        try:
            rendered_code = template.render(
                op_name=self.op_name,
                framework=self.framework,
                dsl=self.dsl,
                device_id=device_id,
                impl_func_name=self.impl_func_name,
                backend=self.backend,
                arch=self.arch,
                warmup_times=warmup_times,
                run_times=run_times,
                total_count=warmup_times + run_times,
                is_dynamic_shape=is_dynamic_shape,
                # Codes generated by Adapter
                framework_imports=self._prepare_code_lines(framework_imports),
                framework_model_import=self._prepare_code_lines(framework_model_import),
                dsl_imports=self._prepare_code_lines(dsl_imports),
                dsl_impl_import=self._prepare_code_lines(dsl_impl_import),
                special_setup_code=self._prepare_code_lines(special_setup_code),
                device_setup_code=self._prepare_code_lines(device_setup_code),
                process_input_code=self._prepare_code_lines(process_input_code),
                create_impl_code=self._prepare_code_lines(create_impl_code),
                set_seed_code=self._prepare_code_lines(set_seed_code),
                binary_io_functions=self._prepare_code_lines(binary_io_functions),
                needs_binary_io=needs_binary_io,
                tensor_type_name=tensor_type_name,
                benchmark_code=self._prepare_code_lines(benchmark_code),
            )
            logger.info(f"[{self.op_name}] The performance test template was retrofitted and the code length was retrofitted: {len(rendered_code)} Character")
        except Exception as e:
            logger.error(f"[{self.op_name}] Performance test template rendering failed: {e}", exc_info=True)
            raise

        # Writing files
        try:
            with open(profile_file, "w", encoding="utf-8") as f:
                f.write(rendered_code)
            logger.info(f"[{self.op_name}] The performance test script has been written: {profile_file}")
        except Exception as e:
            logger.error(f"[{self.op_name}] Failed to write performance test script: {profile_file}, Error: {e}")
            raise

    def _generate_base_benchmark_code(self, framework_adapter, dsl_adapter, warmup, runs, clear_l2_cache: bool = True):
        """Generate base benchmark code (benchmark trade model)

        Args:
            ramework_adapter: framework adapter
            dsl_adapter: DSL adapter
            Warmup: warmup times
            Runs: Effective run times
            clear_l2_cache: Whether to clear L2 Cache (default True) before each iterative
        """
        profiler_dsl = getattr(dsl_adapter, "profiler_dsl", "other")
        sync_code = _get_framework_sync_code(self.framework, self.backend)

        if self.backend == "ascend":
            framework_arg = (
                f', framework="{self.framework}"'
                if self.framework == "mindspore" else ""
            )
            set_framework_code = ""
            if self.framework == "mindspore":
                set_framework_code = """        import os
        os.environ["TRITON_BACKEND"] = "mindspore"
        try:
            from op_autoresearch.op.utils.triton_autotune_patch import set_framework
            set_framework("mindspore")
        except ImportError:
            pass
"""
            return f"""{set_framework_code}        import time
        try:
            from op_autoresearch.op.verifier.profiler import profiler_npu
        except ImportError:
            profiler_npu = None

        def base_benchmark_fn():
            return framework_model(*inputs)

        if profiler_npu is not None:
            execution_time_us = profiler_npu(
                base_benchmark_fn,
                warmup={warmup},
                active={runs},
                prof_dir_name="prof_base_output",
                keep_res=False,
                suppress_warnings=True,
                clear_l2_cache={clear_l2_cache},
                dsl="{profiler_dsl}"{framework_arg}
            )
            execution_time_ms = execution_time_us / 1000
            method = "profiler_npu"
        else:
            for _ in range({warmup}):
                base_benchmark_fn()
                {sync_code}
            start_time = time.perf_counter()
            for _ in range({runs}):
                base_benchmark_fn()
                {sync_code}
            execution_time_ms = (time.perf_counter() - start_time) * 1000 / max({runs}, 1)
            method = "device_loop_timer"
"""

        return f"""        import time
        def base_benchmark_fn():
            return framework_model(*inputs)

        for _ in range({warmup}):
            base_benchmark_fn()
            {sync_code}
        start_time = time.perf_counter()
        for _ in range({runs}):
            base_benchmark_fn()
            {sync_code}
        execution_time_ms = (time.perf_counter() - start_time) * 1000 / max({runs}, 1)
        method = "{self.backend}_loop_timer"
"""

    def save_speedup_result(
        self,
        speedup: float,
        base_time: float,
        gen_time: float,
        unique_dir: str,
        roofline_time: Optional[float] = None,
        roofline_speedup: Optional[float] = None,
    ):
        """Save Accelerator Ratio result to txt file"""
        try:
            profiling_dir = os.path.join(os.path.expanduser(self.log_dir), self.op_name, "profiling")
            os.makedirs(profiling_dir, exist_ok=True)

            filepath = os.path.join(profiling_dir, "speed_up_record.txt")

            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"op_name: {self.op_name}, task_id: {self.task_id}, unique_dir: {unique_dir}, ")
                f.write(f"base_time: {base_time:.6f} us, generation_time: {gen_time:.6f} us, ")
                f.write(f"speedup: {speedup:.6f}x")
                if roofline_time is not None and roofline_speedup is not None:
                    f.write(
                        f", roofline_time: {roofline_time:.6f} us, "
                        f"roofline_speedup: {roofline_speedup:.6f}x"
                    )
                f.write("\n")

            logger.debug(f"[{self.task_id}:{self.op_name}] Accelerator ratio result saved")

        except Exception as e:
            logger.warning(f"[{self.task_id}:{self.op_name}] Failed to save acceleration ratio result: {str(e)}")

    async def run_profile(self, task_info: Dict[str, Any], current_step: int = 0,
                          device_id: int = -1,
                          profile_settings: Optional[dict] = None) -> dict:
        """Run profile analysis

        N. B.: Similar to the run() method, device management is integrated in this method

        Args:
            Device_id: deviceID (Default-1 means automated management, LocalWorker automatically gets it from device_pool)

        Returns:
            dict: profiling results, containing the following fields:
                - gen_time: Generate code execution time (microseconds)
                - base_time: baseline code implementation time (microseconds)
                - Speed-up.
                - Roofline_time: SOlar used roofline time (microseconds, optional)
                - Roofline_speedup: Roofline_time / g_time (optional)
                - Autotune_summary: autotune configuration details (triton DSL only)
        """
        acquired_device = None
        acquired_lease = None
        profile_settings = dict(profile_settings or {})
        unique_dir_name = f"Iteration{self.task_id}_Step{current_step}_verify"
        try:
            logger.info(f"[{self.op_name}] Preparing profile config before device acquire")
            self.dsl_adapter.prepare_config(self.config, task_info=task_info)

            run_times = resolve_run_times(profile_settings.get("run_times"))
            warmup_times = resolve_warmup_times(
                profile_settings.get("warmup_times"))
            effective_profile_settings = dict(profile_settings)
            base_only_after_failed_verify = (
                getattr(self, "last_verify_ok", None) is False
            )

            cached_baseline_time_us = None
            has_user_override = effective_profile_settings.get("override_base_section") is not None
            if not has_user_override:
                cached_baseline_time_us = self._get_cached_baseline_time_us(warmup_times, run_times)
                if cached_baseline_time_us is not None:
                    # Data cache only stores scalar; per_shape array comes
                    # from workspace sticky path (state.baseline_per_shape_us).
                    effective_profile_settings["override_base_section"] = make_profile_section(
                        cached_baseline_time_us, method="override")
                    effective_profile_settings["skip_base_profile"] = True

            if self.worker is not None:
                logger.info(f"[{self.op_name}] Acquiring device for profile project generation")
                acquired_device, acquired_lease = await self.worker.acquire_device(self.task_id)
                actual_device_id = acquired_device
                logger.info(f"[{self.op_name}] Acquired device {actual_device_id} for profile")
            else:
                # No worker (old process compatible)
                actual_device_id = device_id if device_id != -1 else 0
                logger.info(f"[{self.op_name}] Using device {actual_device_id} (no worker, deprecated flow)")

            # Get Authentication Directory
            expanded_log_dir = os.path.expanduser(self.log_dir)
            verify_dir = os.path.join(expanded_log_dir, self.op_name, unique_dir_name)

            # Make sure the directory exists.
            os.makedirs(verify_dir, exist_ok=True)

            # Check if you need a husband to be a code file (required when calling independently).
            # When the immediately preceding verify failed, profile is only
            # allowed to measure the framework baseline; keep it independent
            # from generated DSL artifacts so a broken seed cannot hide the
            # ref latency WA needs for baseline anchoring.
            if base_only_after_failed_verify:
                self._materialize_framework_bundle(verify_dir, self.framework_code)
                stale_gen_files = (
                    f"profile_{self.op_name}_generation.py",
                    "generation_profile_result.json",
                    "roofline_profile_result.json",
                )
                for filename in stale_gen_files:
                    stale_path = os.path.join(verify_dir, filename)
                    if os.path.exists(stale_path):
                        try:
                            os.remove(stale_path)
                        except OSError:
                            logger.warning(
                                f"[{self.op_name}] failed to remove stale "
                                f"profile artifact: {stale_path}")
            elif not self._verify_impl_artifacts_ready(verify_dir):
                # The code file doesn't exist, sir.
                impl_code = task_info.get("coder_code", "")
                if not impl_code:
                    raise ValueError(f"[{self.op_name}] task_info Missing coder_codeCould not generate performance test code file")

                logger.info(f"[{self.op_name}] Generating verify project for profile")
                self.gen_verify_project(impl_code, verify_dir, actual_device_id)
                logger.info(f"[{self.op_name}] Verify project for profile generated")

            # Generate program scripts
            # For RemoteWorker, use 0 as a placeholder for code generation (actual device is managed by a remote server)
            # For LocalWorker, use existing actual_device_id
            # Skips only if the result is visible or a valid base line has been provided.
            skip_base_profile = effective_profile_settings.get('skip_base_profile', False)
            override_base_section = effective_profile_settings.get('override_base_section')
            has_valid_override = (
                isinstance(override_base_section, dict)
                and isinstance(override_base_section.get("avg_us"), (int, float))
                and 0 < override_base_section["avg_us"] < float('inf')
            )
            skip_base = skip_base_profile or has_valid_override
            old_generation_enabled = getattr(
                self, "_profile_generation_enabled", True)
            self._profile_generation_enabled = not base_only_after_failed_verify
            try:
                logger.info(f"[{self.op_name}] Generating profile project")
                self.gen_profile_project(verify_dir, actual_device_id,
                                         warmup_times, run_times,
                                         skip_base=skip_base)
                logger.info(f"[{self.op_name}] Profile project generated")
            finally:
                self._profile_generation_enabled = old_generation_enabled

            # Pack and send to Worker for execution
            logger.info(f"[{self.op_name}] Packing profile project")
            package_data = self._pack_directory(verify_dir)

            if not self.worker:
                # Check if device_id is -1 (means automanage)
                if device_id == -1:
                    raise RuntimeError(
                        f"[{self.op_name}] Worker not set and device_id=-1 (RemoteWorker mode). "
                        "Worker must be provided by Task or WorkerManager for RemoteWorker."
                    )
                # If there is no workingr, create Localworker (for testing scenario) based on device_id
                # Note: At this time, actual_device_id has been set to device_id (because device_id! = -1)
                import warnings
                warnings.warn(
                    f"⚠️  [DEPRECATED] KernelVerifier Automatically Create LocalWorker It's the old bottom logic, only for testing.\n"
                    f"Suggested new formulation:\n"
                    f"  1. Register before calling Worker to WorkerManager(line code):\n"
                    f"     from op_autoresearch.core.worker.manager import register_local_worker\n"
                    f"     \n"
                    f"     await register_local_worker([{actual_device_id}], backend='{self.backend}', arch='{self.arch}')\n"
                    f"  2. Task It's automatic from WorkerManager Access worker\n"
                    f"Example:examples/run_torch_npu_triton_single.py",
                    DeprecationWarning,
                    stacklevel=2
                )
                logger.warning(f"⚠️  [{self.op_name}] Worker not set, creating temporary LocalWorker (deprecated)")

                from op_autoresearch.core.worker.local_worker import LocalWorker
                from op_autoresearch.core.async_pool.device_pool import DevicePool
                logger.info(f"[{self.op_name}] Worker not set, creating LocalWorker with device [{actual_device_id}]")
                device_pool = DevicePool([actual_device_id])
                self.worker = LocalWorker(device_pool=device_pool, backend=self.backend)

            # Check LocalWorker for access_pol
            from op_autoresearch.core.worker.local_worker import LocalWorker
            if isinstance(self.worker, LocalWorker):
                if not hasattr(self.worker, 'device_pool') or self.worker.device_pool is None:
                    raise RuntimeError(
                        f"[{self.op_name}] LocalWorker must have device_pool. "
                        "This should be provided by Task when creating _private_worker."
                    )

            # Send the complete profile_settings to the worker.
            full_settings = {
                **effective_profile_settings,
                'backend': self.backend,
                'dsl': self.dsl,
                'op_name': self.op_name,
                'framework': self.framework,
                'arch': self.arch,
                'bench_type': self.bench_type,
                'enable_roofline': effective_profile_settings.get('enable_roofline', True),
                'roofline_arch_config': effective_profile_settings.get(
                    'roofline_arch_config',
                    self.config.get('roofline_arch_config')
                ),
            }

            logger.info(f"[{self.op_name}] Dispatching profile project to worker")
            result = await self.worker.profile(package_data, self.task_id, self.op_name, full_settings)
            logger.info(f"[{self.op_name}] Worker profile returned")

            # Sync Artifacts to Verify_dir (for RemoteWorker)
            artifacts = result.get('artifacts', {})
            if artifacts:
                sync_artifacts_to_directory(artifacts, verify_dir, self.task_id)

            # Worker returns the canonical field: gen_time / base_time is aggregate
            # scalar, per_shape_gen_us /per_shape_base_us is per-case array.
            # Base_time across backend scene may be None.
            gen_time = result.get('gen_time')
            base_time = result.get('base_time')
            speedup = result.get('speedup', 0.0)
            per_shape_gen_us = list(result.get('per_shape_gen_us') or [])
            per_shape_base_us = list(result.get('per_shape_base_us') or [])
            gen_method = result.get('gen_method')
            base_method = result.get('base_method')
            roofline_time = result.get('roofline_time')
            roofline_speedup = result.get('roofline_speedup', 0.0)
            roofline_result = result.get('roofline')

            # Shape descriptors come from the verify sidecar (populated by
            # ``run()`` immediately before this profile call). One owner,
            # one source — no defensive fallbacks in downstream consumers.
            case_descs: list = []
            sidecar = getattr(self, "last_verify_sidecar", None)
            if isinstance(sidecar, dict):
                case_descs = [c.get("case_desc", "")
                              for c in (sidecar.get("per_case") or [])
                              if isinstance(c, dict)]

            if (not skip_base and base_time is not None
                    and base_time > 0 and base_time < float('inf')):
                self._store_baseline_result_in_data_cache(
                    base_time_us=base_time,
                    warmup_times=warmup_times,
                    run_times=run_times,
                    artifacts=artifacts,
                )

            # Processing None values for log output
            gen_time_display = gen_time if gen_time is not None else float('inf')
            base_time_display = base_time if base_time is not None else float('inf')

            if gen_time is not None:
                self.save_speedup_result(
                    speedup,
                    base_time_display,
                    gen_time_display,
                    unique_dir_name,
                    roofline_time=roofline_time,
                    roofline_speedup=roofline_speedup if roofline_time is not None else None,
                )

            speedup_percent = speedup * 100.0
            logger.info(f"orig performance is {base_time_display:.2f} us")
            if gen_time is not None:
                logger.info(f"op_autoresearch performance is {gen_time_display:.2f} us")
            else:
                logger.info("op_autoresearch performance skipped (verify failed before generation profile)")
            if roofline_time is not None:
                logger.info(f"solar roofline performance is {roofline_time:.2f} us")
                logger.info(f"roofline speedup is {roofline_speedup:.4f}x")
            logger.info(f"[{self.task_id}:{self.op_name}] profilingCompleted, accelerated ratio (baseline is100%): {speedup_percent:.2f} %")

            # Build return result. per_shape_* / case_decs on caller side (eval_bridge) Direct
            # Spell metrics dict, sidecar not needed again.
            profile_result = empty_profile_result(result.get('error'))
            profile_result.update({
                'gen_time': gen_time,
                'base_time': base_time,
                'speedup': speedup,
                'per_shape_gen_us': per_shape_gen_us,
                'per_shape_base_us': per_shape_base_us,
                'case_descs': case_descs,
                'gen_method': gen_method,
                'base_method': base_method,
                'roofline_time': roofline_time,
                'roofline_speedup': roofline_speedup,
                'roofline': roofline_result,
                'artifacts': artifacts,
                'unique_dir': unique_dir_name,
            })

            if self.dsl_adapter.emits_autotune_artifacts:
                autotune_summary = self.read_autotune_results_from_directory(verify_dir)
                if autotune_summary:
                    profile_result['autotune_summary'] = autotune_summary
                    logger.info(f"[{self.op_name}: {self.task_id}] AutotuneConfigure Details:\n{autotune_summary}")

            return profile_result
        except Exception as e:
            logger.warning(f"[{self.task_id}:{self.op_name}] profilingFailed: {str(e)}")
            result = empty_profile_result(error=str(e))
            result.update(case_descs=[], unique_dir=unique_dir_name)
            return result
        finally:
            # Always release the device for the whole profile lifecycle.
            if acquired_device is not None:
                await self.worker.release_device(acquired_device, acquired_lease, self.task_id)
                logger.info(f"[{self.op_name}] Released device {acquired_device}")

    def read_autotune_results_from_directory(self, verify_dir: str) -> str:
        """Read all autotune results and format the output from the authentication directory

        Read all autotune_info_case_*.json files in the specified directory.
        and output in a format similar to TRITON_PRINT_AUTOOTUNING=1.

        Args:
            Verify_dir: Verified Directory Path

        Returns:
            Formatted autotune result string, formatted as follows:

            Case 0:
            All config timings for kernel_name:
              Config 1: BLOCK_M=128, BLOCK_N=256 -> 145.2300us (BEST)
              Config 2: BLOCK_M=64, BLOCK_N=128 -> 178.5600us
              ...
        """

        result_lines = []

        # Find all autotune files
        verify_path = Path(verify_dir)
        autotune_files = sorted(verify_path.glob("autotune_info_case_*.json"))

        if not autotune_files:
            return ""

        # Read & Format
        for autotune_file in autotune_files:
            # Extract Case Index
            case_idx = autotune_file.stem.split('_')[-1]

            try:
                with open(autotune_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                result_lines.append(f"Case {case_idx}:")

                # Every kernel
                for kernel_name, configs in data.items():
                    result_lines.append(f"All config timings for {kernel_name}:")

                    # Sort Output by Rank
                    sorted_configs = sorted(configs, key=lambda x: x['rank'])

                    for config_info in sorted_configs:
                        config_str = config_info['config']
                        timing_us = config_info['timing_us']
                        is_best = config_info['is_best']
                        rank = config_info['rank']

                        status = " (BEST)" if is_best else ""
                        result_lines.append(f"  Config {rank}: {config_str} -> {timing_us:.4f}us{status}")

                result_lines.append("")  # Empty Lines Separate Different Case

            except Exception as e:
                logger.warning(f"[{self.op_name}: {self.task_id}] ReadautotuneFile Failed {autotune_file.name}: {e}")

        return "\n".join(result_lines)

    # ------------------------------------------------------------------
    # Autotune config authentication aids (used at OP_AUTORESEARCH_VERIFY_PER_CONFIG=1)
    # ------------------------------------------------------------------

    def _detect_triton_autotune(self, code: str) -> bool:
        """Check if the code contains @triton.autotune decorator"""
        return '@triton.autotune' in code or '@autotune' in code

    def _extract_autotune_configs(self, code: str) -> list:
        """Extract all unannotated autotonne configs from the Triton code"""

        pattern = r'@triton\.autotune\s*\(\s*configs\s*=\s*\[(.*?)\]'
        match = re.search(pattern, code, re.DOTALL)
        if not match:
            return []

        configs_str = match.group(1)
        config_pattern = r'triton\.Config\s*\([^)]*\{[^}]+\}[^)]*\)'
        all_matches = re.finditer(config_pattern, configs_str, re.DOTALL)

        valid_configs = []
        for match in all_matches:
            start_pos = match.start()
            last_newline = configs_str.rfind('\n', 0, start_pos)
            line_start = last_newline + 1 if last_newline != -1 else 0
            prefix = configs_str[line_start:start_pos]
            if '#' not in prefix:
                valid_configs.append(match.group(0))

        return valid_configs

    def _count_all_autotune_configs(self, code: str) -> int:
        """Quantity of all autotune config (including annotated)"""

        pattern = r'@triton\.autotune\s*\(\s*configs\s*=\s*\[(.*?)\]'
        match = re.search(pattern, code, re.DOTALL)
        if not match:
            return 0
        return match.group(1).count('triton.Config')

    def _generate_single_config_code(self, original_code: str, config_to_keep: str, config_index: int) -> str:
        """Generate code containing only one config"""

        all_configs = self._extract_autotune_configs(original_code)
        if not all_configs:
            return original_code

        new_configs_block = f"configs=[\n        {config_to_keep},\n    ]"
        pattern = r'configs\s*=\s*\[(.*?)\]'
        return re.sub(pattern, new_configs_block, original_code, count=1, flags=re.DOTALL)

    def _generate_final_code_with_valid_configs(self, original_code: str, valid_configs: list, all_configs: list) -> str:
        """Generate final code: keep the right config, comment on the wrong config"""

        if not all_configs:
            return original_code

        new_configs_lines = []
        for config in all_configs:
            if config in valid_configs:
                new_configs_lines.append(f"        {config},")
            else:
                config_lines = config.split('\n')
                commented_lines = [f"        # {line}" if line.strip() else line for line in config_lines]
                new_configs_lines.append('\n'.join(commented_lines) + ',  # Failed verification')

        new_configs_block = "configs=[\n" + "\n".join(new_configs_lines) + "\n    ]"
        pattern = r'configs\s*=\s*\[(.*?)\]'
        return re.sub(pattern, new_configs_block, original_code, count=1, flags=re.DOTALL)

    def _save_verification_result_to_jsonl(self, verify_dir: str, current_step: int, verification_passed: bool,
                                           verify_logs: str, all_configs_count: int = 0, valid_configs_count: int = 0):
        """
        Save authentication results to JSONL file

        Args:
            Verify_dir: Authentication Directory
            current_step: Current steps
            Verification_passed: Verify pass
            Verify_logs: Validation log
            All_configs_count: all config quantities (autotune earmarked)
            Valid_configs_count: Number of adopted configs (autotune exclusive)
        """
        result_jsonl_path = os.path.join(os.path.expanduser(self.log_dir), "verification_results.jsonl")
        result_info = {
            "task_name": self.op_name,
            "task_id": self.task_id,
            "step": current_step,
            "verify_dir": verify_dir,
            "passed": verification_passed,
            "error_log": verify_logs,
            "timestamp": datetime.now().isoformat(),
            "framework": self.framework,
            "dsl": self.dsl,
            "backend": self.backend,
            "arch": self.arch
        }

        if all_configs_count > 0:
            result_info["autotune_configs"] = {
                "total": all_configs_count,
                "passed": valid_configs_count
            }

        with open(result_jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_info, ensure_ascii=False, indent=2) + '\n\n')

    async def _verify_configs_separately(self, target_code: str, verify_dir: str, device_id: int, verify_timeout: int, current_step: int = 0) -> Tuple[bool, str, str]:
        """
        The config authentication mode (which is enabled when OP_AUTORESEARCH_VERIFY_PER_CONFIG=1) is then run back to the full code.

        Returns:
            Tuple [bool, st, st]: (whether config passed, validation log, final code)
        """
        logger.info(f"[{self.op_name}] [PVconfigMode] Detected autotune, start checking one by one...")

        total_configs_count = self._count_all_autotune_configs(target_code)
        all_configs = self._extract_autotune_configs(target_code)

        if not all_configs:
            if total_configs_count > 0:
                logger.info(f"[{self.op_name}] Detected {total_configs_count} individual config, but all have been commented")
                verify_logs = [
                    "=== Autotune ConfigAuthentication===\n",
                    f"Detected {total_configs_count} individual config, all commented (failure to verify)\n",
                    "Skip authentication, directly return failed result \n",
                ]
                try:
                    self.gen_verify_project(target_code, verify_dir, device_id)
                except Exception as e:
                    verify_logs.append(f"\nFailed to generate authentication project: {e}\n")
                self._save_verification_result_to_jsonl(
                    verify_dir, current_step, False, "".join(verify_logs),
                    total_configs_count, 0
                )
                return False, "".join(verify_logs), target_code
            else:
                logger.warning(f"[{self.op_name}] Could not initialise Bonobo configBack to direct authentication.")
                return None, "", target_code

        skipped_count = total_configs_count - len(all_configs)
        if skipped_count > 0:
            logger.info(f"[{self.op_name}] {total_configs_count} individual config in {skipped_count} Annotated to verify the remaining {len(all_configs)} individual")
        else:
            logger.info(f"[{self.op_name}] Extracted {len(all_configs)} individual config, start checking one by one...")

        valid_configs = []
        verify_logs = []
        verify_logs.append("=== Autotune ConfigArticle by Article Validation===\n")
        if skipped_count > 0:
            verify_logs.append(f"Detected {total_configs_count} individual config,among them {skipped_count} A comment has been made\n")
            verify_logs.append(f"To be validated config Number: {len(all_configs)}\n\n")
        else:
            verify_logs.append(f"Total {len(all_configs)} individual config\n\n")

        consecutive_timeouts = 0
        max_consecutive_timeouts = 2

        for i, config in enumerate(all_configs):
            config_num = i + 1
            logger.info(f"[{self.op_name}] Authentication Config {config_num}/{len(all_configs)}...")
            verify_logs.append(f"--- Config {config_num} ---\n{config}\n")

            try:
                single_config_code = self._generate_single_config_code(target_code, config, i)
                temp_verify_dir = os.path.join(verify_dir, f"config_{config_num}_verify")
                os.makedirs(temp_verify_dir, exist_ok=True)

                self.gen_verify_project(single_config_code, temp_verify_dir, device_id)
                config_res, config_log = await self.run_verify(temp_verify_dir, timeout=verify_timeout)

                if config_res:
                    verify_logs.append("Validation via \n \n")
                    valid_configs.append(config)
                    logger.info(f"[{self.op_name}] Config {config_num} Validation pass")
                    consecutive_timeouts = 0
                else:
                    verify_logs.append(f"Authentication Failed\nError Log:\n{config_log}\n\n")
                    logger.info(f"[{self.op_name}] Config {config_num} Authentication Failed")
                    log_lower = config_log.lower()
                    if "timed out" in log_lower or "timeout after" in log_lower or "timeouterror" in log_lower or "Calculate timeout" in log_lower:
                        consecutive_timeouts += 1
                        if consecutive_timeouts >= max_consecutive_timeouts:
                            skip_count = len(all_configs) - i - 1
                            warn_msg = f"Continuous {consecutive_timeouts} individual config Timeout, trigger. Fail-Fast"
                            if skip_count > 0:
                                warn_msg += f"Skip the rest {skip_count} individual"
                            logger.warning(f"[{self.op_name}] {warn_msg}")
                            verify_logs.append(f"{warn_msg}\n\n")
                            shutil.rmtree(temp_verify_dir, ignore_errors=True)
                            break
                    else:
                        consecutive_timeouts = 0

                shutil.rmtree(temp_verify_dir, ignore_errors=True)
            except Exception as e:
                verify_logs.append(f"Authentication anomaly: {e}\n\n")
                logger.error(f"[{self.op_name}] Config {config_num} Authentication anomaly: {e}")
                consecutive_timeouts = 0

        verify_logs.append(f"Adopted config Number: {len(valid_configs)}/{len(all_configs)}\n")

        full_code_verify_passed = True
        if len(valid_configs) == len(all_configs) and len(all_configs) > 1:
            verify_logs.append("\n = = complete code regression verification (all config combined) = = \n")
            logger.info(f"[{self.op_name}] PV config All pass. Start full code back validation....")

            try:
                full_verify_dir = os.path.join(verify_dir, "full_code_verify")
                os.makedirs(full_verify_dir, exist_ok=True)
                self.gen_verify_project(target_code, full_verify_dir, device_id)
                full_res, full_log = await self.run_verify(full_verify_dir, timeout=verify_timeout)

                if full_res:
                    verify_logs.append("Full code back authentication via \n")
                    logger.info(f"[{self.op_name}] Full code back authentication passed.")
                else:
                    full_code_verify_passed = False
                    verify_logs.append(f"Full code backvalidation failed!\nError Log:\n{full_log}\n\n")
                    verify_logs.append(
                        "[Key issues] Each config pass individually, but all config merge failed. \n"
                        "This is usually due to the lack of restore_value parameters for the @triton.autotune decorator. \n"
                        "Autotune benchmark will repeat the kernel, and the output of different config will pollute each other. \n"
                        "Add a restore_value=['all output pointer parameter'] to @triton.autotune. \n"
                    )
                    logger.warning(f"[{self.op_name}] PV config The complete code verification failed and is suspected to be missing. restore_value")

                shutil.rmtree(full_verify_dir, ignore_errors=True)
            except Exception as e:
                verify_logs.append(f"Full code regression authentication anomaly: {e}\n")
                logger.error(f"[{self.op_name}] Full code regression authentication anomaly: {e}")

        final_code = self._generate_final_code_with_valid_configs(target_code, valid_configs, all_configs)
        verification_passed = len(valid_configs) > 0 and full_code_verify_passed

        if verification_passed:
            verify_logs.append(f"Validation passed, reserved. {len(valid_configs)} That's right. config\n")
            logger.info(f"[{self.op_name}] Autotune config Validation completed: {len(valid_configs)}/{len(all_configs)} Pass.")
        elif len(valid_configs) > 0 and not full_code_verify_passed:
            verify_logs.append("PVconfigCould not close temporary folder: %srestore_value\n")
            logger.info(f"[{self.op_name}] Required Add restore_value")
        else:
            verify_logs.append("None of the config failed to verify \n")
            logger.info(f"[{self.op_name}] All config None verified")

        verify_logs.append("\n== Generating Final Validation Project==\n")
        try:
            self.gen_verify_project(final_code, verify_dir, device_id)
            logger.info(f"[{self.op_name}] Final validation of project generation success")
        except Exception as e:
            verify_logs.append(f"Failed to generate final validation project: {e}\n")
            logger.error(f"[{self.op_name}] Failed to generate final validation project: {e}")
            verification_passed = False

        self._save_verification_result_to_jsonl(
            verify_dir, current_step, verification_passed, "".join(verify_logs),
            len(all_configs), len(valid_configs)
        )
        return verification_passed, "".join(verify_logs), final_code

    async def run(self, task_info: Dict[str, Any], current_step: int = 0, device_id: int = -1):
        """
        Run kernel verifier, verify the validity of the code

        Args:
            task_info: Task Info Dictionary, with all codes and status
            current_step: Current steps
            Device_id: deviceID (Default-1 means automated management, LocalWorker automatically gets it from device_pool)

        Returns:
            Tuple [bool, st]: (validation results, error log)
        """
        logger.info(f"Verifier Run - Step: {current_step}")
        self.last_verify_ok = None

        # Get code from tsk_info according to the type of realization
        target_code = task_info.get('coder_code', '')

        if not target_code:
            logger.error("No target code found for verification")
            self.last_verify_ok = False
            return False, "No target code found for verification"

        # Dynamically create a authentication directory
        verify_dir = self._create_verify_dir(current_step)

        logger.info(f"[{self.op_name}] Preparing verify config before device acquire")
        self.dsl_adapter.prepare_config(self.config, task_info=task_info)
        logger.info(f"[{self.op_name}] Verify config prepared")

        # One device for the whole verify (generate scripts + execute),
        # released in the finally below. LocalWorker and RemoteWorker share
        # the same acquire/release contract; on the remote path the daemon's
        # lease reaper backstops a client that dies before release.
        acquired_device = None
        acquired_lease = None
        if self.worker is not None:
            logger.info(f"[{self.op_name}] Acquiring device for verify project generation")
            acquired_device, acquired_lease = await self.worker.acquire_device(self.task_id)
            actual_device_id = acquired_device
            logger.info(f"[{self.op_name}] Acquired device {actual_device_id} for verify")
        else:
            # No worker (old process compatible)
            actual_device_id = device_id if device_id != -1 else 0
            logger.info(f"[{self.op_name}] Using device {actual_device_id} (no worker, deprecated flow)")

        try:
            verify_per_config = (
                os.environ.get("OP_AUTORESEARCH_VERIFY_PER_CONFIG", "0") == "1"
                or self.config.get("verify_per_config", False)
            )
            is_triton_autotune = (
                self.dsl_adapter.supports_autotune_configs
                and self._detect_triton_autotune(target_code)
            )

            if verify_per_config and is_triton_autotune:
                config_verify_result, config_verify_log, final_code = await self._verify_configs_separately(
                    target_code, verify_dir, actual_device_id,
                    resolve_eval_timeout(self.config.get('verify_timeout')),
                    current_step
                )
                if config_verify_result is not None:
                    if config_verify_result:
                        task_info['coder_code'] = final_code
                    self.last_verify_ok = bool(config_verify_result)
                    return config_verify_result, config_verify_log

            # uses_cannbench_precision DSLs need the reference run LIVE so we can
            # produce the FP64-CPU golden + target-precision native pair
            # (dual_reference); the cached-inputs path stores only one precomputed
            # golden and sets framework_model=None, which would defeat the parity.
            _cannbench_precision = getattr(
                self.dsl_adapter, "uses_cannbench_precision", False)
            if _cannbench_precision:
                logger.info(
                    f"[{self.op_name}] cannbench precision: skipping reference "
                    f"cache to run FP64 golden + native reference live")
            else:
                logger.info(f"[{self.op_name}] Preparing cached reference data")
                cached_reference_data = await self._prepare_cached_reference_data(actual_device_id)
                if cached_reference_data:
                    self._apply_cached_reference_data(cached_reference_data)
                    logger.info(f"[{self.op_name}] Cached reference data applied")
                else:
                    logger.info(f"[{self.op_name}] No cached reference data applied")

            # Default mode: verify the complete code directly
            project_gen_log = ""
            try:
                # For RemoteWorker, use 0 as a placeholder for code generation (actual device is managed by a remote server)
                # For LocalWorker, use existing actual_device_id
                logger.info(f"[{self.op_name}] Generating verify project")
                self.gen_verify_project(target_code, verify_dir, actual_device_id)
                logger.info(f"[{self.op_name}] Verify project generated")
            except Exception as e:
                # Catch abnormalities in gen_verify_project, recorded in project_gen_log
                error_msg = str(e)
                logger.error(f"Validation of project generation failed: {error_msg}")
                project_gen_log = f"Project generation failed: {error_msg}\n"

            # Fetch timeout configuration from config
            verify_timeout = resolve_eval_timeout(
                self.config.get('verify_timeout'))

            # Run Authentication
            # Worker.verify() just executes scripts without managing device (Device is already set in scripts)
            logger.info(f"[{self.op_name}] Running verify project")
            verify_res, verify_log = await self.run_verify(
                verify_dir, timeout=verify_timeout, device_id=actual_device_id
            )
            logger.info(f"[{self.op_name}] Verify project finished: result={verify_res}")

            # Collapse project generation logs and validation logs
            verify_log = project_gen_log + verify_log

            # The verify script (kernel_verify_template_refactored.j2) drops
            # a structured per-case sidecar at verify_dir/verify_result.json.
            # Surface it via `self.last_verify_sidecar` so callers that want
            # the per-case shape (per_case / failed_indices / error_source /
            # failure_kind) can read it without an extra disk hop. The
            # (bool, str) tuple return stays unchanged for backward compat.
            self.last_verify_sidecar = None
            self.last_verify_dir = verify_dir
            sidecar_path = os.path.join(verify_dir, "verify_result.json")
            if os.path.isfile(sidecar_path):
                try:
                    with open(sidecar_path, "r", encoding="utf-8") as _fp:
                        self.last_verify_sidecar = json.load(_fp)
                except Exception as _e:
                    logger.warning(
                        f"[{self.op_name}] failed to read verify_result.json: {_e}")

            # Save authentication results to JSONL file
            self._save_verification_result_to_jsonl(verify_dir, current_step, verify_res, verify_log)

            # Note: Not copied to passed_cases
            # If multiple case tests are enabled, more case validation is required before copying
            # Copy operation is managed centrally by task.py

            self.last_verify_ok = bool(verify_res)
            return verify_res, verify_log
        finally:
            # Always release the device for the whole verify lifecycle.
            if acquired_device is not None:
                await self.worker.release_device(acquired_device, acquired_lease, self.task_id)
                logger.info(f"[{self.op_name}] Released device {acquired_device}")
