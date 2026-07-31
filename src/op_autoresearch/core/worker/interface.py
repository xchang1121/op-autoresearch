from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Tuple, Any, Dict, Optional, Union


def empty_profile_result(error: Optional[str] = None) -> Dict[str, Any]:
    """Canonical profile failure shape shared by local and remote workers."""
    result: Dict[str, Any] = {
        "gen_time": None,
        "base_time": None,
        "speedup": 0.0,
        "per_shape_gen_us": [],
        "per_shape_base_us": [],
        "gen_method": None,
        "base_method": None,
        "roofline_time": None,
        "roofline_speedup": 0.0,
        "roofline": None,
        "artifacts": {},
    }
    if error is not None:
        result["error"] = error
    return result


class WorkerInterface(ABC):
    """
    Abstract base class for OP_AUTORESEARCH Workers (Local and Remote).
    """

    @abstractmethod
    async def verify(self, package_data: Union[bytes, str], task_id: str,
                     op_name: str, timeout: Optional[int] = None
                     ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute verification task.

        Note: device management (acquire/release) is the responsibility of the caller
        {\\cHFFFFFF}{\\cH00FFFF} Worker is only responsible for implementing the script that has been generated (script contains the correct data_id)

        Args:
            Package_data: Verification project content.
                or a local directory path (only used when LocalWorker directly reuses an existing directory).
            task_id: Unique task identifier.
            op_name: Operator name.
            timeout: Execution timeout in seconds.

        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, log_output, artifacts)
            - Access: Verify success
            -log_output: Execute log
            - documents: generated during execution in {relative_path: json_content}
              For example: \"autotune_info_case_0.json\":, \"subdir/result.json\": }
        """
        pass

    @abstractmethod
    async def profile(self, package_data: bytes, task_id: str, op_name: str, profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute profiling task.

        Args:
            package_data: The compressed verification project (TAR bytes).
            task_id: Unique task identifier.
            op_name: Operator name.
            profile_settings: Settings for profiling (e.g., warmup_times, run_times).

        Returns:
            Dict[str, Any]: Profiling results, including:
                - gen_time: Generate code execution time
                -base_time: baseline code implementation time
                - Speed-up.
                - Roofline_time: SOlar used roofline predictive time (microseconds, optional)
                - Roofline_speedup: Roofline_time / g_time (optional)
                -Roofline: Roofline Detail Dictionary (optional)
                - documents: generated during execution in {relative_path: json_content}
        """
        pass

    @abstractmethod
    async def generate_reference(self, package_data: bytes, task_id: str,
                                 op_name: str,
                                 timeout: Optional[int] = None
                                 ) -> Tuple[bool, str, bytes]:
        """
        Execute task_desc and generate reference data.

        For the CUDA-to-Asend conversion scenario: execute the Triton-CUDA code on the GPU Worker.
        Saves the output as a reference data (.pt file) for NPU Worker to verify the correctness of the converted code.

        Args:
            package_data: The compressed project (TAR bytes) containing reference.py and verify script.
            task_id: Unique task identifier.
            op_name: Operator name.
            timeout: Execution timeout in seconds.

        Returns:
            Tuple[bool, str, bytes]: (success, log_output, reference_data_bytes)
            -success: Successfully generate reference data
            -log_output: Execute log
            - Reference_data_bytes:.pt binary content (when successful), empty b''
        """
        pass

    @abstractmethod
    async def profile_single_task(self, package_data: bytes, task_id: str, op_name: str,
                                   profile_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute single task profiling (only measure task_desc performance, no base comparison).

        A separate measure of the performance of a certain section of the code does not allow comparison.
        Applies to a scenario that requires a separate measurement of a Model execution time.

        Args:
            package_data: The compressed project (TAR bytes) containing profile script.
            task_id: Unique task identifier.
            op_name: Operator name.
            profile_settings: Settings for profiling (e.g., warmup_times, run_times).

        Returns:
            Dict[str, Any]: Profiling results, including:
                - time_us: execution time (microseconds)
                - Success.
                -log: Execute Log
        """
        pass

    @abstractmethod
    async def get_doc(self, doc_name: str) -> str:
        """
        Retrieving the visible document contents of the worker environment.

        Args:
            Doc_name: Document identifier, e. g. \"triton_ascend_api\"

        Returns:
            st: Document content
        """
        pass

    @abstractmethod
    async def acquire_device(self, task_id: str = "unknown",
                             timeout: Optional[float] = None) -> Tuple[int, int]:
        """Reserve a device. Returns ``(device_id, lease_id)``; the lease_id
        must be presented on release. LocalWorker delegates to its DevicePool;
        RemoteWorker calls the daemon's /acquire_device."""
        ...

    @abstractmethod
    async def release_device(self, device_id: int, lease_id: int,
                             task_id: str = "unknown") -> None:
        """Return a device acquired under ``lease_id``. Idempotent; a stale
        lease_id is rejected (won't free a successor's device)."""
        ...

    @asynccontextmanager
    async def device_lease(self, task_id: str = "unknown", *,
                           timeout: Optional[float] = None):
        """Hold a device for the block, releasing on normal exit, exception,
        AND cancellation. The single structural way to obtain a device — the
        lease_id is tracked internally so callers only see the device id. If
        the client process is killed before /release_device runs, the daemon's
        reaper reclaims the device, so it can never be permanently leaked.
        """
        device_id, lease_id = await self.acquire_device(task_id, timeout=timeout)
        try:
            yield device_id
        finally:
            await self.release_device(device_id, lease_id, task_id)
