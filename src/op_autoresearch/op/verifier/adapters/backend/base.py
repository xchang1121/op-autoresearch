"""Base class for backend adapters."""

from abc import ABC, abstractmethod
from typing import Optional, Any


class BackendAdapter(ABC):
    """Abstract base class for backend adapters.

    Backend adapters provide a unified interface for different hardware backends
    (CUDA, Ascend, CPU) to handle environment setup, synchronization, and profiling.
    """

    @abstractmethod
    def setup_environment(self, device_id: int, arch: str) -> None:
        """Setup environment variables.

        Args:
            device_id: Device ID
            arch: Architecture
        """
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize device (wait for computation to complete)."""
        pass

    @abstractmethod
    def get_profiler(self) -> Optional[Any]:
        """Get profiler object for performance analysis.

        Returns:
            Profiler object or None
        """
        pass

    @abstractmethod
    def get_device_string(self, device_id: int) -> str:
        """Get device string for logging.

        Args:
            device_id: Device ID

        Returns:
            str: Device string (e.g., "cuda:0", "npu:0", "cpu")
        """
        pass

    @abstractmethod
    def validate_arch(self, arch: str) -> bool:
        """Validate if architecture is supported.

        Args:
            arch: Architecture string

        Returns:
            bool: True if architecture is supported
        """
        pass

