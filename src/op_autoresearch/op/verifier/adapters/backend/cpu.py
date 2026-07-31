"""CPU backend adapter."""

from typing import Optional, Any

from op_autoresearch.op.utils.arch_normalize import CPU_ARCH_PATTERN

from .base import BackendAdapter


class BackendAdapterCpu(BackendAdapter):
    """Adapter for CPU backend."""

    def setup_environment(self, device_id: int, arch: str) -> None:
        """Setup CPU environment (no-op)."""
        pass

    def synchronize(self) -> None:
        """Synchronize CPU (no-op)."""
        pass

    def get_profiler(self) -> Optional[Any]:
        """Get CPU profiler (time-based)."""
        return None

    def get_device_string(self, device_id: int) -> str:
        """Get CPU device string."""
        return "cpu"

    def validate_arch(self, arch: str) -> bool:
        """Validate CPU architecture."""
        return bool(CPU_ARCH_PATTERN.match((arch or "").lower()))
