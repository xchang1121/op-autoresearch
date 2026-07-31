"""CUDA backend adapter."""

import torch
from typing import Optional, Any

from op_autoresearch.op.utils.arch_normalize import CUDA_ARCH_PATTERN

from .base import BackendAdapter


class BackendAdapterCuda(BackendAdapter):
    """Adapter for CUDA backend."""

    def setup_environment(self, device_id: int, arch: str) -> None:
        """Setup CUDA environment variables."""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    def synchronize(self) -> None:
        """Synchronize CUDA device."""
        torch.cuda.synchronize()

    def get_profiler(self) -> Optional[Any]:
        """Get CUDA profiler (nsys)."""
        # Profiler is handled in kernel_verifier, not here
        return None

    def get_device_string(self, device_id: int) -> str:
        """Get CUDA device string."""
        return f"cuda:{device_id}"

    def validate_arch(self, arch: str) -> bool:
        """Validate CUDA architecture."""
        return bool(CUDA_ARCH_PATTERN.match((arch or "").lower()))
