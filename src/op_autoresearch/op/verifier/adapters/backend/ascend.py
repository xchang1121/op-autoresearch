"""Ascend backend adapter."""

import torch
from typing import Optional, Any

from op_autoresearch.op.utils.config_utils import is_supported_arch
from .base import BackendAdapter


class BackendAdapterAscend(BackendAdapter):
    """Adapter for Ascend backend."""

    def setup_environment(self, device_id: int, arch: str) -> None:
        """Setup Ascend environment variables."""
        import os
        os.environ['DEVICE_ID'] = str(device_id)

    def synchronize(self) -> None:
        """Synchronize Ascend device."""
        try:
            torch.npu.synchronize()
        except AttributeError:
            # If torch_npu is not available, skip synchronization
            pass

    def get_profiler(self) -> Optional[Any]:
        """Get Ascend profiler (msprof)."""
        # Profiler is handled in kernel_verifier, not here
        return None

    def get_device_string(self, device_id: int) -> str:
        """Get Ascend device string."""
        return f"npu:{device_id}"

    def validate_arch(self, arch: str) -> bool:
        """Validate Ascend architecture."""
        return is_supported_arch("ascend", arch)
