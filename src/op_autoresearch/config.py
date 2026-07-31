"""Small runtime configuration helpers shared by verifier and workers."""

import os
from typing import Optional


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a namespaced setting, with the unprefixed name as fallback."""
    return os.environ.get(f"OP_AUTORESEARCH_{name}", os.environ.get(name, default))

