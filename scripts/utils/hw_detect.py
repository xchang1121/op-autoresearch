"""Compat shim — hardware arch detection now lives in
:mod:`op_autoresearch.op.utils.hw_detect` (one probe implementation shared with the
CLI worker). Re-exported here so workspace scripts keep importing
``utils.hw_detect``.
"""

from op_autoresearch.op.utils.hw_detect import derive_arch, probe_hint  # noqa: F401
