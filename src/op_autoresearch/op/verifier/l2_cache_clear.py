"""
L2 Cache clears the module.

Provides NPU L2 Cache clearance to ensure that measurements are not affected by caches during performance tests.

Two types of clearance are supported:
1. Triton_ascend: Use a special Triton Kernel (recommended, accurately filtered)
2. Other DSL: Use tensor.zero_() (fallback, risk of error)
"""

from typing import Literal, List

# latency import torch/torch_npu to avoid triggering aclInit conflict in mindspore settings
_torch = None
_torch_npu = None


def _ensure_torch():
    global _torch, _torch_npu
    if _torch is None:
        import torch
        import torch_npu
        _torch = torch
        _torch_npu = torch_npu
    return _torch, _torch_npu

# Try importing triton (possibly not installed)
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# L2 Cache clears associated constants
L2_CACHE_SIZE_DEFAULT = 192 * 1024 * 1024  # 192MB Default
L2_CACHE_CLEAR_KERNEL_NAME = "OP_AUTORESEARCH_l2cache_clear"  # Special kernel name for filtering

# DSL Type Definition
DslType = Literal["triton_ascend", "triton_cuda", "torch", "tilelang_npuir", "ascendc", "other"]

# ============================================================================
# L2 Cache Warning Collection (backside suppress_output)
# ============================================================================

_l2_cache_warnings: List[str] = []


def _add_l2_cache_warning(message: str):
    """Add warning messages to collection list (twirl suppress_output)"""
    _l2_cache_warnings.append(message)


def get_l2_cache_warnings() -> List[str]:
    """Fetch all collected L2 Cache warning messages"""
    return _l2_cache_warnings.copy()


def clear_l2_cache_warnings():
    """Empty all collected L2 Cache alerts"""
    global _l2_cache_warnings
    _l2_cache_warnings = []


# ============================================================================
# L2 Cache size test
# ============================================================================

_l2_cache_size_detected = None


def _get_l2_cache_size(device_id: int = 0) -> int:
    """
    Retrieving L2 size from NPU device properties.

    Args:
        Divity_id: NPU device ID

    Returns:
        Int: L2 Cache Size (bytes)
    """
    global _l2_cache_size_detected

    if _l2_cache_size_detected is not None:
        return _l2_cache_size_detected

    try:
        _, torch_npu = _ensure_torch()
        device_props = torch_npu.npu.get_device_properties(device_id)
        l2_size = getattr(device_props, 'L2_cache_size', None)
        if l2_size is not None and l2_size > 0:
            _l2_cache_size_detected = l2_size
            return l2_size
    except Exception:
        pass

    _l2_cache_size_detected = L2_CACHE_SIZE_DEFAULT
    return _l2_cache_size_detected


# ============================================================================
# Get core number
# ============================================================================

_core_nums_cache = None


def _get_core_nums(vec_default=40, cube_default=20):
    """
    Get NPU core number (VEC + CUBE).

    Accessed through triton runtime API, resulting in caches avoiding calls.

    Returns:
        tuple[int, int]: (vec_core_num, cube_core_num)
    """
    global _core_nums_cache

    if _core_nums_cache is not None:
        return _core_nums_cache

    vec, cube = vec_default, cube_default
    try:
        import torch
        import triton
        device = torch.npu.current_device()
        properties = triton.runtime.driver.active.utils.get_device_properties(device)
        vec = properties.get("num_vectorcore", vec_default)
        cube = properties.get("num_aicore", cube_default)
    except Exception:
        pass

    _core_nums_cache = (vec, cube)
    return _core_nums_cache


# ============================================================================
# Triton-Asend dedicated L2 Cache Clear Kernel (module level definition)
# ============================================================================

# Define triton kernel at the module level, avoid JIT compilation domain problems
# Performance optimization: use large BLONK_SIZE to reduce the number of cycles and increase the utilization of bandwidth
if _TRITON_AVAILABLE:
    @triton.jit
    def OP_AUTORESEARCH_l2cache_clear(
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        CORE_NUM: tl.constexpr,
    ):
        """
        Specialized L2 Cache clears Kernel.

        Force the updating of L2 Cache by writing to a big buffer.
        Kernel is called OP_AUTORESEARCH_l2cache_clar, which allows for identification and filtering in profiller results.

        Use the stagger cycle, grid size equal to core number, and refer to triton-ascend to prepare the norm.
        Large BLONK_SIZE ensures high bandwidth utilization to reduce recycling costs.
        """
        pid = tl.program_id(0)

        # Calculate the total number of blocks
        num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

        # Intersect cycle processing: per core processing pid, pid+CORE_NUM, pid+2*CORE_NUM,... block
        # So all the data will be processed exactly once, and the load will be balanced.
        for block_idx in range(pid, num_blocks, CORE_NUM):
            block_start = block_idx * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            # Writing 0 Values to Clear Cache
            tl.store(output_ptr + offsets, tl.zeros([BLOCK_SIZE], dtype=tl.int32), mask=mask)


# ============================================================================
# L2 Cache Buffer Management
# ============================================================================

_l2_cache_buffer = None


def _get_l2_cache_buffer(device_id: int = 0):
    """
    Get a PyTorch version of the Buffer for clearing L2 Cache.
    Use inert initialization to avoid duplicate distribution of memory.

    Args:
        Divity_id: NPU device ID

    Returns:
        Torch. Tensor: Int32 tensor that covers the size of L2 Cache
    """
    global _l2_cache_buffer

    if _l2_cache_buffer is None:
        torch, _ = _ensure_torch()
        l2_size = _get_l2_cache_size(device_id)
        n_elements = l2_size // 4
        _l2_cache_buffer = torch.empty(n_elements, dtype=torch.int32, device='npu')

    return _l2_cache_buffer


# ============================================================================
# L2 Cache Clear Function
# ============================================================================

def clear_l2_cache_triton():
    """
    Clear L2 Cache with triton-ascend Kernel.

    This is the recommended method of clearance because:
    1. Use a unique kernel name (OP_AutoRESEARCH_l2cache_clar) for precise identification and filtering in profiller
    2. Avoid confusion with zeros/zero_operating in user code

    Performance optimization:
    - Use large BLONK_SIZE (32768) to reduce the number of cycles
    - Grid size equals the VEC core, making full use of parallelity
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not available for L2 cache clearing")

    torch, _ = _ensure_torch()
    buffer = _get_l2_cache_buffer()
    n_elements = buffer.numel()

    core_num, _ = _get_core_nums()

    BLOCK_SIZE = 32768

    grid = (core_num,)

    OP_AUTORESEARCH_l2cache_clear[grid](buffer, n_elements, BLOCK_SIZE=BLOCK_SIZE, CORE_NUM=core_num)
    torch.npu.synchronize()


def clear_l2_cache_zero():
    """
    Use tensor.zero_() to clear L2 Cache (fallback method).

    Warning: This method will be recorded as \"ZerosLike\" type in profiler.
    If zeros_like/zero_() are also used in the user code, this may lead to wrong filtering.
    """
    torch, _ = _ensure_torch()
    buffer = _get_l2_cache_buffer()
    buffer.zero_()
    torch.npu.synchronize()


# ============================================================================
# MindSpore version L2 Cache Buffer Management
# ============================================================================

_l2_cache_buffer_ms = None


def _get_l2_cache_buffer_ms():
    """
    Get the MindSpore version of the buffer used to clean L2 Cache.
    Use inert initialization to avoid duplicate distribution of memory.

    AscendDeviceProperties of MindSpore does not provide L2_cache_size.
    So use the default L2_CACHE_SIZE_DEFAULT.

    Returns:
        Mindspore. Tensor: int32 tensor for L2 size
    """
    global _l2_cache_buffer_ms

    if _l2_cache_buffer_ms is None:
        import mindspore as ms
        n_elements = L2_CACHE_SIZE_DEFAULT // 4
        _l2_cache_buffer_ms = ms.ops.zeros((n_elements,), dtype=ms.int32)

    return _l2_cache_buffer_ms


def clear_l2_cache_zero_ms():
    """
    Use MindSpore tensor.zero_() to clear L2 Cache.

    Warning: This method will be recorded as \"ZerosLike\" type in profiler.
    If zeros_like/zero_() are also used in the user code, this may lead to wrong filtering.
    """
    import mindspore as ms
    buffer = _get_l2_cache_buffer_ms()
    buffer.zero_()
    ms.runtime.synchronize()


def clear_l2_cache(dsl: DslType = "other", framework: str = "torch"):
    """
    Clears the unified entry function for L2 cape.

    Args:
        dsl: DSL type, determine which method to remove
             - \"triton_aspend\": with a dedicated triton kernel (recommended, torch framework only)
             - Other: use tensor.zero_() (fallback)
        ramework: framework type (\"toch\" or \"mindspore\"), determine which tensor interface to use

    Returns:
        None
    """
    if framework == "mindspore":
        clear_l2_cache_zero_ms()
        return

    if dsl == "triton_ascend":
        try:
            clear_l2_cache_triton()
        except Exception as e:
            _add_l2_cache_warning(
                f"[L2 Cache] Triton kernel call failed ({e}), falling back to zero_() method. "
                "Results may have false positive filtering risk."
            )
            clear_l2_cache_zero()
    else:
        if not hasattr(clear_l2_cache, '_warned_for_dsl'):
            clear_l2_cache._warned_for_dsl = set()

        if dsl not in clear_l2_cache._warned_for_dsl:
            _add_l2_cache_warning(
                f"[L2 Cache] Current DSL ({dsl}) has no dedicated L2 cache clear method. "
                "Using tensor.zero_() for clearing. "
                "Note: This will be recorded as 'ZerosLike' type in profiler. "
                "If the target operator also uses zeros_like/zero_(), timing may be inaccurate. "
                "For precise results, please analyze the specific operator manually."
            )
            clear_l2_cache._warned_for_dsl.add(dsl)

        clear_l2_cache_zero()
