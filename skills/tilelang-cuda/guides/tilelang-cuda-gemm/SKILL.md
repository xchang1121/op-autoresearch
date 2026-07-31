---
name: tilelang-cuda-gemm
description: "TileLang CUDA GEMMCompleteness optimizes guidance, covering base templates,Swizzling,L2 Cache Rasterization,Auto-Pipelining,Persistent Kernel,Autotuning,Split-K,Stream-K,Fine-grained MMAand so on.tilelangOfficialexamples/gemm/ best practice. For allmatrix multiplicationand its variant ()BMM,FP8 GEMM,Int4 GEMM,Dequant GEMM) kernel code generation and optimization"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
  operator_patterns: "matmul, bmm, fp8_gemm, int4_gemm, dequant_gemm"
structure:
  optimization_levels:
    level1: "basic_gemm"
    level2: "swizzling_rasterization"
    level3: "autotuning_persistent"
    level4: "splitk_streamk"
    level5: "fine_grained_mma"
---

# TileLang GEMM Performance Optimization Full Guide

Ben Skill is organized on the basis of the tilelang official `examples/gemm/` and `docs/programming_guides/instructions.md`, covering all GEMM optimization techniques from base to high end.

## 1. Basic GEMM Templates

The core model for all GEMMs is as follows:

```python
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm
```

### Key concepts

1. **Memory level**
   - `T.alloc_shared`: shared memory (__shared__)
   - `T.alloc_fragment`: Storer Snippets (Tensor Core local)
   - Data stream: Global → Shared → Fragment (GEMM) → Shared → Global

2. **Recommended value for block parameters**
   - `block_M, block_N = 128, 128` (classic), `256, 256` (large matrix), `128, 256` (asymmetric)
   - `block_K = 32` (recommended default), `64` (SM90+)
   - `threads = 128` (classic GEMM value), `256` (some scenarios)
   - `num_stages = 3` (recommended), `2` (minimal shared memory), `4` (SM90+)

3. **Cumulative accuracy**
   - `accum_dtype=T.float32` guarantees accuracy, especially for float16 input
   - FP8 scene requires `accum_dtype=T.float32` + 2xaccumulate mode

## 2. Advanced Optimization: Swizzling + Rasterization + Parallel Copy

The following optimized combinations can significantly enhance GEMM performance:

```python
import tilelang.language as T
from tilelang.cuda.intrinsics import make_mma_swizzle_layout as make_swizzle_layout


@tilelang.jit(out_idx=[-1])
def matmul_optimized(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            # Optimize 1: Swizzle shared memory playout → avoid bank condition
            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            # Optimizing 2: Rasterization → Increase L2 Cache Hit Rate
            T.use_swizzle(panel_size=10, enable=True)

            T.clear(C_local)
            for idx in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, idx * block_K], A_shared)

                # Optimization 3: Parallel copy B → multi-line load
                for ko, j in T.Parallel(block_K, block_N):
                    B_shared[ko, j] = B[idx * block_K + ko, bx * block_N + j]

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main
```

### Optimization and detail.

| Optimizing technology | Role | When to use |
|----------|------|----------|
| `T.annotate_layout` | shared memory Swizzling, avoid bank compliance | Any GEMM |
| `T.use_swizzle` | Grid-level L2 Cache grid | Large Matrix M,N > 2048 |
| `T.Parallel(..., ...)` copy | Multi-line parallel global → load | When `T.copy` is a bottleneck |
| Middle `C_shared` buffer | Avoid fragment→global writing directly | When Swizzle Layout Requires |

### ⚠ ️ note.

- `T.annotate_layout` swizzle playout requires `from tilelang.cuda.intrinsics import make_mma_swizzle_layout`
- Panel_size control particle size for `T.use_swizzle(panel_size=10)`, 10 is the recommended default
- When using Parallel copy, the index of `B_shared` must contain the complete map of `bx`, `ko`, `j`

## 3. Autotuning

tilelang provides `@tilelang.autotune` and `AutoTuner` for automatic optimization:

### 3.1 Manual search space

```python
import tilelang as tl
import tilelang.language as T
import torch
from tilelang.autotuner import AutoTuner


def ref_program(A, B):
    return A @ B.T


def get_configs(M, N, K):
    """Generate modulated search space"""
    import itertools
    block_M_list = [64, 128, 256]
    block_N_list = [64, 128, 256]
    block_K_list = [32, 64]
    num_stages = [0, 1, 2, 3]
    thread_num = [128, 256]
    enable_rasterization = [True, False]

    _configs = list(itertools.product(
        block_M_list, block_N_list, block_K_list,
        num_stages, thread_num, enable_rasterization,
    ))
    return [
        {
            "block_M": c[0], "block_N": c[1], "block_K": c[2],
            "num_stages": c[3], "thread_num": c[4],
            "enable_rasteration": c[5],
        }
        for c in _configs
    ]


def autotune_gemm(M, N, K):
    def kernel(block_M=None, block_N=None, block_K=None, num_stages=None,
               thread_num=None, enable_rasteration=None):
        dtype = T.bfloat16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])
        return main

    autotuner = AutoTuner.from_kernel(
        kernel=kernel, configs=get_configs(M, N, K)
    ).set_compile_args(
        out_idx=[-1], target="auto",
    ).set_profile_args(
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program, skip_check=False, backend="event",
    )
    return autotuner.run(warmup=3, rep=20)
```

### 3.2 Search using Roller (BitBLAS) smart

```python
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA


def get_roller_configs(M, N, K, topk=20):
    arch = CUDA("cuda")
    carve_template = MatmulTemplate(
        M=M, N=N, K=K,
        in_dtype=T.float16, out_dtype=T.float16, accum_dtype=T.float32,
    ).with_arch(arch)
    roller_hints = carve_template.recommend_hints(topk=topk)
    if roller_hints is None:
        raise ValueError("No Roller Hints Found for TensorCore Scheduling")
    configs = []
    for hint in roller_hints:
        block_m, block_n = hint.block
        warp_m, warp_n = hint.warp
        block_rows, block_cols = block_m // warp_m, block_n // warp_n
        configs.append({
            "block_M": block_m, "block_N": block_n,
            "block_K": hint.rstep[0],
            "num_stages": hint.pipeline_stage if hint.pipeline_stage > 1 else 0,
            "thread_num": block_rows * block_cols * 32,
            "enable_rasteration": hint.rasterization_plan is not None,
        })
    return configs
```

### 3.3 Heuristic Config version of SM

```python
import torch


def get_heuristic_config():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, _ = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10
    if sm_version == 80:  # A100
        return {"block_M": 128, "block_N": 256, "block_K": 32,
                "num_stages": 2, "thread_num": 128, "enable_rasteration": True}
    elif sm_version == 90:  # H100
        return {"block_M": 128, "block_N": 256, "block_K": 64,
                "num_stages": 3, "thread_num": 256, "enable_rasteration": True}
    else:  # Default
        return {"block_M": 128, "block_N": 256, "block_K": 32,
                "num_stages": 0, "thread_num": 128, "enable_rasteration": True}
```

## 4. Persistent Kernel

Persistent Kernel applies to a large number of output tile scenes, reducing kernel lanch overhead by limiting the grid to `sm_num` block:

```python
import tilelang.language as T
from tilelang.carver.arch import driver


@tilelang.jit(out_idx=[-1])
def matmul_persistent(M, N, K, block_M, block_N, block_K, threads, num_stages,
                      dtype=T.float16, accum_dtype=T.float32):
    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    group_size = 8

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(10)

            for w in T.serial(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[bx * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, by * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)

                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return main
```

### Persistent Kernel vs. T. Persistent

Tilelang also supports the original language of `T.Persistent` (more concise):

```python
    @T.prim_func
    def main_persistent(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            for bx, by in T.Persistent(
                [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                sm_num, block_id
            ):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[bx * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, by * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[bx * block_M, by * block_N])

    return main_persistent  # Return this version
```

### When to use Persistent Kernel

| scene | Recommendations | Reason |
|------|------|------|
| Large Matrix (M,N > 4096) | ✅ Yes | Reduce kernel lanch overhead and increase SM utilization |
| Small Matrix (M,N < 1024) | ❌ No | It's not worth the extra expenses. |
| Multi-CTA scene | ✅ Yes | 2-CTA policyr better for SM100+ |

## 5. Split-K

Sprit-K applies to the scene of K-dimensional extremes (the effect is obvious when K/M > 4):

```python
import tilelang.language as T


@tilelang.jit(out_idx=[2])
def matmul_splitk(M, N, K, block_M, block_N, block_K, split_k,
                  dtype=T.float16, accum_dtype=T.float32, out_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            T.clear(C_local) if bz == 0 else None

            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K // split_k, block_K), num_stages=3):
                T.copy(A[bz * (K // split_k) + by * block_M, k * block_K], A_shared)
                T.copy(B[bz * (K // split_k) + k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

### Sprit-K Points

- **Extra grid dimension**: `split_k` as 3D
- **Atomscumulation**: using `T.atomic_add` to merge parties
- **Applicable scene**: K > M, K > N (e.g. K = 16384, M = N = 1024)
- **Typical split_k value**: 2, 4, 8

## 6. Transpose B (B^T * AMode)

When shape for B is `(N, K)` instead of `(K, N)`:

```python
@tilelang.jit(out_idx=[-1])
def matmul_transpose_b(M, N, K, block_M, block_N, block_K,
                       dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10, enable=True)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main
```

**Keyword**: `transpose_B=True` Autoprocessing B conversion calculation.

## 7. FP8 GEMM

PP8 Use MMA commands instead of automatic T.gemm dispatch, need to be noted:

```python
import tilelang.language as T
from tilelang.utils import determine_fp8_type


@tilelang.jit(out_idx=[-1])
def matmul_fp8(M, N, K, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
    @T.prim_func
    def gemm_fp8(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_fp8


# Call
# dtype = determine_fp8_type()  # e4m3
# kernel = matmul_fp8(1024, 1024, 1024, 128, 128, 64, dtype)
# a = torch.randn(M, K).cuda().to(dtype)
# b = torch.randn(N, K).cuda().to(dtype)
# c = kernel(a, b)
```

### FP8 Elements

- **shape with `transpose_B=True`**: FP8 GEMM for B is `(N, K)`
- **dtype**: `determine_fp8_type()` returns e4m3; `determine_fp8_type("e5m2")` returns e5m2
- **Accumulated accuracy**: Always using `accum_dtype=T.float32`
- **Validation**: FP8 Validation with `calc_diff` instead of `torch.testing.assert_close`

## 8. Fine-grained MMA (particle size MMA)

When an automatic `T.gemm` does not satisfy the demand (e. g. dequantize GEMM needs to customize playout), use `TensorCoreIntrinEmitter`:

```python
from tilelang.intrinsics import TensorCoreIntrinEmitter, make_mma_swizzle_layout


def dequant_gemm(M, N, K, in_dtype, out_dtype, accum_dtype):
    micro_size_x = micro_size_y = micro_size_k = 16
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = 32
    shared_scope = "shared.dyn"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype, b_dtype=in_dtype, accum_dtype=accum_dtype,
        a_transposed=False, b_transposed=True,
        block_row_warps=block_row_warps, block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles, warp_col_tiles=warp_col_tiles, chunk=chunk,
    )

    @T.prim_func
    def main(A: T.Tensor((M, K), in_dtype),
             B: T.Tensor((N, K), in_dtype),
             C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=32 * block_row_warps * block_col_warps) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(
                (block_M // micro_size_x, block_N // micro_size_y, micro_size_x, micro_size_y),
                out_dtype, scope=shared_scope)
            A_local = T.alloc_local((block_row_warps * (micro_size_x * micro_size_k) // 32), in_dtype)
            B_local = T.alloc_local((block_col_warps * (micro_size_y * micro_size_k) // 32), in_dtype)
            C_local = T.alloc_local((
                block_row_warps * block_col_warps * (micro_size_x * micro_size_y) // 32), accum_dtype)

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })
            T.use_swizzle(panel_size=10)
            T.clear(C_local)

            for ko in T.Pipelined(K // block_K, num_stages=2):
                # Parallel copy from global to shared
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                # Fine-grained MMA
                for ki in T.serial(block_K // micro_size_k):
                    mma_emitter.ldmatrix_a(A_local, A_shared, ki)
                    mma_emitter.ldmatrix_b(B_local, B_shared, ki)
                    mma_emitter.mma(A_local, B_local, C_local)

            mma_emitter.stmatrix(C_local, C_shared)

            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x, j // micro_size_y, i % micro_size_x, j % micro_size_y]

    return main
```

### Fine-grained MMA Applied scene

- **Dequantize GEMM**(W4A8, FP4, Int4): Need to go straight after dequantize MMA
- **Custom playout**: Auto `T.gemm` 's playout does not meet demand
- **Polarity**: Need to fully control the Idmatrix/mma/stmatrix sequence

## 9. Profiling / Benchmark

### 9.1 Use Profiller Classes

```python
kernel = matmul(4096, 4096, 4096, 128, 128, 32)
profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

# correctness verification
profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)

# Evaluation
latency = profiler.do_bench(backend="event")    # CUDA Event (Default)
# # CUPTI programmer (more precise)
# latency = profiler.do_bench(backend="cudagraph")  # CUDA graph

# TFLOPs
M, N, K = 4096, 4096, 4096
tflops = 2 * M * N * K / latency * 1e-9
```

### 9.2 Use do_bench function

```python
from tilelang.profiler import do_bench

kernel = matmul(4096, 4096, 4096, 128, 128, 32)
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)

latency = do_bench(
    lambda: kernel(a, b),
    warmup=25, rep=100, backend="event", return_mode="min"
)
```

### 9.3 Benchmark Parameters Recommended

| Parameters | Value | Annotations |
|------|-----|------|
| `warmup` | 25ms | Warmup Target Time |
| `rep` | 100ms | Target time for evaluation |
| `backend` | "event" | Default, CUDA Event Time |
| `backend` | "cupti" | CUPTI programr, more precise |
| `backend` | "cudagraph" | CUDA graph replay |
| `return_mode` | "min" | Recommended. Minimal. |

## 10. Command speed check.

### Data Move

| Command | Purpose | Example: |
|------|------|------|
| `T.copy(src, dst)` | Sync Copy | `T.copy(A[...], A_shared)` |
| `T.async_copy(src, dst)` | Step Copy (cp.async) | `T.async_copy(A[...], A_shared)` + `T.ptx_wait_group(0)` |
| `T.tma_copy(src, dst)` | TMA Step Copy (SM90+) | `T.tma_copy(desc, A_shared)` |

### Memory Allocation

| Command | Purpose | Example: |
|------|------|------|
| `T.alloc_shared` | shared memory | `T.alloc_shared((128, 32), "float16")` |
| `T.alloc_fragment` | Can not open message | `T.alloc_fragment((128, 128), "float")` |
| `T.alloc_local` | Thread Local | `T.alloc_local((1,), "float32")` |
| `T.alloc_tmem` | Tensor Memory (SM100) | `T.alloc_tmem((128, 256), "float32")` |

### Calculate

| Command | Purpose | Example: |
|------|------|------|
| `T.gemm(A, B, C)` | Tile GEMM | `T.gemm(A_shared, B_shared, C_local)` |
| `T.gemm(A, B, C, transpose_B=True)` | B conversion mode | `T.gemm(A_s, B_s, C_l, transpose_B=True)` |
| `T.clear(buf)` | Clear. | `T.clear(C_local)` |
| `T.reduce_max/min/sum` | Return | `T.reduce_sum(input, output, dim=0)` |

### Cycle control

| Command | Purpose | Example: |
|------|------|------|
| `T.Pipelined(n, stages)` | Software pipeline | `for k in T.Pipelined(..., num_stages=3)` |
| `T.Parallel(d1, d2)` | Parallel Loop | `for i, j in T.Parallel(128, 128)` |
| `T.serial(n)` | Serial Loop | `for ki in T.serial(block_K // 16)` |
| `T.Persistent(...)` | Persistent loop | `for bx, by in T.Persistent([...], sm_num, block_id)` |

### Synchronization and barriers

| Command | Purpose |
|------|------|
| `T.sync_threads()` | Thread Block Sync |
| `T.ptx_wait_group(n)` | Waiting to make a copy of the walk |
| `T.mbarrier_wait_parity(barrier, parity)` | MBarrier, wait. |
| `T.warpgroup_arrive()` / `T.warpgroup_commit_batch()` / `T.warpgroup_wait(n)` | WMMA Sync |

### Note and Optimization

| Command | Purpose |
|------|------|
| `T.annotate_layout({buf: layout})` | Memory Layout Comment |
| `T.use_swizzle(panel_size, enable)` | L2 Scanning |
| `T.annotate_l2_hit_ratio(buf, ratio)` | L2 Cache Hint |

### Warp Operations

| Command | Purpose |
|------|------|
| `T.shfl_sync(value, src_lane)` | Radio |
| `T.shfl_down(value, delta)` | Move Down |
| `T.shfl_xor(value, delta)` | XOR Exchange |
| `T.warp_reduce_sum/max` | Warp Return |
| `T.ballot(predicate)` | Vote. |

### Atomic Operations

| Command | Purpose |
|------|------|
| `T.atomic_add(dst, val)` | Atom plus |
| `T.atomic_max/min(dst, val)` | Maximum/minimum atom |

## 11. Common Errors

1. **Forget `T.clear(C_local)`**→ cumulative waste values cause errors
2. **`num_stages` is too big**→ exceeds shared memory's limit
3. **`T.sync_threads()` in Conditional Branch**→ Deadlock
4. **FP8 failed or miscalculated without `transpose_B=True`**→
5. **`T.annotate_layout` missing `make_mma_swizzle_layout` import**→ Bank conflicts
6. **`T.copy` for incompatible scope**→ compilation failed
7. **Border check missing in Persistent Kernel `bx * block_M < M`**→ Cross-border access
8. **Split-K does not use atomic operation to merge results**→ data competition

## 12. Performance is improved, Checklist

- [ ] Block size: Start with `block_M=128, block_N=128, block_K=32`
- [ ] Cumulative accuracy: Use `accum_dtype=T.float32`
- [ ] Pipeline: Setup `num_stages=3`
- [ ] Swizzle: Add `T.annotate_layout` + `T.use_swizzle(panel_size=10)`
- [ ] Large Matrix (M,N > 4096): Try Persistent Kernel
- [ ] K > M,N: Try Split-K
- [ ] Final Modifier: Search for optimal configuration using `AutoTuner`
- [ ] Authentication: `profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)`
