---
name: triton-ascend-matmul
description: "Applicable tomatrix multiplication(matmul)Categoryoperator. WhenoperatorThe core calculation involves two or more dimensions.matrix multiplicationshould select this guide, typicallyoperatorIncluding:matmul, mm, bmm, linear, gemm, outer_product, einsum(With Matrix Multiplication), conv(Convert to Matrix Multiplication)and so on.Cube CoreUse, Segments(tiling)Strategy,SwizzleOptimization, largeKKey techniques such as dimensional processing. This does not apply to pure element-by-element or pure contractual operations.attentionof the MechanismQK^T and score*VMatrix times, ifoperatorThe whole thing's a focus calculation. It's a priority.attentionGuide."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "matmul"
---

# MatMul operator Optimization

> Applicable to matrix multiplication and related operations

## Core number selection (hard constraint)

- `tl.dot` / matrix multiplication Operations →**Must use CUBE_CORE_NUM**
- Mixed (matmul and elementwise after processing) →**CUBE_CORE_NUM**
- / scalar Operations → VEC_CORE_NUM

**Use VEC_CORE_NUM to start the matmul Kernel will result in a numerical result error.**

## Tile Size limit (hardware constraint)

MatMul data go L0A/L0B/L0C, tile size is limited by hardware storage capacity, exceeding which will result in `ub overflow`/ `cbuf overflow` compilation error.

A binding formula (specific capacity reference target hardware information document):
- L0A:`BLOCK_M × BLOCK_K × sizeof(dtype) ≤ L0ACapacity`
- L0B:`BLOCK_K × BLOCK_N × sizeof(dtype) ≤ L0BCapacity`
- L0C:`BLOCK_M × BLOCK_N × sizeof(acc_dtype) ≤ L0CCapacity`

`ub overflow` / `cbuf overflow` →**Shrink BLONK_M, BLONK_N or BLONK_K**

## Ascend backend Slice Optimization

**Key principles**: Full utilization of bandwidth, operator row width of 512B.

Take fp16/ bf16 for example (two bytes per element):

### Cut-off configuration (based on conversion)

1. **A, B do not change**
   - Branch widths K0 and N0 respectively
   - **Recommended: M0=128, K0=256, N0=256

2. **A no switch, B switch**
   - It's all about K-0.
   - **Recommended**: K0 = 256, M0 and N0

3. **A, B, all transferred**
   - The width of the segments is M0 and K0 respectively.
   - **Recommended: M0=256, K0=256, N0=128

4. **A conversion, B not conversion**
   - The widths of the segments are M0 and N0 respectively.
   - **Note: Left or right matrix cannot meet the integer multiple of 512B at the same time and needs to be adjusted to the actual situation

### Why 512B?

- 512B = 256 fp16/bf16 Element(256 ×2 bytes)
- Best bandwidth alignment for NPU
- Ensure that every memory access makes full use of bandwidth

## Fixed core number activated.

MatMul operator uses**CUBE core number**(matrix core calculation).

**Key**: Use `grid=(num_cores,)` instead of `(NUM_BLOCKS,)`

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Key: Start with fixed core numbers, each core has multiple blocks
    pid = tl.program_id(0)  # CoreID: 0~num_cores-1
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # Multiple blocks per core cycle
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # Calculating a 2D index to the current block
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        # Initialise accumulator
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # KD cycle
        for k in range(0, K, BLOCK_K):
            # Load Block A
            a_offset = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * K + \
                       (k + tl.arange(0, BLOCK_K))[None, :]
            a_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < M
            a = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0)

            # Load Block B
            b_offset = (k + tl.arange(0, BLOCK_K))[:, None] * N + \
                       (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
            b_mask = (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] < N
            b = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0)

            # Matrix multiplied by cumulative
            accumulator += tl.dot(a, b)

        # Storage Results
        c_offset = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * N + \
                   (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        c_mask = ((block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < M) & \
                 ((block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] < N)
        tl.store(c_ptr + c_offset, accumulator, mask=c_mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import torch
            import triton
            device = torch.npu.current_device()
            properties = triton.runtime.driver.active.utils.get_device_properties(device)
            self.CUBE_CORE_NUM = properties.get("num_aicore", 20)
        except:
            self.CUBE_CORE_NUM = 20

    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        num_cores = self.CUBE_CORE_NUM
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256

        matmul_kernel[(num_cores,)](
            a, b, c, M, N, K, num_cores,
            BLOCK_M, BLOCK_N, BLOCK_K
        )
        return c
```

**Key points**:
- Fixed start core number using `grid=(num_cores,)` (CUBE_CORE_NUM)
- Each core circulates multiple blocks through `for block_idx in range(pid, NUM_BLOCKS, num_cores)`
- Do not use `grid=(NUM_BLOCKS_M * NUM_BLOCKS_N,)` to start a program for each block