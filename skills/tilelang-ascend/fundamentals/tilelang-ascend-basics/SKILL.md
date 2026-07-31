---
name: tilelang-ascend-basics
description: "TileLang Ascend code basics: code templates, function design principles, dimension parameters self-guided, integration of operator workspace configuration mode, host pre-processing statements. All operator generation tasks must be based on a basic code agreement."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
---

# TileLang Ascend Code Foundation

---

## Standard kernel structure and start-up function template

### Code Template

TileLang Ascend operator consists of three layers of nested:

```python
@tilelang.jit(out_idx=[...], pass_configs=...)
def kernel(Parameters for compilation period):
    @T.prim_func
    def main(Run-timetensorParameters):
        ...  # Calculating Logic
    return main

class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()

    def forward(self, *inputs):
        kernel_func = kernel(Parameters for compilation period)  # I don't think so.1Sub-calls: uploading the translation period parameters, returning the executable function
        outputs = kernel_func(*inputs)  # I don't think so.2Sub-call: Incomingtensor
        return outputs
```

### `@jit(out_idx=[...])` call protocol

- **Input out_idx**: automatically allocate output tensor and**return**, not written in situ

```python
outputs = kernel_func(*inputs)
```

- **Not imported out_idx**: output from host side tensor written in situ

```python
output_i = torch.empty/empty_like/zeros/zeros_like()
kernel_func(*inputs, *outputs)
```

### Size variable should be defined at @jit level, not inside @T.prim_func

The `@jit` layer variable has been replaced by a specific value when compiled, the variable in `@T.prim_func` will be retained as a symbol expression by TVM, and errors will be reported in subsequent calculations in the guide.

```python
@jit(...)
def kernel(M, N, block_M, block_N, dtype):
    # Correct approach: define at @jit level to push TVM to a specific value
    sub_block_M = block_M // 2
    m_num = M // block_M

    @T.prim_func
    def main(A, C):
        # Error: Internal definition in @T.prim_func
        # sub_block_M = block_M // 2

        buf = T.alloc_shared((sub_block_M, block_N), dtype)
        T.tile.mul(buf, buf, -1.0)
```

## Design operator

### Core decision-making
- **Programming mode selection**: Devloper / Express / Mixed mode
- **API Map**: Disassembly mathematical formulas into TileLang DSL original language combinations
- **Memory Level Planning**: GM → L1/ UB → L0 Data handling path
- **Tiling Policy**: Block Division and Tile Shape Design
- **Cycle structure**: T. Parallel / T. Serial / T. Pipelined / T. Persistent selection
- **Sync Policy**: AutoSync vs ManualSync

### Known Limits

| Constraints | Annotations | Impact | Alternatives |
|------|------|------|----------|
| **Not supported 3D Kernel** | `T.Kernel` accepts only 1D block number | We can't do 3D in parallel. | Use `block_metadata` projection mechanism |
| **threads parameter limits** | Only 1 or 2 supported, no large value supported | `threads=128` etc. error in design | Default does not specify threads or set to 2 |
| **Dynamic circular boundaries not supported** | The number of loops cannot depend on tensor values (e. g. `batch_sizes[bz]`) | `T.Pipelined(batch_sizes[bz])` Error | The maximum number of cycles is projected, as judged by `T.serial(max_iters)` + conditions |
| **pipeline does not support dynamic boundaries** | `T.Pipelined` cycles must be static | Dynamic batch cannot pipeline | Change to `T.serial` or expect to fix the number of iterations |
| **Part of GPU API not available** | CUDA-specific API does not exist in Ascend | Failed to directly port GPU code | See API chapter confirmation Ascend API |
| **GEMM Request M,N is block integer** | `M // block_M` block-depend; open at zero block at `M < block_M` | Output Zero or Zero Collapse | Must explicitly address the policy: post side padding+corp or Kernel dynamic block |
| **L0C capacity cap** | A2/A3 device L0C = 128KB | `block_M × block_N × sizeof(accum) > 128KB` leads to segfault | Meet `block_M × block_N ≤ 16384` (faat32acum) when designing block |

## Key code instruction

### GEMM operator: non-integrated dimension processing

GEMM Kernel uses `M // block_M` and `N // block_N` internally, requesting M and N to double the size of block. Non-incorporation needs to be cropped after the Python Layer zero-pacing:

```python
# padding
M_pad = ((M + block_M - 1) // block_M) * block_M
N_pad = ((N + block_N - 1) // block_N) * block_N
K_pad = ((K + block_K - 1) // block_K) * block_K

if M_pad > M or K_pad > K:
    kernel_padded = torch.zeros(M_pad, K_pad, ...)
    kernel_padded[:M, :K] = kernel_flat

# GEMM Post Crop
output = output[:M, :N]
```

**Key constraints**: `M // block_M = 0` (when M < block_M) does not padding will cause a zero block start (out of all output) or a zero-coding collapse.

### Autotune operator: protocol_prog interface with get_configs

- **`supply_prog(params)`**: `params` only contains the input tensor description (excluding output param). The dimensions from `params[0].shape` / `params[1].shape` are not accessible to `params[2]`.
- **`get_configs` as callable**: autotuner calls in `get_configs(key_args_tuple, key_kwargs_tuple)` and must be signed as `get_configs(key_args, _key_kwargs=None)`, extracting M/N/K from `key_args`.
- **config filter**: Invalid combinations of `block > dimension` (avoiding error in zero compilation) and `block_M * block_N * sizeof(accum) > L0C_capacity` (avoiding L0C spills segfault) must be filtered in `get_configs`.

### Buffer Allocation

```python
# VEC_NUM = 2, each vector nuclear processing block_M / / VEC_NUM line
a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
```

### Data Moving Index

```python
# Standard index mode
row_start = bx * block_M + vid * block_M // VEC_NUM
T.copy(A[row_start, by * block_N], a_ub)
T.copy(a_ub, B[row_start, by * block_N])
```

### Sync

```python
# Express Mode: Manual Synchronization
with T.Scope("V"):
    T.copy(A[...], a_ub)
    T.barrier_all()
    T.tile.exp(a_ub, a_ub)
    T.barrier_all()
    T.copy(a_ub, B[...])

# Devloper Mode + AutoSync: no manual barrier
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

### Radio

```python
# Recursive result [M, 1] broadcast to [M, N]
max_ub = T.alloc_ub([block_M // VEC_NUM, 1], dtype)
max_2d_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
T.tile.broadcast(max_2d_ub, max_ub)
```
