---
name: pypto-basics
description: "PyPTO Programming Principles and Core Mode"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "all"
---

# PyPTO Programming Principles

## Principle 0: Semantic contract drawn first from Torch baseline

Before writing PyPTO, transform baseline `forward` into an "enforceable semantic contract" before achieving and optimizing it.

Do at least five things:
1. Write a mathematical form (who is involved in the calculation, who is the subject of the regulation, what is the definition of output).
2. Check API syntax instead of variable names (`input/target/prediction`s are often misleading).
3. (c) Reinforce the contract (`sum`, `batchmean`, or `mean` syntax = `sum/count`, and specify the axes of the statute and `keepdim`.
4. Asymmetric determination (`A ' )||B` with `B '||A`Equivalence.
5. Do a 1 group semantic self-check sample (priority asymmetric input, a priori checking whether the direction is correct).

N. B. The performance rules (tie/loop) can only be applied after a semantic contract has been established and cannot be reversed.

## Principle 1: Static Shape

kernel.**Plant Functions**Encapsulating Kernel, Shape and scalar Parameters (eps, Slope, etc.) are sent in as closed packages. Forward is sent directly to Torch. Tensor.**ModelNew._init_signature must be identical to the original Model**, Shape obtained in Forward.

Supplementary (high priority):
- `get_inputs/get_init_inputs` of benchmark is a fixed parameter in a single mission; it is treated as a static contract and is not over-utilized.
- Start with "Standing Parameters": Read return value of this `get_init_inputs` from the title file as a constant for this task (e.g. `dim=1`).
- Annotation in the title, such as `Example, change to desired ...`, is considered a data set to describe noise, not a requirement for this realization.
- For fixed `dim` tasks, generate**a single fixed dim Kernel**; do not write `if dim == 0/1/2` branches in a Kernel.
- If you really want to support more than one dim, use multiple plant functions/ multiple kernels to create them separately, do not rub a polylingual into the same kernel.

## Principle 2: Forward function

1. **Assert**:`assert x.dim() == N`,`assert tuple(x.shape) == (...)`
2. **Reshape**(if required): torch reshape is a sape that can be handled by Kernel
3. **Call the kernel** with a contiguous tensor, then reshape the result to its original shape.

Forward prohibits torch calculations. Can not reshape insidekernel.
Output semantics (`keepdim`/Whether squeeze) are to be aligned directly with baseline; do not change semantics and replace them with extra `squeeze/unsqueeze`.

## Principle 3: Selecting dimensions

| operator Type | Forward Policy | kernel dimensions |
|----------|-------------|-------------|
| Elementwise / Simple Los | `reshape(-1)` | 1D |
| GroupNorm / InstanceNorm | `reshape(flat_batch, hidden)` | 2D |
| BatchNorm / RMSNorm | `reshape(B, C, -1)` | 3D |
| Batched matmul | Keep 3D, no loop | 3D |
| 2D Matmul | Hold 2D. | 2D |
| Single-axis Return | Keep original dimensions | Original |

Supplementary:
- Elementwise operator does not depend on data,**priority flatten is 1D**; loop is only triggered by `auto_tiles > 2048` and is not relevant to dimensions.
- The only reason for maintaining the high dimensions is business semantics (e.g. subsequent operator dependent layout) rather than tile constraints.

Supplementary:
- If the multiaxis statutes are continuous and the intermediate results are not used by other operators, priority is given to merging them into a single axis (e.g. `H,W -> HW`).

## Principle 4: tile double bound

1. `prod(tile_shape)` ≤ 16384
2. `auto_tiles = prod(ceil(shape[i]/tile[i]))` ≤ 2048

The number of tile parameters = rank of operated tensor. Usual:
- 1D: `(8192)` | 2D: `(1, 16384)` | 3D: `(1, 1, 16384)` or `(1, 16, 256)`
- matmul: `set_cube_tile_shapes([128, 128], [32, 128], [256, 256], True, False)`
- **Core red line**: Shape, tile, BLONK constant (e.g. 16384 /8192/4096)**in the Skill/example.**Reference only.**Recalculation must be made to the input dimension of the current task,**direct copying is prohibited.**
- **Empirical rule**: priority should be given to avoiding the obvious waste of parameters for `tile[i] > shape[i]`; where use is necessary, there must be clear justification (e.g., measured proceeds).

## Principle 5: Operator rules

`+` `*`: scalar at any location.**`-` `/`:tensor has to be left.**`1.0 - x` crash → uses `x * (-1.0) + 1.0`. The function calls the first parameter and must Tensor.

## Principle 6: Loop

The loop scenario: auto_tiles > 2048 (Prior Reasons)/ Large-Range Axis / Matmul M axis (set first at a logarithmic scale `loop_count`, often tested first `16/32` and then pushed `BASIC_BATCH=ceil_div(m, loop_count)` back; if necessary, expanded to `8/64`).**Without embedding, follow the outer axis.**View Shape can only be used for compilation periods constants.

---

# Core programming mode

**Module name `pypto`**(not `pyto`).

```python
import os, pypto, torch
_PYPTO_RUN_MODE = int(os.getenv("OP_AUTORESEARCH_PYPTO_RUN_MODE", "0"))
_PYPTO_RUNTIME_DEBUG_MODE = int(os.getenv("OP_AUTORESEARCH_PYPTO_RUNTIME_DEBUG_MODE", "0"))
```

## Mode A: Elementwise

**Critical judgement: `auto_tiles = prod(ceil(shape[i]/tile[i]))` must be ≤ 2048.**More than that must be loop.

**Priority policy**: Elementwise does not depend on data,**Priority flatten is 1D**; loop is triggered only by `auto_tiles > 2048` and not by dimensions.

**Small Matrix**(after flatten) auto_tiles ≤ 2048) `reshape(-1)` → 1D Kernel.
**Large matrix**(e.g. 16384 × 4096, after flatten auto_tiles > 2048):**1D + loop**(or 2D + loop only if semantics need to be maintained 2D).

```python
# Small matrix: ELU (16 × 16384 = 262144, tile 8192, auto_tiles = 32)
def create_elu_kernel(flat_size, alpha):
    @pypto.frontend.jit(...)
    def kernel(x: pypto.Tensor((flat_size,), pypto.DT_FP32)) -> ...:
        output = pypto.tensor([flat_size], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(8192)
        pos = pypto.maximum(x, 0.0)
        neg = pypto.minimum(x, 0.0)
        output[:] = pos + (pypto.exp(neg) - 1.0) * alpha
        return output
    return kernel

# Large Matrix 1D version: scalar mul (16384 × 4096, later loop)
def create_scalar_mul_kernel_1d(m, n, s):
    flat_size = m * n
    TARGET_LOOP_COUNT = 32
    BASIC_BATCH = ceil_div(flat_size, TARGET_LOOP_COUNT)
    num_iters = ceil_div(flat_size, BASIC_BATCH)
    @pypto.frontend.jit(...)
    def kernel(a: pypto.Tensor((m, n), pypto.DT_FP32)) -> ...:
        a_flat = pypto.view(a, [flat_size], [0])
        c = pypto.tensor([flat_size], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(8192)
        for bi in pypto.loop(0, num_iters, 1, name="LOOP", idx_name="bi"):
            off = bi * BASIC_BATCH
            chunk = pypto.view(a_flat, [BASIC_BATCH], [off])
            pypto.assemble(chunk * s, [off], c)
        return pypto.view(c, [m, n], [0])
    return kernel

# Large Matrix 2D Version: Use only when semantics need to keep 2D layouts
def create_scalar_mul_kernel_2d(m, n, s):
    TARGET_LOOP_COUNT = 32
    BASIC_BATCH = ceil_div(m, TARGET_LOOP_COUNT)
    num_iters = ceil_div(m, BASIC_BATCH)
    @pypto.frontend.jit(...)
    def kernel(a: pypto.Tensor((m, n), pypto.DT_FP32)) -> ...:
        c = pypto.tensor([m, n], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(8, 4096)
        for bi in pypto.loop(0, num_iters, 1, name="LOOP", idx_name="bi"):
            off = bi * BASIC_BATCH
            chunk = pypto.view(a, [BASIC_BATCH, n], [off, 0])
            pypto.assemble(chunk * s, [off, 0], c)
        return c
    return kernel
```

## Mode B: Matmul

- 2D: Follow M-dimensional loop (prefer the mid-point `loop_count`, `1~128`, often starting from `16/32`, then inverting `BASIC_BATCH`), `view → matmul → assemble`
- Convert B:`pypto.matmul(a_chunk, b, pypto.DT_FP32, b_trans=True)`
- 3D watched matmul: `c[:] = pypto.matmul(a, b, pypto.DT_FP32)`, no loop
- Triangular/symmetric/Diagonal Matrix = Standard matmul

## Mode C: Norm + Loop

**GroupNorm / InstanceNorm**(2D):reshape `(flat_batch, hidden_size)`,tile `(1, 16384)`,loopAlongflat_batch.`var = sq_sum * inv_hidden - mean * mean`.

**RMSNom / BatchNom**(3D): `(B, C, S)`, tile `(1, 1, 16384)` or `(1, 16, 256)`.
- RMSNorum: loop along the watch, `sum(dim=1)` unified along the C-axis
- BatchNolm: loop along Channel, `sum(dim=0)` + `sum(dim=2)` cross-batch and spatial
- Application of the RMS Norm rule: If RMS Norm is contracted along `C`'s axis and `C`'s is medium (e.g. 64), it is guaranteed to carry the standard continuously and then compare it in `C`'s axial candidate (often `16/32/64`).

## Mode D: Los → scalar

Simple loss (MSE/Hinge/KLDiv) flatten to 1D. Per-sample loss (Triplet/Cosine) maintains 2D, two tiles.
- A "directive sensitive/asymmetric" loss (dispersion class, certain probability distance classes): First matching a semantic contract, then writing a formula.
- Example (example only): `F.kl_div(torch.log(pred), target, reduction='batchmean')` should be
  `KL(target || (d) `, and in accordance with the `batchmean ' Statute.

## Mode E: Large tensor global return

`pypto.zeros` presser + loop segment + `acc[:] = acc + part`.

```python
def ceil_div(a, b):
    return (a + b - 1) // b
```
