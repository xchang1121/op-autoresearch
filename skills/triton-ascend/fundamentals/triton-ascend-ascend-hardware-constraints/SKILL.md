---
name: triton-ascend-ascend-hardware-constraints
description: "AscendHardware constraints andcompilerLimit speed check. CoverCUBE/VECStorage level budget calculation methodology,bishengIR compilerKnown limitations,strided accessPerformance characteristics. They apply to allTriton Ascend operatorGenerate and debug scenes."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "all"
---

# Ascend Hardware Containment and compiler Limit

> Depending on the specific size of the hardware for the different models, operator is generated with simultaneous input into the hardware information document, and the volume values in the formula below refer to that document.

## 1. Storage level budget

### CUBE Path (matmul / tl.dot)

Matmul data walk L0A/L0B/L0C,**without going through UB**:

| Buffer zone | Purpose | A binding formula |
|--------|------|---------|
| L0A | A file (m0 × k0) | 'm0 × k0 × size (A.dtype) ≤ L0A capacity ' |
| L0B | Right Matrix B file (k0 × n0) | `k0 × n0 × size (B.dtype) ≤ L0B capacity ' |
| L0C | Result C file (m0 × n0), supports add-up | `m0 × n0 × size (C.dtype) ≤ L0C capacity ' |

**Example of calculation**(in the case of a hardware L0A = 64KB):
- fp16 (2 bytes/elements): can accommodate 32K elements → BLONK_M=128, BLONK_K=256
- fp32 (4 bytes/elements): can accommodate 16K elements → BLONK_M=128, BLONK_K=128

When selecting the size of the tile, ensure that the three buffer zones are open. fp32 is two times the size of fp16 and needs to be reduced accordingly.

### VEC Path (element-wise / reduce /norm)

vectorOperating data go.UB:

| Buffer zone | Purpose | A binding formula |
|--------|------|---------|
| UB | All active tensor and intermediate variables | `BLOCK_SIZE × sizeof(dtype) ×ActivetensorNumber× multi_buffercoefficient≤ UBCapacity` |

compilerEnable`auto-multi-buffer`After that, the actual occupancy rate is based on2~3Double.kernelintermediate variable in (e.g.`tl.where`And the temporary buffers that are created are also occupied.UBThe actual occupancy will be significantly higher.`BLOCK_SIZE × sizeof(dtype) ×Number of inputs`.

**tile selection policy**: Starts with a larger BLONK_SIZE, downgrades when `ub overflow` compiles.

## 2. bishengir compiler known limits

### 2.1 Range() borders cannot be mixed with runtime variables

```python
# compiler crash (bishengIR SIGABRT)
for k in range(start_n, start_m + BLOCK, BLOCK_K):
    ...
```

`start_n`, `start_m` is the runtime value, `BLOCK`, `BLOCK_K` is the `tl.constexpr`. This hybrid use leads to an internal error in compiler.

**Hiding scheme**: using the full constexpr's range, skipping the invalid trajectories with runtime if:

```python
for k in range(0, N, BLOCK_K):  # N and BLOCK_K Both. constexpr
    # Optional: runtime condition skips invalid block
    ...
```

### 2.2 Complex Mask + tl. where to cause HiVM error

compiler backend may report `hivm.hir.vsel: Unsupported op for finding the root alloc` when the embedded mask group is brought into `tl.where`.

**Hiding scheme**: replace tl.where with multiplication, multiplying the data after converting the bool mask to float:

```python
# Trigger hivm.hir.vsel error
a = tl.where(tri_mask & bounds_mask, a, 0.0)

# Quest: mask spin float after multiplying
a = a * tri_mask.to(tl.float16) * bounds_mask.to(tl.float16)
```

## 3. The performance cost of Strided memory access

Ascend hardware has significant performance penalties for non-continuous memory access. When Kernel 's core path consists of memory access mode of stride > 1 (e.g. slided window for popling, spaced solution sampling), Triton 's generation of codes requires element-by-component or block gather, while CANN's original operator may use a proprietary mode of hardware data moving units (MTEs) that can be several dozen times different in performance.

**Recommendation**:
- Prioritizes the data in host to a continuous layout (e. g. `F.pad`+contigous view) and then to a continuous load processing
- K-dimensional alignment of matmul at 512B to increase bandwidth utilization
