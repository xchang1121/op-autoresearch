# Broadcast - Dynamic UB Broadcast (DAV_3510)

> **Applicable scene**: DAV_3510 chip, multidimensional after axle. Use dynamic Broadcast API (rank 1-9) broadcast in UB without 32B alignment limit.
>
> **DAV_2201**Please use the static interface.rank=1/2) for more details[ub-broadcast.md](ub-broadcast.md).

---

## I. Comparison with static interfaces

| Dimensions | Static interface (DAV_2201/DAV_3510) | Dynamic interface (DAV_3510) |
|------|-------------------------------|---------------|
| Chip. | DAV_2201/DAV_3510 Common | DAV_3510 only |
| rank | 1D/2D only | **1~9** |
| axis | 0/1 (compilation) only | **Any axis**(runtime) |
| Alignment | dim=2, axis=0 need srcShape [1] 32B alignment | **No Match Limit** |
| tmpBuffer | Requires manual management or framework applications | Tiling Internal Management |
| dtype | int8/uint8/half/float | int8/uint8/int16/uint16/half/bfloat16/int32/uint32/float/int64/uint64 |

---

## II. API

```cpp
// 1. Kernel side calculation Tiling
BroadcastTiling tiling;
GetBroadcastTilingInfo<T>(rank, dstShape, srcShape, false, tiling);

// 2. Implementation of broadcasting
Broadcast<T>(dstLocal, srcLocal, dstShape, srcShape, &tiling);
```

**Parameter description**:

| Parameters | Annotations |
|------|------|
| rank | Dimensions, [1, 9] |
| dstShape | Output Shape, uint32_t array, length = rank |
| srcShape | Enter Shape, uint32_t array, length = rank. srcShape[i]=1 and dstShape[i]>1 at this axis broadcast |
| srcInnerPad | Whether the last dimension is 32B alignment, only false is currently supported |
| tiling | Output of `GetBroadcastTilingInfo` to `Broadcast` |

**Example:**

```cpp
// [2, 1,4] → [2, 3, 4](broadcast along axis=1, rank=3)
uint32_t dstShape[] = {2, 3, 4};
uint32_t srcShape[] = {2, 1, 4};
BroadcastTiling tiling;
GetBroadcastTilingInfo<float>(3, dstShape, srcShape, false, tiling);
Broadcast<float>(dstLocal, srcLocal, dstShape, srcShape, &tiling);

// [1] → [4,8](broadcast along axis=0, rank=2, no 32B alignment requirements)
uint32_t dstShape2[] = {4, 8};
uint32_t srcShape2[] = {1, 8};
BroadcastTiling tiling2;
GetBroadcastTilingInfo<half>(2, dstShape2, srcShape2, false, tiling2);
Broadcast<half>(dstLocal, srcLocal, dstShape2, srcShape2, &tiling2);
```

---

## III. CONSTRAINTS

| Constraints | Annotations |
|------|------|
| **Chip** | DAV_3510 only |
| **rank** | [1, 9] |
| **Broadcasting conditions** | srcShape[i]=1 and dstShape[i]>1 |
| **Address overlap** | Src and dst cannot overlap |
| **srcInnerPad** | Current only support |

---

## IV. Data flows

Same as static UB Broadcast, the difference is only called by Broadcast API:

```
GM → DataCopyPad → UB [srcShape, Not broadcast]
  ↓
GetBroadcastTilingInfo + Broadcast → UB [dstShape, Broadcasted]
  ↓
Compute → UB → DataCopyPad → GM
```

Tiling parameter calculation, multi-nuclei, multi-dimensional index management is identical to [ub-broadcast.md] (ub-broadcast.md).
