---
name: ascendc-hardware-tiling
description: "AscendC hardware and tilling base: UB/L1/L0 capacity, Victor/Cube nuclear relationship, DataCopy alignment, blockDim/tail calculation and batch development of operator no variable."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# AscendC hardware and Tiling Foundation

The use of this skill before writing or modifying AscendC Kernel is particularly applicable to a scenario where a operator needs to cover multiple groups of Shape, or multiple operators requires batch re-use of the same set of tilling skeletons.

## 1. Hardware capacity and query principles

The host Tiling code does not scatter hard-coding chip parameters. If SDK can query first, ask first; if the current seed already provides constant, it should be in a helper.

| Buffer zone | Typical uses | Development rules |
|---|---|---|
| UB | Victor input, output, temporary tensor | file design must accommodate all live LocalTensor and queue |
| L1 | Cube input reuse or larger statusing | Use only when a real reuse exists |
| L0A/L0B | Cube Left/Right Operations | Design Cube API binding to avoid exceeding 64KB level |
| L0C | Cube cumulative result | Visible Controls Plus Shape and dtype |

A2/A3/A5CategorydeviceLet's go.VectorCore and CubeCoreCould be separated.kernelOrganisationCube and VectorWork should ensure that the two core sets of tasks do not wait long for each other.

## 2. Tiling Data Compact

Tiling struct should be small, stable, host/kernel fully consistent:

```cpp
struct TilingData {
    uint32_t total;
    uint32_t tile;
    uint32_t tail;
    uint32_t tilesPerCore;
    uint32_t coreNum;
};
```

Rules:

- Use fixed width integer types such as `uint32_t`, `int64_t`.
- Most must be consistent with the order, type and meaning of the fields in Kernel.
- Field names clearly distinguish between the number of elements and the number of bytes, such as `elemLen` and `byteLen`.
- `DataCopy` length units must be consistent with API requirements and do not mix elements and bytes.
- Leaves the precalculated scalar field in a host; too big tilling struct increases setup expenses.

Multi-Model Kernel recommends visible addition to the Mode field instead of repeating the shape:

```cpp
enum Mode : uint32_t {
    MODE_GENERIC = 0,
    MODE_SAME_SHAPE = 1,
    MODE_LAST_DIM_BROADCAST = 2,
    MODE_SMALL_REDUCE = 3,
};

struct TilingData {
    uint32_t mode;
    uint32_t total;
    uint32_t inner;
    uint32_t tile;
    uint32_t tilesPerCore;
    uint32_t coreNum;
};
```

Rules:

- Mode must be judged in terms of dtype, rank, stride, broadcast, axis, numel, etc., on the side of host.
- Mode only indicates the algorithm path, and does not encode the full spectrum into a bunch of Modes.
- ggeneric Mode must keep full semantic coverage.
- The device end mode branch should be selected from the outer layer to avoid repeated judgement within each element.

## 3. Block and Tail Formula

The security starting point for the one-dimensional mission is:

```text
core_num       = min(max_cores, ceil_div(total, min_work_per_core))
work_per_core  = ceil_div(total, core_num)
core_start     = block_idx * work_per_core
core_len       = min(work_per_core, total - core_start)
tile_count     = core_len / tile_len
tail_len       = core_len % tile_len
```

Kernel side must meet:

- If `core_start >= total` returns directly before using the GM pointer.
- The last valid core uses `core_len`, not the nominal `work_per_core`.
- `tail_len == 0` does not perform zero length or old length tail copy.
- The active length of tail vector mask is equal to the actual number of remaining elements.

## 4. Alignment Rules

- GM/UB move preference to 32B alignment file size.
- Non-32B alignment data using `DataCopyPad` or separate tail paths.
- When the UB bank appears, the Hotspot LocalTensor at least staggers 32B.
- The vector API unit requires confirmation on a line-by-line basis and does not apply the host bytes directly.

## 5. Queue with Pipeline

The Queue Kernel maintains the following structure:

```cpp
auto in = inQueue.AllocTensor<T>();
DataCopy(in, gmIn + offset, len);
inQueue.EnQue(in);

auto ready = inQueue.DeQue<T>();
Compute(ready, outLocal, len);
inQueue.FreeTensor(ready);
```

Inspection entry:

- Each `AllocTensor` has an `FreeTensor`.
- Each `EnQue` has a matching `DeQue`.
- Branches cannot skip release after `AllocTensor`.
- The tail branch cannot skip the `DeQue` required for the main path follow-up.
- `PipeBarrier` is added only when there is a clear data dependency; too many barriers will destroy overlap.

## 6. Bulk development rules

When handling multiple operators:

1. operator is first classified as elementwise, routecast, reduction, indexed, matmul-like or used.
2. Each class only maintains one tilling skeleton.
3. Each operator changes the semantic point only: output Shape, dtype branch, calculation expression, contract axis, tail policy.
4. Direct-invoke wrapper and CMake layouts remain stable.
5. Do not allow Kernel to pass by changing tolerance or syntax.

## 7. Tiling and Performance Split

When the same operator covers multiple groups of Shape, tilling requires both service correctness and performance. It is recommended to record at least the following partition fields:

```text
same_shape:
last_dim_broadcast:
small_reduce:
single_tile_row:
large_integral:
aligned_main:
tail_only_pad:
```

Example:

```cpp
bool sameShape = x.sizes() == y.sizes();
bool bothContig = x.is_contiguous() && y.is_contiguous();
bool lastDimBroadcast = x.dim() >= 2 && y.dim() == 1 &&
                        y.size(0) == x.size(x.dim() - 1);

if (sameShape && bothContig) {
    tiling.mode = MODE_SAME_SHAPE;
    tiling.total = x.numel();
} else if (lastDimBroadcast && bothContig) {
    tiling.mode = MODE_LAST_DIM_BROADCAST;
    tiling.inner = y.size(0);
    tiling.total = x.numel();
} else {
    tiling.mode = MODE_GENERIC;
}
```

Example branch:

```cpp
if (mode_ == MODE_SAME_SHAPE) {
    ProcessFlatten();
} else if (mode_ == MODE_LAST_DIM_BROADCAST) {
    ProcessRowBroadcast();
} else {
    ProcessGenericIndexMapping();
}
```

Do not make gratuitous index apping a default performance path for all inputs; it usually contains `div/mod` and multi-level stride calculations, only suitable for fallback.
