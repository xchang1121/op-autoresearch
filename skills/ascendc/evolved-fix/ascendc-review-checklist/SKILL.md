---
name: ascendc-review-checklist
description: "Quick review list before and after revision of AscendC operator: direct-invoke wrapper, CMake/Registration, launch ABI, Tiling, DataCopy, Sync, dtype/shape over and unquietly downgraded."
category: fix
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  case_type: review
---

# AscendC Revision Review List

This skill is used before submission, before batching, or when the code appears reasonable but the authentication fails. It is used to quickly rule out common hypothetical errors and is not a substitute for real verify.

## 1. Wrapper and Build

- `kernel.py` only defines one public `ModelNew`.
- `ModelNew` lazyly loads the constructed `.so`, which is not compiled at the Import stage.
- The rules for loading the target library are stable when multiple `.so` exists.
- `torch.ops.npu.<op>` namespace corresponds to the registration of C++.
- The Meta function returns the exact output Shape and dtype.
- CMake produces a shared library under `ascendc_op/build`.
- `--npu-arch` is controlled by CMake Variables or Adaptor Patch, without expired hard encoding.

## 2. Launch ABI

This post is part of our special coverage Global Voices 2011.

- Number of parameters.
- parameter order.
- The pointer type is agreed.
- Tiling pointer/tensor parameter.
- workspace pointer.
- stream parameter.
- `blockDim`.

Synchronizes the host Bridge and Python call paths when changing kernel parameters.

## 3. Tiling and Memory

- Tiling fields are in the same order and type on both sides of the host/kernel.
- `SetGlobalBuffer` span equals the actual number of elements that can be accessed by the current core.
- `GetBlockIdx()` calculates that it will not exceed total.
- tail path uses real tail length and valid vector mask.
- `DataCopy` length units are correct.
- Use `DataCopyPad` or special tail path for non-matched loads.

## 4. Queue and Sync

- Each `AllocTensor` was released.
- Each `EnQue` has a match `DeQue`.
- No branch skips the cleanup after allocation.
- Cross-nucleus waits on all involved cores matter.
- Barrier relies on real data and is not used as a cover for unknown reasons.

## 5. Value & Override

- Use fp32 intermediate results by reference accuracy for sensitive paths such as reduction, exp/log/div/sqrt.
- Cast round Mode in float-to-half, half-to-float.
- dtype branch covers the visible scope of mission requirements.
- Shape-specific path cannot be merged silently unless the mandate expressly permits.
- Tolerance changes cannot be used to mask errors made.

## 6. Batch safe.

- A single op fix should not modify shared scaffold, leading to the failure of other ops.
- If you have simplified removal of dtype, playout or shape paths, you need to write down deliberate exception in the report.
- Priority is given to minor minor minor modifications, accompanied by new corroborating evidence.

## 7. Special inspection for performance optimization

The performance change does not mean that it is worth retaining.

- Whether only one explicit performance assumption, such as a file, synchronization, paste, batch, index addition, has been changed.
- Whether the total and sample-by-sampling indicators are viewed at the same time to avoid the return of the majority of the samples being covered by individual sample benefits.
- Whether narrow path conditions are semantic: dtype, rank, contiguous, Broadcast, reduce axis, numel, special value mode, rather than an unexplained whole shape.
- Whether the library achieves by-passes cover only the specified degradation mode and the main AscendC path still covers ordinary inputs.
- Whether non-contigous, broadcast, empty tensor, tail, non-32B alignment or dtype variants are omitted because of the fast path.
- tile, BUFER_NUM,rowsPerTile changes recalculated the UB account book.
- Whether to delete sync is supported by data; do not delete barrier as accidentally correct.
- Deletes whether dead code is synchronised to clean CMake, extran declaration, register, Meta and launcher.
- If you use in-place or alias output, you must confirm that the synonym allows overlaying of the input.

Sample-by-sample performance report recommended records of:

```text
dominant slow samples:
shared semantic pattern:
changed path:
total metric before/after:
worst regression sample:
coverage preserved:
rollback condition:
```

## 8. Minimal check before submission

```text
build/load:
correctness shapes:
tail/non-aligned shapes:
dtype variants:
known exclusions:
profiling status:
```

The above-mentioned fields are empty, and performance conclusions are not considered final.
