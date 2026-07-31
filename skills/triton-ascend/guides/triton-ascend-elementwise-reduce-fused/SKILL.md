---
name: triton-ascend-elementwise-reduce-fused
description: "Applies to the combination of two phases that include both element-by-component calculations and global integrationoperatorIt's typical.operatorIncludes: Loss FunctionsMSELoss, HuberLoss, HingeLoss, SmoothL1Loss, CrossEntropyLoss, KLDivLoss, CosineSimilarityLoss, TripletMarginLossWaiting, and custom-defined pre-element transformations and global aggregationsoperatorThis kind.operatorAnd the mode of calculation is: the first step is right.tensorEach element performs independently the transformation (margin, square,clampetc.), step 2 is to make a global or dimensional return to the transformation result (sum/meanGot it.scalarOr low-dimensional results. And pure.elementwiseOr pure.reduceIt's different. It's kind of...operatorIt needs to be the same.kernel2 stages of integration to avoid additional intermediate resultsGMRead and write."
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "elementwise_reduce_fused"
---

# Elementwise + Reduce Integration operator Guide

> Composite operator (loss function, etc.) applicable to element-by-fact calculations and global regression

## Calculator Mode

Common processes for this type of operator:
1. **Elementwise phase**: execute changes to input tensor elements per element (e. g., margin, square, kamp, log, etc.)
2. **Reduce phase**: global return of transformation results (sum / mean), with scalar or low-dimensional output

## Combining Kernel Writing

Place the calculation and partial integration in the same Kernel to avoid the intermediate result writing back GM:

```python
@triton.jit
def fused_loss_kernel(
    pred_ptr, target_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    local_sum = tl.zeros((1,), dtype=tl.float32)

    for block_id in range(pid, num_blocks, CORE_NUM):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        # Elementwise Phase
        diff = pred - target
        loss_elem = diff * diff  # MSELoss Example:

        # Internal Convention
        local_sum += tl.sum(loss_elem, axis=0)

    # Cross-Register
    tl.atomic_add(output_ptr, local_sum / n_elements)
```

## Key points

1. **single kernel integration**: elementwise transforms and returns are done in the same kernel with intermediate results only in the register/ UB
2. **Atom Operations Summary**: local results of multiple programs are aggregated to global output through `tl.atomic_add`
3. **reductionParameters**: AttentionPyTorchLoss function`reduction`Parameters (%2)`'mean'`/`'sum'`/`'none'`),`'none'`It's degenerative to purity.elementwise
4. **Use VEC_CORE_NUM**: such operator does not involve `tl.dot`, use vector core
5. **Numerical stability**: mid-calculation applied to float32 to avoid a semi-accuracy spill
