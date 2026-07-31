---
name: triton-ascend-kahan-precision-fix
description: Triton-ascend Large K Approximate accuracy fix: Kahan compensation and replacement simple add-up, eliminating the difference between the NPU Cube engine FP32 simulation path and the Triton sequence cumulative path of accuracy
category: fix
version: "1.0.0"
metadata:
  case_type: fix
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3, Atlas A5"
---

# Kahan compensation claim and accuracy repair

## Trigger Condition

- Matmul / reduction Kernel verify hard_fail > 0 under K ≥ 4096
- Mere normal (<1e-4) but mare oversize (>1e-2), indicating that a few points error are very large

## Repair: Kahan compensation claim and sum

```python
# Error: Simple add-up, K-time error O (K × eps)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(...)
    b = tl.load(...)
    acc += tl.dot(a, b)

# Repair: Kahan compensation, error down to O(eps)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
comp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(...)
    b = tl.load(...)
    partial = tl.dot(a, b)
    y = partial - comp
    t = acc + y
    comp = (t - acc) - y
    acc = t
```

### Rationale

Floating point addition does not satisfy the combination rule: `(a + b) + c ≠ a + (b + c)`. When `acc` is large and `partial` is very hourly, `acc + partial` loses `partial`'s low-level accuracy. Kahan Algorithm tracks each lost accuracy through a compensatory variable, `comp`, and is next added.

Gradual dismantling:

```
partial = tl.dot(a, b)       # This time. dot Result
y = partial - comp           # Less the last one lost.accuracy(Compensation)
t = acc + y                  # Gradient
comp = (t - acc) - y         # We'll catch the missing one.accuracy
acc = t                      # Update Thrust
```

- `y = partial - comp`: Recover the last part that was lost
- `t = acc + y`: Actual cumulative execution
- `comp = (t - acc) - y`: `(t - acc)` is the value actually added to acc minus `y` to get the low of this loss
- `comp` is always zero on algebra, but it's rounded to error in the float operation

## Scope of application

| scene | Whether it applies |
|------|---------|
| matmul K ≥ 4096 | ✅ |
| Reduction (sum/mean) dimensions | ✅ |
| matmul K < 4096 | ❌ simple enough to add |
| Elementwise / No Return | ❌ free of charge error |

## Quick Checklist

1. **hard_fail > 0 + mare > 1e-2 + K ≥ 4096**→**Kahan Compensation (§ Rehabilitation)
2. **NPU vs NPU after Kahan still has hard_fail**→ to check Kernel logic (non-accumulative problem)
