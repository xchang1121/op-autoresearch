---
name: triton-ascend-verify-diagnose
description: Triton-ascend Validation Failed Diagnosis: The root causes of accuracy, mask, index, boundary, NAN/Inf are determined by the verifier 's [precision] line, hard/outlier layers thresholds, dimensional error distribution and sample coordinates
category: fix
version: "1.0.0"
dsl: triton_ascend
metadata:
  case_type: fix
  backend: ascend
  dsl: triton_ascend
---

# Triton Ascend Diagnosis Guide for Failed Validation

This skill is only used at the failed debug/fix stage of the verifier. Read the `[precision]` judgement failure type and combine the dimension-by-dimensional distribution and the sample standard location code area. The dimension-by-dimensional distribution is a supporting thread, not a forced root cause; do not blindly change the mask or file because you see a dimension range.

## 1. Read `[precision]` first.

Format:

```text
[precision] dtype=torch.float32 total=256 strict=2 outlier=0/0 hard=254 mere=2.549964e+00 mare=4.284246e+01
```

Field meaning:

| Fields | Meaning | Diagnosis priority |
|------|------|------------|
| `hard` | Number of elements above the relaxing threshold `relaxed_tol` | Prefer `hard > 0` by logic, index, mask, store, cumulative error |
| `outlier=a/b` | Quantity above, but not above, strict threshold / allowed ceiling | `hard=0` and `a>b` are more than slightly accuracy or about error |
| `strict` | Number of elements through strict thresholds | Approaching `total` means most values are correct |
| `mere` | Average relative error | Let's judge the overall deviation, not as a cause alone. |
| `mare` | Maximum relative error | View extreme error with sample coordinates |

Comparative formula:

```text
strict_tol  = atol + rtol * abs(ref)
relaxed_tol = outlier_atol + outlier_rtol * abs(ref)
hard_fail   = abs(ref - impl) > relaxed_tol
outlier     = strict_tol < abs(ref - impl) <= relaxed_tol
```

If `hard > 0`, fix hard_fail; do not first adjust the accuracy threshold.

## 2. Reread-Drive Error Distribution

Typical format:

```text
Error location per dimension ([start:end]=error index range, count/size=coverage):
  dim0: [:]  (2/2 = 100.0%)
  dim1: [:]  (3/3 = 100.0%)
  dim2: [4:5]  (1/5 = 20.0%)
  Location[0, 0, 4]: ref=5.000000e+00 impl=0.000000e+00 abs_diff=5.000000e+00 relaxed_tol=6.200000e-03
```

Read rules:

- `dim0/dim1/...` is the original dimension of the output tensor.
- `[start:end]` is the index range in which that dimension has been wrong, left closed to right.
- `[:]` indicates that there have been at least errors in all indices of this dimension.
- `count/size` is the only index number of errors in this dimension/ the size of the dimension, not the number of error elements.
- Each dimension is an independent projection and cannot be multiplied by the number of errors.
- Individual dimensions may be omitted; for example, the second-dimensional positioning value of `[M, 1]` is low.
- When there is only one non-single dimension, the dimension-by-dimensional distribution usually consists of only a little more information than the sample coordinates, giving priority to `[precision]` and the sample standard values.
- When all non-single dimensions are `[:]`, priority is given to checking global formulae, cumulation, dtype, store or buffer over, rather than fixing only local baseary mask.
- If the log does not have `Error location per dimension`, do not speculate on the dimensions mode, but only according to `[precision]` and sample normal values.

## 3. Location mode to check direction

| Location Mode | More likely the root causes. | Priority check |
|----------|--------------|----------|
| One dimension, other dimensions `[:]` | Error at the dimension boundary or index | Corresponding dimensions of fset, stride, tail mask, broadcast |
| Single continuous boundary zone, such as last column/last section | Border file or tail processing error | `tl.load`/`tl.store` mask,padding,boundary_check |
| Periodically Multiple | File Map Error | `program_id` decomposition, BLONK_M/N/K, Swizzle |
| All non-uniforms are `[:]` and `hard` more | Global calculation or writeback error | Formula, transpose, acc dtype, store pointer, buffer over |
| Only `[M]` or `[M,1]` | Weak position information on dimensions | Axis of engagement, cumulative order, sample values, duplicate returns |
| `hard=0` and `outlier > cap` | Minor value error | fp32 Plus, Cast position, Kahan, order of return |

## 4. Sample Common Value Features

| Sample Normal Values | Common causes |
|--------|----------|
| `impl=0` and `ref!=0` | Uncalculated, store mask too strict, tail omitted, padding misused |
| `impl` is much larger than `ref` | Pointer/stride error, read other files, acc uncompleted, buffer over |
| `impl` contrary to `ref` symbol | Subtract direction, shift, input order error |
| Multiple positions `impl` repeats the same value | Write overwrite, program_id map error, calculated only one row/ tile |
| error is small but above state | accuracy/cumulative questions, do not significantly change the index logic |

## 5. Rapid decision-making

1. `NaN/Inf` location does not match: repairing illegal operations first, invalid load under mask, de-zero or spill.
2. `hard > 0` and the sample value is clearly wrong: look at logic, index, mark, store first.
3. `hard > 0`, however the error covers all non-single dimensions: check first the global formula, add, dtype and store, not just the boundary.
4. `hard=0 && outlier > cap`: Deal with accuracy issues with priority fp32 accumulator, cast position and Kahan.
5. Low-dimensional output `[M]` / `[M,1]`: Do not over-reliance on the dimension-by-dimensional distribution; it is usually only a supplement to the sample index.
