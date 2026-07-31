---
name: triton-ascend-case-vector-elemwise-bench-atlas-a3
description: "Atlas A3 for Triton vector one dollar/ two dollars `tl.*`: fp32/fp16/bf16 three dtypes each operator end-to-end time (ms); and give \" semantic equivalents, accuracy versus \" to replace tritoon API and recommended writing. For example, on fp32 `tl.exp2(x)` has a poor performance compared to `tl.exp(x*LN2)` and can choose to use `tl.exp(x*LN2)`."
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A3"
---


## note

1. **`exp` / `exp2`**: There is a significant difference in performance between the two (Fp32, `tl.exp2` has poor performance compared to `tl.exp`, Mathematics Equivalence**`exp2(x) = exp(x * LN2)`**, do not confuse with**`LOG2E`**, but ensure accuracy's consistency.

---

# Replacement of API Recommendations (Atlas A3)

> **accuracy warns**: The following replacements are for the single operator unit test: Semantic Equivalence, accuracy Alignment, but when operator merges or is used on the Internet, the following is used:
> error may be cumulatively amplified.
> **If accuracy cannot be aligned after replacing an API, please retain the original text and do not replace it.**

## fp32

| Original | Suggested Replace | Proceeds |
|---|---|---|
| `tl.exp2(x)` | **`tl.exp(x * LN2)`**(`LN2=0.6931471805599453`) | 45% |
| `tl.where(x>0, x, 0)` Achievedrelu | `(x + tl.abs(x)) * 0.5` or `tl.maximum(x, 0)` | 72% |
| `tl.div_rn(x, y)` | `x / y` or `tl.fdiv(x, y)` or `x * (1.0/y)` | 17% |
| `x * tl.sigmoid(x)` achieve silu | `x / (1 + tl.exp(-x))` | 14% |
| `tl.sigmoid(x)` | `tl.exp(x) / (1 + tl.exp(x))` | 7% |
| `x * tl.rsqrt(x)` or `1.0/tl.rsqrt(x)` achieves sqrt | `tl.sqrt(x)` or `tl.sqrt_rn(x)` | 5-6% |

## fp16

| Original | Suggested Replace | Proceeds |
|---|---|---|
| `tl.where(x < y, x, y)` achieves minum | **`tl.minimum(x, y)`** | **57%** |
| `tl.sigmoid(x)` | `tl.exp(x) / (1 + tl.exp(x))` | 10% |
| `tl.rsqrt(x)` | `1.0 / tl.sqrt(x)` | 11% |
| `tl.log2(x)` | `tl.log(x) * LOG2E`(`LOG2E=1.4426950408889634`) | 8% |
| `tl.abs(x)` | `tl.maximum(x, -x)` | 8% |
| `tl.fdiv(x, y)` | `tl.div_rn(x, y)` | 7% |
| `tl.where(x>0, x, 0)` or `(x+\|x\|*0.5 ' Realization relu | `tl.maximum(x, 0)` | 7-8% |


## bf16

| Original | Suggested Replace | Proceeds |
|---|---|---|
| `tl.abs(x)` | **`tl.maximum(x, -x)`**(A3 bf16Let's go.`tl.abs`I don't know what you're doing. | **103%** |
| `tl.where(x>0, x, 0)` Achievedrelu | **`tl.maximum(x, 0)`** | **86%** |
| `tl.where(x<y, x, y)` achieves minum | **`tl.minimum(x, y)`** | **78%** |
| `2*tl.sigmoid(2*x)-1` achieves the taunh | `1 - 2/(tl.exp(2*x)+1)` or `(exp(x)-exp(-x))/(exp(x)+exp(-x))` | 33% |
| `(x + tl.abs(x)) * 0.5` Achievedrelu | `tl.maximum(x, 0)` | 21% |
| `tl.exp2(x)` | `tl.exp(x * LN2)` | 21% |
| `tl.sqrt_rn(x)` | `tl.sqrt(x)` | 11% |
| `tl.log2(x)*LN2` realization log | `tl.log(x)` | 6% |
| `tl.log2(x)` | `tl.log(x) * LOG2E` | 6% |


## 2D/Special scene across dtype

| Original | Suggested Replace | Remarks |
|---|---|---|
| `acc / l[:, None]`(`acc: (M, D)`, `l: (M,)`)—— fp32 | `l_recip = 1.0/l; acc * l_recip[:, None]` or `acc * (1.0/l)[:, None]` | A3 fp32 Direct detachment**42%**, drop `M*D` vdiv to `M`, revolving vmul |
| `acc * (1.0 / l)[:, None]`(fp16) | `l_recip = 1.0/l; acc * l_recip[:, None]` | Inline writing for `1.0/l` fp16 is 42% slow; visible tearing out `l_recip` is more stable|
