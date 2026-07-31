---
name: triton-ascend-case-vector-elemwise-bench-atlas-a5
description: "Atlas A5Let's go.Triton vectorOne dollar./Two dollars.`tl.*`:fp32/fp16/bf16Three.dtypeDownwards.operatorEnd-to-endtime (ms);and give\"Semantic equivalent,accuracyAlignment\"Should have been replaced.triton APIAnd recommendive writing."
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
---


# Replacement of API Recommendations (Atlas A5)

> **accuracy warns**: The following replacements are for the single operator unit test: Semantic Equivalence, accuracy Alignment, but when operator merges or is used on the Internet, the following is used:
> error may be cumulatively amplified.
> **If accuracy cannot be aligned after replacing an API, please retain the original text and do not replace it.**

## fp32

| Original | Suggested Replace | Proceeds |
|---|---|---|
| `x / (1 + tl.exp(-x))` achieve silu | **`x * tl.sigmoid(x)`** | **128%** |
| `tl.fdiv(x, y)` / `tl.div_rn(x, y)` / `x*(1.0/y)` achieved div | **`x / y`**(A5 fp32 directly divided by the fastest) | 87%-94% |
| `x * tl.rsqrt(x)` achieves sqrt | **`tl.sqrt(x)`** | **71%** |
| `tl.sqrt_rn(x)` | `tl.sqrt(x)` | 55% |
| `tl.where(x < y, x, y)` achieves minum | `tl.minimum(x, y)` | 75% |
| `tl.where(x > y, x, y)` achieve maxium | `tl.maximum(x, y)` | 53% |
| `tl.where(x>=0, x, -x)` achieve abs | `tl.maximum(x, -x)` or `tl.abs(x)` | 32% |
| `1.0/tl.rsqrt(x)` achieves sqrt | `tl.sqrt(x)` | 27% |
| `exp(x)/(1+exp(x))` achieve sigmoid | `0.5*(1+tanh(x/2))` or `tl.sigmoid(x)` | 23% |
| `tl.exp2(x*LOG2E)` achieves exp | `tl.exp(x)` | 11% |
| `tl.log2(x)` | `tl.log(x) * LOG2E` | 10% |

## fp16

| Original | Suggested Replace | Proceeds |
|---|---|---|
| `tl.sqrt(x)` | **`1.0 / tl.rsqrt(x)`** | **84%-89%** |
| `tl.sigmoid(x)` | **`1/(1+tl.exp(-x))`** | **50%** |
| `tl.where(x > y, x, y)` achieve maxium | **`tl.maximum(x, y)`** | **59%** |
| `(exp(x)-exp(-x))/(exp(x)+exp(-x))` / `2*sigmoid(2x)-1` achieve the taunh | **`1 - 2/(tl.exp(2*x)+1)`** | **46%-50%** |
| `tl.exp2(x)` | `tl.exp(x*LN2)` | 65% |
| `x / y` Achieveddiv | `tl.div_rn(x, y)` or `tl.fdiv(x, y)` | 25% |
| `tl.maximum(x, 0)` / `tl.where(x>0, x, 0)` Achieved relu | `(x + tl.abs(x)) * 0.5` | 6-11% |
| `tl.where(x>=0, x, -x)` / `tl.abs(x)` | `tl.maximum(x, -x)` | 11-15% |

## bf16

| Original | Suggested Replace | Proceeds |
|---|---|---|
| `tl.where(x < y, x, y)` achieves minum | **`tl.minimum(x, y)`** | **199%** |
| `tl.where(x > y, x, y)` achieve maxium | **`(x+y+tl.abs(x-y))*0.5`**or `tl.maximum(x, y)` | 75% |
| `tl.rsqrt(x)` | **`1.0 / tl.sqrt(x)`** | **73%** |
| `tl.sqrt_rn(x)` / `x*tl.rsqrt(x)` achieves sqrt | **`1.0 / tl.rsqrt(x)`**or `tl.sqrt(x)` | 93%-101% |
| `tl.sigmoid(x)` | **`exp(x)/(1+exp(x))`**or `0.5*(1+tanh(x/2))` | **63%** |
| `tl.div_rn(x, y)` | `x / y` | 17% |
| `tl.log(x)` | `tl.log2(x) * LN2` | 14% |
| `x / (1 + tl.exp(-x))` achieve silu | `x * tl.sigmoid(x)` | 29% |


## 2D/Special scene across dtype

| Original | Suggested Replace | Remarks |
|---|---|---|
| `acc / l[:, None]` or `acc*(1.0/l)[:, None]` | **`l_recip = 1.0/l; acc * l_recip[:, None]`** | A5 fp32 on `l_recip+vmul`: Fastest, original is directly slow**46%**, inline recip 74% |
