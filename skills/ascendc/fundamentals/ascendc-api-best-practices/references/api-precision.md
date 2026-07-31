# Guide to accuracy Conversion and Mixed accuracy

Cast API uses standard and hybrid accuracy mode of calculation.

---

## Contents

1. [Cast RoundMode Selection](#cast-roundmode-Selection)
2. [Mixed accuracy mode of calculation (FP16 input)](#Mixed accuracy mode of calculationfp16-Input)
3. [MX block quantitative format accuracy path](#mx-block quantitative format accuracy path mxfp8--mxfp4- etc.)

---

## Cast RoundMode Selection

### Selection Rule

| Convert direction | RoundMode | Reason |
|---------|-----------|------|
| **half → float** | `CAST_NONE` | Lower accuracy → High accuracy, no accuracy losses |
| **float → half** | `CAST_ROUND` | High accuracy → low accuracy with accuracy's loss. |
| half → int32_t | `CAST_ROUND` / `CAST_CEIL` | Quantified scene, selected according to demand |
| int32_t → float | `CAST_NONE` | Integer number → floating point, no accuracy loss |

### Use correctly

```cpp
// ✅ half → float: low accuracy to high accuracy
AscendC::LocalTensor<float> xFloat = workBuf.Get<float>();
AscendC::Cast<float, half>(xFloat, xHalf, AscendC::RoundMode::CAST_NONE, count);

// ✅ float → half: High accuracy to low accuracy
AscendC::LocalTensor<half> yHalf = outQueue.AllocTensor<half>();
AscendC::Cast<half, float>(yHalf, xFloat, AscendC::RoundMode::CAST_ROUND, count);
```

---

## Mixed accuracy calculation mode (FP16 input)

### Apply scene

When the input output is FP16 but requires FP32 accuracy for intermediate calculations (e.g. Softmax, Layer Norm).

### Calculating Processes

```
half Input → Cast(FP32) → Intermediate calculation(FP32) → Cast(half) → half Output
```

### Why do you need FP32 mid-calculation?

1. **Reduce Max /Exp/ReduceSum**accuracy is more stable on FP32
2. **Avoid FP16 numeric spill**: Exp result may exceed the FP16 expression
3. **Accumulated error control**: cumulative error with multiple operations is smaller under FP32

### Examples of additions and subtractions

Semi-accuracy plus minus by default raise FP32; direct `Add/Sub<half>` is permitted only when spec expressly "input equivalents" (e.g. mask supercharge, combined with a reduced probability). BF16 applies the same rule as FP16, with a different threshold ratio (BF16 = 128, FP16 = 1024).

> Full examples, decision tables, and Kernel integration features are presented in [api-arithmetic.md → 3](api-arithmetic.md# 3 1/2 accuracy plus minus accuracy optimised).

---

## MX block quantification format accuracy path (mxfp8 / mxfp4 etc.)

### Apply scene

Enter or export the MX block quantification format (mxfp8 / mxfp4 / mxfp6 etc.): E8M0 scale is shared for each 32 element group, with the data master going low accuracy dtype (e.g. fp8_e4m3 / fp8_e5m2 / fp4).

### Overview of data access

```
Highaccuracy fp32 tensor → Quantified axis per hour 32 Element Group Calculations amax
                ↓
            E8M0 scale Generate
                ↓
            Cast<Quantitative dtype, fp32>(x / scale) → LowaccuracyData + Accompany scale
```

### E8M0 code code

E8M0 code**Must be offset by ceil**:

```cpp
e8m0_byte = (biased_exp_amax - emax_quant_dtype) + 1;  // ✅ ceil
```

Parameters:
- `biased_exp_amax`: fp32 Biased exponent (0-255)
- `emax_quant_dtype`ObjectivesscalarDilutiondtype of max exponent(fp8_e4m3=8,fp8_e5m2=15,fp4_e2m1=2)
- `+1`:ceil offset to ensure `amax / decoded_scale ≤ quant_dtype_max`

### Inverse mode: floor offset caused NAN

```cpp
❌ e8m0_byte = biased_exp_amax - emax_quant_dtype;  // Missing +1  To fall in   floor Intersection
```

Floor offset `amax / decoded_scale` may fall between `[quant_dtype_max, 2 × quant_dtype_max)` (for e4m3 or `[448, 896)`). Cast <Quantified dtype, fp32, RINT> will output NAN over dtype_max (for fp8_e4m3 for `0x7F`).

### API Path for Cast<fp8_e8m0_t>

The scale dtype in MX-type format is `fp8_e8m0_t`, which needs to be encoded as e8m0 for fp32's biased exponent part:

| Path | Annotations |
|------|------|
| **MemBase Cast API** | Do not provide `fp8_e8m0_t` to reload |
| **Reg-API Cast** | Provides `bfloat16_t ↔ fp8_e8m0_t` overload (checking Reg-API variants such as `Cast-45.md`), subject to the introduction of the Regbase programming paradigm |
| **Handwritten position calculations** | Insert by `ReinterpretCast<uint32_t>(fp32_tensor)`, `ShiftRight + And + Sub + Add` extract based ext and apply ceil offset |

The main path selection is based on the programming paradigm of the operator as a whole: the pure MemBase path is run in hand-written bits, and the Reg-API path is directly used in Cast.

### Checklist

New MX Quantified Path must verify:
- [ ] A max returns along the right axis
- [ ] E8M0 scale formula offset with ciel
- [ ] Cast after output no NAN (with NAN → mostly floor offset)
- [ ] `amax / decoded_scale` always ≤ quantified dtype max

### Compiled misdiagnostic quick check

| Error Compiled | Magen. | Rehabilitation |
|--------|-------|------|
| `Mmad` does not accept fp8 ptr | The Mmad fm/filter template parameter homobar is not met | Visible Assign `Mmad<fp32, fp8_e4m3_t, fp8_e4m3_t>` Equivalent Group |
| `Cast<fp8_e8m0_t, fp32>` unresolved | MemBase Cast does not support e8m0 directcast | Manually run or switch the Reg-API path |

> MXFormattedscaleAxis Selection (where along)tensorQuantification of dimensions) is a design-level decision that involvesmatmul reductionAlignment of axes—— only Cube/matmulCategoryoperatorRelated, this warehousevector / reductionIt's not going to spread.
