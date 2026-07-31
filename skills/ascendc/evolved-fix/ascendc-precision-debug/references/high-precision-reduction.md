# High accuracy Convention: FP32 When the compressor is insufficient

> [common-traps.md] (common-traps.md)'s trap 4 (Reduce accuracy's loss).
> The conclusion is "with a FP32 cumulator." This paper says that**FP32 cumulator is not enough**scenes and tools.

## When do you need this?

FP32 Accumulator (approximately 6-7 bits) will still fail in three cases:

1. **Fp64 Ref: evaluation based on MERE/MARE vs. fp64 golden, small range/resistance position requirements
   NPU error ≤ 2×CPU-fp32 error (or even error 0). fp32 add differently from CPU**in order,
   Close to zero is different.
2. **Large K Convention**: K≈1e3 ~1e4, single fp32 loaders add up so many items ~1e-3
   (Not related to input bit width, is the rounding of the loader itself).
3. **Catastrophe offsets**: large items have shrunk to nearly zero and absolute error has been magnified to a large relative error.

Order of decision: Make sure it's not enough to compress the loader (to downsize K/ replace fp64 reference comparison) and then move on.

---

## Method 1: Compensation sum (Kahan → Neumaier → guarded)

Compensation and recovery of lost lows per step plus with an `comp` variable to pull fp32 to ~fp64.

**Don't use nudity Kahan**: `t - acc` to produce `inf - inf = NaN` at inf that poisons the whole chain.
Using**Neumaier**(subsidiary by operational size)**and**plus finite guard**— non-limited step over compensation,
Just add inf/nan to the correct semantics.

```cpp
// Guarded-Neumaier scalar cum: acc + comp for ~fp64 and
float acc = 0.0f, comp = 0.0f;
for (int i = 0; i < n; ++i) {
    float v = ToFloat(x[i]);          // Item by Item
    float t = acc + v;
    // Missing low: press|acc| and |v|Who's the big branch?
    float lo = (Abs(acc) >= Abs(v)) ? (acc - t) + v : (v - t) + acc;
    if (IsFinite(lo)) comp += lo;     // Guard:inf/nan I'm not gonna make it up to you. I'm gonna take it with me.
    acc = t;
}
float result = acc + comp;
```

- Remedial sequence losses only,**without repairing the volume of individual items/casts error**(means 2).
- Host side Aten Group Contract: `where(isfinite(lo), lo, 0)` as guard, by element tensor version.

## Method 2: TwoProduc - Accurate (no FMA)

When**Accumulation**per se exceeds the end number of fp32 (24 bit), the claim for compensation cannot be saved because the loss occurred in a multiplication:

| scene | Accumulated active bits | Is it necessary? |
|---|---|---|
| fp16 × int8 | ≤19 | No (fp32 accurate) |
| fp16 × fp16 | ≤22 | Yes |
| fp16 × fp32 (full) | Toda ~29 | **Yes** |
| Large value fp32 × fp32 | Toda ~46 | **Yes** |

Veltkamp split (separation factor `4097 = 2^12+1`, fp32 special,**not dependent on FMA**) for each volume
Disassembly to `p + e`, `p+e` equals exact volume, `e` is the lost low, fed to `comp`:

```cpp
// TwoProduct(a, b) - (p, e), p+e = a*b
inline void TwoProduct(float a, float b, float& p, float& e) {
    const float S = 4097.0f;          // 2^12 + 1
    p = a * b;
    float ca = a * S, ah = ca - (ca - a), al = a - ah;
    float cb = b * S, bh = cb - (cb - b), bl = b - bh;
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}
```

## Method 3: Small Return = TwoProduct + guarded-Neumaier → fp64-exact

**Minor return**(conv single point 9 ~ 125 tap, RoPE 2 item, small stencil): Each buildup
& Useguarded-Neumaier,`comp`It's also a low-level accumulation.`e`And plus low places.`lo`:

```cpp
float acc = bias, comp = 0.0f;
for (Every one tap) {
    float p, e; TwoProduct(x_tap, w_tap, p, e);
    float t = acc + p;
    float lo = (Abs(acc) >= Abs(p)) ? (acc - t) + p : (p - t) + acc;
    float delta = lo + e;
    if (IsFinite(delta)) comp += delta;
    acc = t;
}
return acc + comp;                     // ~fp64Close to zero, too. golden
```

Actual: fp32 depthwise conv close to zero-sum case (large weight domain) from system outlier
Pressed to MERE ~1e-14; RoPE fp32 to equalize value.

## Method 4: Major return (matmul / conv-backward, K~1e3-1e4)

Unable to complete materialize ([M,K,N] RAM implosion), renumber**Part + Inter-part compensation**:

```
Take it. K Cut CHUNK=256 (a) Blocks;
I'll do it once in a piece. matmul(In block) fp32 (b) Added, with small losses);
between blocks Neumaier Compensating part of the uniting.
```

```cpp
// pseudocode: chunked-K + Neumaier
acc = matmul(A[:, 0:CH], B[0:CH, :]); comp = zeros_like(acc);
for (k0 = CH; k0 < K; k0 += CH) {
    p = matmul(A[:, k0:k0+CH], B[k0:k0+CH, :]);
    t = acc + p;
    comp += where(abs(acc) >= abs(p), (acc - t) + p, (p - t) + acc);
    acc = t;
}
y = acc + comp;
```

- There are still fp32 rounded in the block, so this is**approaching**is not accurate; it can press ~1e-3 for large K to close the threshold,
  Raise the average, but the deepest dichotomy may still fall short (see ceiling).
- If the accumulation is also greater than 24 bit (e.g. weights vs. quantification fp32), stacking means 2 to right.

## HF32 trap: fp32 matmul/conv default not full accuracy

Ascend Cube against**fp32 matmul defaulted to HF32**- multiplies the input end number to ~11 bits.
**Conv's HF32 default is open.**For fp64 reference, turn off:

```python
import torch, torch_npu
torch.npu.matmul.allow_hf32 = False    # Allaccuracy fp32 matmul
# Toch.npu.conv.allow_hf32 =False #conv
```

- Off HF32 is**accuracy mode switch**, not kernel, not circumventing anti-fiction.
- HF32 Impact only**Input round**does not affect the fp32 loader itself (for integer results that can be accurately expressed)
  It's precise - so when the HF32 is off, the problem is in the loader (Return 3/4).

## Ceiling: When should we stop?

**Expressed +fp16/bf16 output rounded to**group,**fp64 cumulative**— 910B hardware not available.
Symptoms: Output is close to zero of the difference between two large items, and ULP of fp16/bf16 is very small in this mass, rounded correctly
Must count to > 24 bit.

The path of false "thinks you can pass" (to save you from retrying):
- **Ozaki/error-free matmul**(indexed pair of slices and unrounded): controlled integer test,
  But true case alignment and close to 2⁄24, more matmul is more frequent at the border, and the average value is not up or down.
- **A simple reduction of chunk / slices**: non-uniform, bottom round to knife-edge twitch without constriction.

Conclusion: small return can stabilize fp64-exact; large return can significantly improve but most apart
**fp32 Hardware + a narrow output rounded up the physical ceiling**with software to compensate for approaching but not securing the line. This is the case.
Just stop. Don't try indefinitely on the Fp32.

## Reuse Decision Tree

```
ReturnaccuracyUncompliant (relationship) fp64/Strict MERE-MARE)
├─ Let's make sure the compressor's not a bottleneck. K / Yeah. fp64 Reference)
├─ matmul/conv and fp32 Enter?→ Turn it off first. HF32(allow_hf32=False)
├─ Small Convention of Return≤~125 What's going on?→ TwoProduct + guarded-Neumaier → fp64-exact  Can pass
├─ Grand Convention of ReturnK~1e3-1e4)?→ chunked-K + Neumaier(+Scratch) → Approaching, lift the average.
└─ It's over. + Thin Output(fp16/bf16)?→ Yes. fp64 Plus, no hardware. → Judge the ceiling. Stop it.
```
