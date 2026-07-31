# Multi-output returned Buffer Planning

> Many of the returned operators need to scan and output multiple results over and over again, more than the single output of the UB. This paper gives the generic Buffer equation.

---

## Common multiple output scenes

| operator | Output Number | Output Contents | Thrusters dtype |
|------|--------|---------|-------------|
| bn_training_reduce | 2 | sum[C] + squareSum[C] | FP32 |
| reduce_var / reduce_std | 2 | mean[A] + var[A] | FP32 |
| reduce_std_with_mean | 2 | std[A] + mean[A] | FP32 |
| arg_max / arg_min | 2 | value[A] + index[A] | FP32 + INT32/FP32 |
| arg_max_with_value | 2 | value[A] + index[A] | Ibid. |

## Universal UB Equation

```
Input:
  T_in  = Enter Element Size (FP16=2, FP32=4)
  T_acc = Threshold element size (Usually. FP32=4)
  K     = Output Number (double output) K=2Three. K=3)
  A_aligned = Resize the axis after alignment

Buffer List:
  inBuf × 2              = tileSize × T_in × 2          ← Inputdouble buffering
  castBuf (Low onlyaccuracy)      = tileSize × T_acc             ← FP16/BF16 → FP32
  accumBuf × K           = A_aligned × T_acc × K         ← K A compressor.
  tmpBuf                 = A_aligned × T_acc             ← Intermediate calculation (e.g.) x²)
  outBuf × 2             = A_aligned × T_acc × 2         ← Outputdouble buffering(optional)

UB Equation:
  tileSize × (T_in × 2 + T_acc)              ← Input + Cast
  + A_aligned × T_acc × (K + 1 + 2)          ← K Composer + tmp + outdouble buffering
  ≤ UB_SIZE (DAV_2201: 192KB)

Solution tileSize:
  fixedBuf = A_aligned × T_acc × (K + 3)
  perTileBuf = T_in × 2 + T_acc
  tileSize = (UB_SIZE - fixedBuf) / perTileBuf
```

## Double output example: bn_training_reduce (K=2, FP16 input)

```
A_aligned = CeilAlign(C, 8)        ← FP32 Press 32B Alignment: 32/4=8
T_in = 2 (FP16), T_acc = 4 (FP32)

fixedBuf = A_aligned × 4 × (2 + 3) = A_aligned × 20
  Decomposition: sumBuf(A×4) + sqSumBuf(A×4) + tmpBuf(A×4) + outBuf×2(A×4×2)

perTileBuf = 2 × 2 + 4 = 8   (FP16 double buffering + FP32 Cast)

tileSize = (192KB - A_aligned × 20) / 8

Example:: C=64 → A_aligned=64
  fixedBuf = 64 × 20 = 1280B
  tileSize = (196608 - 1280) / 8 = 24416 Elements
  Actual tileRows = tileSize / A_aligned = 381 Okay.
```

## Double output example: Arg Max (K=2, value + index)

```
A_aligned = CeilAlign(A, 8)

Composer:
  maxValBuf: A_aligned × 4 (FP32)      ← Current max
  maxIdxBuf: A_aligned × 4 (FP32)      ← Subscript of current maximum value (stored as) float)

Extra:
  cmpBuf: A_aligned / 8 (uint8_t)      ← Compare mask
  Attention.: DAV_2201 Let's go. Select Not supported int32 dst → Synchronising folder floatAnd finally, Cast as int32

fixedBuf = A_aligned × 4 × 2 + A_aligned × 4 + max(A_aligned/8, 32) + outBuf
```

## Workspace Equation for Trans-nuclear Merge

Multi-output workspace also times K:

```
workspace = coreNum × CeilAlign(A_aligned × T_acc × K, cacheLineSize)

Example:: bn_training_reduce, C=256, 20Nuclear
  workspace = 20 × CeilAlign(256 × 4 × 2, 256) = 20 × 2048 = 40KB
```

---

## NCHW layout data moving note

The memory of NCHW layouts is not continuous by channel:

```
NCHW Memory Layout: x[n,c,h,w] Address = n×C×H×W + c×H×W + h×W + w
  The same. (n,h,w) of C Longness of each channel = H×W(inconsistent!
  The same. (n,c) of H×W A continuous space position.

Two treatments.:
  ModalitiesA(Recommended,DAV_2201/DAV_3510 General): All the way through the tunnel, on the outside. (n, c)The inner layer is moving continuously. H×W
    → Continuous memory for each removal (efficient), required C A stand-alone loader
    → It suits most scenes.

  ModalitiesB(only DAV_3510): Use it. NDDMA Multi-dimensional handling, one configuration auto-processing stride Jump.
    → DAV_2201 Not supported NDDMA

  ModalitiesC(DAV_2201): DataCopyPad Configure stride Parameters
    → blockCount=R Lines, blockLen=Snippet bytes of series, srcStride=Long jump.
    → It's for regulars. stride Mode

Actual batch_norm_v3 ApproachA: We'll go through every channel. We'll add it up. N×H×W Continuous value [v]
  See /ascendc-api-best-practices Access DataCopyPad stride Configure Details
```
