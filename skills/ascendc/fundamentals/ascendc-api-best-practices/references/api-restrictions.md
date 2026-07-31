# Ascend C API Use Restrictions and Alternatives

> **Important**: read before using any API to avoid clerical errors and runtime problems

---

## 1. Compiler period limitation

### 1.1 Prohibition of use of std:: Calculating function

**Reason**: Kernel side does not support the C++ Standard Library and must use the dedicated API provided by Ascend C

**Trigger scene**: all mathematical calculations, comparative operations

**Ban list**:

| std:: Function | Misuse ❌ | ✅ AscendC replacement | Annotations |
|-----------|----------|----------------|------|
| `std::abs` | `std::abs(x)` | `AscendC::Abs(dst, src, count)` | Absolute value [u] |
| `std::min/max` | `std::min(a, b)` | `(a < b) ? a : b` or `AscendC::Min/Max` | Min / max |
| `std::sqrt` | `std::sqrt(x)` | `AscendC::Sqrt(dst, src, count)` | Square root |
| `std::pow` | `std::pow(x, y)` | `AscendC::Power(dst, src, count)` | Logic Operations |
| `std::exp` | `std::exp(x)` | `AscendC::Exp(dst, src, count)` | Index |
| `std::log/log2/log10` | `std::log(x)` | `AscendC::Log/Log2/Log10(dst, src, count)` | logour |
| `std::sin/cos/tan` | `std::sin(x)` | `AscendC::Sin/Cos/Tan(dst, src, count)` | Triangular Functions |
| `std::floor/ceil/round` | `std::floor(x)` | `AscendC::Floor/Ceil/Round(dst, src, count)` | Pickup |
| `std::isnan/isinf` | `std::isnan(x)` | Manual Check | Special value judgement |

**Example of error**:
```cpp
#include <algorithm>
#include <cmath>

uint32_t result = std::min(a, b);  // ❌ Compiler error
float val = std::sqrt(x);          // ❌ Compiler error
float val = std::exp(x);           // ❌ Compiler error
```

**Correct replacement**:
```cpp
// Min/max: Use the tri-channel operator
uint32_t result = (a < b) ? a : b;  // ✅ min
uint32_t result = (a > b) ? a : b;  // ✅ max

// Ascend C API (volume operation)
AscendC::LocalTensor<T> minLocal = minBuf.Get<T>();
AscendC::LocalTensor<T> srcLocal = srcBuf.Get<T>();
AscendC::Min<T>(minLocal, srcLocal, src2Local, count);  // ✅ Minimum batch value

// sqrt/exp/log etc: use Ascend C API
AscendC::LocalTensor<T> dstLocal = dstBuf.Get<T>();
AscendC::LocalTensor<T> srcLocal = srcBuf.Get<T>();
AscendC::Sqrt<T>(dstLocal, srcLocal, count);  // ✅ Square root
AscendC::Exp<T>(dstLocal, srcLocal, count);   // ✅ Index
AscendC::Log<T>(dstLocal, srcLocal, count);   // ✅ logour
```

**⚠ ️ Important**: All mathematical calculations must use Ascend C API, not mix std: function!

### 1.2 Prohibition of dynamic memory distribution

**Reason**: AI Core no dynamic memory management capability

**Trigger scene**: creation of arrays, buffer zones, etc.

**Example of error**:
```cpp
std::vector<int> vec;       // ❌ Dynamical distribution
int* ptr = new int[10];     // ❌ Dynamical distribution
int* arr = malloc(100);     // ❌ Dynamical distribution
```

**Correct replacement**: use of static distribution
```cpp
int arr[10];                          // ✅ Catalyst AllocationHost Side)
constexpr uint32_t SIZE = 1024;       // ✅ Compiler period constant
pipe.InitBuffer(inQueue, 2, SIZE);    // ✅ UB Static distribution(s)Kernel Side)
```

### 1.3 Host/Kernel Header file segregation

**Rule**:
- **Host side**(`.cpp`): Ban the inclusion of `kernel_operator.h`
- **Kernel side**(`.asc/.h`): may contain `kernel_operator.h`

**Example of error**:
```cpp
// host/tiling.cpp
#Include "kernel_operator.h" // ❌ Host side forbidden
```

**Correct use**:
```cpp
// host/tiling.cpp
#Include "tiling.h" // ✅ only headers required
#include <cstring>

// kernel/operator.h
#Include "kernel_operator.h" // ✅ Kernel side allowed
```

---

## 2. API use limit index

The following limitations are detailed in each of the thematic documents:

| Limit Type | Detailed documents | Core elements |
|---------|---------|---------|
| **GM Data Removal** | [api-datacopy.md](api-datacopy.md) | Disable SetValue/GetValue, force DataCopyPad |
| **Reduce API** | [api-reduce.md](api-reduce.md) | dst ≠ tmpBuffer, Disable low step API |
| **Compare 256 bytes aligned** | See below 2.1 | Count requires 256B alignment, padding policy |
| **repeatTime Limit** | [api-repeat-limits.md](api-repeat-limits.md) | uint8_t Maximum 255, batch processing required |
| **pipeline Synchronization** | [api-pipeline.md](api-pipeline.md) | MTE/Vector must sync with EnQue/ DeQue |

### 2.1 Compare API 256 byte binding

**Constraint**: space occupied by `count` elements must**256 bytes aligned**

**Process**: Padding Policy

```cpp
// Calculate alignment size (fload type: multiple of 64)
constexpr uint32_t A0 = 32;
constexpr uint32_t A0_ALIGN = (A0 + 63) / 64 * 64;  // = 64

// 2. UB Buffer Use alignment sizes
pipe.InitBuffer(inQueue, 1, R * A0_ALIGN * sizeof(float));

// CopyIn Filling Polars
Duplicate(xLocal, -FLT_MAX, R * A0_ALIGN);  // ArgMax Use very small
// Copy actual data to first A0 position

// 4. API Call Use Alignment Size
Compare(cmpLocal, srcLocal, maxLocal, CMPMODE::GT, A0_ALIGN);

// Copyout Only Output Valid Data
DataCopy(dstGm, yLocal, A0);  // Output only A0 individual
```

**Polar selection**:
- Arg Max/ find maximum value: `-FLT_MAX` or `-INFINITY`
- ArgMin / Find Minimum: `FLT_MAX` or `INFINITY`

---

## 3. Type & Constant Regulation

### 3.1 Constants for compilation periods

**Rule**: Buffer size, number of cycles, etc. using `constexpr`

```cpp
// ✅ Correct: Compiler-period constant
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_SIZE = 192 * 1024;
constexpr uint32_t BLOCK_SIZE = 32;

// ❌ Not recommended: run-life constant
const uint32_t buffer_num = 2;  // Could affect performance.
```

### 3.2 Type conversion

**Rule**: Visible type conversion to avoid hidden accuracy losses

```cpp
// ✅ Correct: Visible conversion
T sumVal = scalarLocal.GetValue(0);
T invSumVal = (T)1.0 / sumVal;  // Visible conversion to T
Muls<T>(dst, src, invSumVal, count);

// ❌ error: hidden conversion
float val = 1.0 / sumVal;  // if T Yes. half,accuracyLosses
```

---

## 4. Quick diagnostic checklist

Check that, in case of a compilation error:

- [ ] Whether to use**any std:: Calculate function**(min/max/abs/sqrt/exp/log, etc.) → instead of Ascend CAPI or Foundation
- [ ] Whether dynamic memory (`std::vector`, `new`) → is used instead of static distribution
- [ ] Does the host side contain `kernel_operator.h` → to remove this inclusion
- [ ] Whether the dst and tmp of Reduce API are the same buffer → uses different buffer
- [ ] Whether or not to replace the lower API → with the upper Reduce API
- [ ] Whether to use `const` instead of `constexpr` → instead of `constexpr`
- [ ] Whether to use non-existent types (e. g. `TensorShape`) → for correct API access
- [ ] Compare API count satisfies 256 byte alignment → with pedding policy

---

## 5. Relevant documents

- [api-datacopy.md] (api-datacopy.md): DataCopyPad Usage Standard
- [api-reduce.md] (api-reduce.md):Reduce API detailed usage
- [api-repeat-limits.md] (api-repeat-limits.md):repeatTime Limiting and Processing
- [api-buffer.md] (api-buffer.md):Buffer Management best practice
- [api-precision.md] (api-precision.md): accuracy Conversion Standard
