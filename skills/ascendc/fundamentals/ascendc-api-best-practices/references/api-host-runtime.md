# Runtime API usage norm for the host side

> **Scope of application**:KernelUnder Straight ModeHostSide Code`.asc`Medium`main()`function)

---

## 1. device Initialize API Call Order ⚠Z2XQ**Forced**

### 1.1 Quantities to get API selection ⚠Z1XQ**Key**

**Select the correct number for API**according to operator type:

| operator Type | API used | Annotations |
|---------|-----------|------|
| **pure vector calculations**(Add /Mul/Div/Reduce et al.) | `ACL_DEV_ATTR_VECTOR_CORE_NUM` | Number of using Victor Core |
| **Matrix calculations**(MatMul/Conv et al.) | `ACL_DEV_ATTR_CUBE_CORE_NUM` | Number of Cube Cores |
| **Mixed calculations** | `ACL_DEV_ATTR_AICORE_CORE_NUM` | Number of AI Cores used |

**910B3 chip core reference**:
- AI Core: 20
- Cube Core: 20
- Victor Core: 40 (2 Victor Cores per AI Core)

### 1.2 aclrtGetDeviceInfo call request

**Rule**: `aclrtGetDeviceInfo`**Must**be called after `aclrtSetDevice`

**Reason**: device resources must be captured before setting the context of device

**Correct example**(pure vector operator):
```cpp
int32_t main() {
    // 1. Initialization ACL
    aclInit(nullptr);
    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSetDevice failed, ret=%d\n", ret);
        return ret;
    }

    // 2. Obtain device (must be after aclrtSetDevice)
    int64_t availableCoreNum = 8;  // Default value
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    if (ret != ACL_SUCCESS) {
        printf("aclrtGetDeviceInfo failed, ret=%d\n", ret);
        aclrtResetDevice(deviceId);
        return ret;
    }

    // 3. Calculation of the use of cores
    uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

    // 4. Follow-up
}
```

**Example of matrix operator**:
```cpp
// Matrix calculates the number of Cube Core used by operator
int64_t availableCoreNum = 8;
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &availableCoreNum);
```

**Example of error**:
```cpp
int32_t main() {
    // ❌ error: call aclrtGetDeviceInfo without calling aclrtSetDevice
    int64_t availableCoreNum = 8;
    aclError ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    // Could return an error or get an error value
}
```

---

## 2. Common Errors

| Error Type | Example of error | Consequences | The right way. |
|---------|---------|------|---------|
| **Call order error** | Call `aclrtSetDevice` without calling `aclrtGetDeviceInfo` | Could not close temporary folder: %s | `aclrtSetDevice` first, then get resources. |
| **Write death toll** | `uint32_t numBlocks = 8;` | Different device performance does not match | Use `aclrtGetDeviceInfo` dynamic access |
| **API selection error** | Just vector operator for `ACL_DEV_ATTR_AICORE_CORE_NUM`. | Underutilization of Victor Core | Select the correct API based on the operator type |

---

## 3. Full Host Side Initialisation Process

```cpp
int32_t main() {
    // Step 1: Initialize ACL
    aclInit(nullptr);

    // Step 2: Setup device
    int32_t deviceId = 0;
    aclError ret = aclrtSetDevice(deviceId);
    CHECK_ACL(ret);

    // Step 3: Get device number (selected according to operator type)
    int64_t availableCoreNum = 8;
    // Just vector operator.
    ret = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
    // or matrix operator: ACL_DEV_ATTR_CUBE_CORE_NUM
    // or mixed operator: ACL_DEV_ATTR_AICORE_CORE_NUM
    CHECK_ACL(ret);

    // Step 4: Distribute GM RAM
    size_t gmSize = ...;
    void* gmPtr = nullptr;
    ret = aclrtMalloc(&gmPtr, gmSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_ACL(ret);

    // Step 5: Calculating Tiling Parameters
    MyTilingData tiling;
    uint32_t numBlocks = (uint32_t)availableCoreNum;
    computeTiling(tiling, totalRows, numBlocks);

    // Step 6: Start Kernel
    KernelCall(..., (uint8_t*)&tiling);

    // Step 7: Cleaning up resources
    aclrtFree(gmPtr);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```

---

## 4. Relevant documents

- **Code review inspection item**: [code-review-checklist.md] (../../ascendc-kernel-develop-workflow/references/code-review-checklist.md) § 0.2.1
