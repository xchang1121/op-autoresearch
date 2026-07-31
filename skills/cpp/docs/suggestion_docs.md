# CPU C++ Expert skills and optimization recommendation

This document provides guidance on techniques, performance optimization and problem mapping developed by CPU C++.

## 1. Performance optimization

### data type selection policy
- **Priority support**: float 32, float64, int 32, int64
- **Autoconversion**: non-supported type automatically converts to float32
- **Type check**: Check tensor type using `scalar_type()`
- **pointer acquisition**: using the corresponding `data_ptr<T>()` method

### Cycle Optimization Policy
- **Cycle expansion**: manual roll-out cycle reduces branch costs
- **Cache friendly**: access data according to line priority
- **Boundary processing**: vector processes most of the data, scalar handles the remaining elements
- **Avoidance of branches**: reduced conditionalities and more efficient implementation

### Memory access optimization
- **Consequencing inspection**: ensure continuous RAM layout of tensor
- **Data locality**: access to adjacent memory as much as possible
- **Avoid copy**: Reduce unnecessary tensor copying operations
- **On-site operations**: change on-site as far as possible

## 2. Numerical stability technique

### Spill-proofing
```cpp
// Minus maximum value before integration
float max_val = *std::max_element(x_ptr, x_ptr + numel);
for (int64_t i = 0; i < numel; ++i) {
    float stable_data = x_ptr[i] - max_val;
    out_ptr[i] = std::exp(stable_data);
}
```
### accuracy Upgrade
- **Intermediate calculation**: accuracy does not have to be raised to dooble at calculation, use the float type
- **Accumulation operation**: using the Kahan Summation Method to prevent the loss of accuracy
- **Avoiding Numeric Spill**: Checking Off-Zero and Open Operations
- **Typologies conversion**: careful handling of conversions between accuracys

## 3. Programming constraints and best practice

### Rules to be followed
- **Boundary check**: all arrays must check the border before visiting
- **Type security**: ensure that the type of pointer matches the type of tensor
- **Unusual security**: use of RAII to manage resources
- **data type support**: priority support for commonly used types, automatic conversion of other types

### Principles of kernel design
- **Single function**: One thing for each function
- **Parameters Simple**: Avoid complex data structure transfer
- **Memorial Locality**: access to adjacent memory as much as possible
- **Avoid dynamic distribution**: kernel avoidancenew/delete

### OpenMP Parallel Programming Constraint
- **⚠️ Key Constraint**: OpenMPruntimeAPI call position limit
  - **No scenes allowed**:Don't be here.SIMDRegional, parallel regionsintervening codeMedium Call`omp_get_thread_num()`Wait.OpenMPruntimeAPI
  - **Correct use**: OpenMP API should be called in the normal code path in the parallel area, not in the restricted context of compiler
  - **Example of error**:
    ```cpp
    // ❌ error: Call OpenMP API in restricted context
    std::mt19937 gen(seed + omp_get_thread_num());  // Compiler error!
    ```
  - **Correct example**:
    ```cpp
    // ✅ Correct: Normal Call within Parallel Area
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();  // Correct.
        std::mt19937 gen(seed + tid);
    }
    ```
- **Language security**: ensuring an example of an independent random number generator for each thread
- **Data competition**: Avoid multiple threads writing the same memory position at the same time

## 4. Debugging and queuing lists

### Memory access issues
- [ ] Do all array visits have border checks?
- [ ] Is tensor continuous?
- [ ] Does the pointer type match the tensor type?
- [ ] Are there cross-border visits?

### Types of treatment of problems
- [ ] data type check correctly?
- [ ] Is the type conversion safe?
- [ ] Is the output type consistent with the input?
- [ ] Are all types of support addressed?

### Performance issues
- [ ] Is a cycle being used to expand?
- [ ] Do memory access continue?
- [ ] Were unnecessary copies avoided?
- [ ] Is it properly optimized?

## 5. Frequent Error Scanning

| Error Type | Problem | Solutions |
|---------|------|---------|
| Cross-border visits | Paragraph error or abnormal result | Add Border Check |
| Type does not match | Error compilation or runtime error | Check the pointer type |
| Non-continuing visits | Performance drops or wrong results | Ensure the continuity of tensor |
| Memory Leak | Process memory continues to grow | Avoid manual memory management |


## 6. Development proposal

### The code generated is referred to as the embedded C++ code in the Python module.

### Code Style
- Attention! The resulting code does not contain any test code.
- Attention! The embedded C++ code must be written in the embedded script to ensure that it is properly compiled and loaded.
- The embedded C++ code uses a three-quote string to keep the format clear
- Add sufficient comments to explain the calculation logic
- Use descriptive variable and function names
- Uniform error-processing mode