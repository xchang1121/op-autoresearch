# accuracy debug best practice

## Debugging principles

### 1. I doubt the formula, then the formula.

Most of the problems of accuracy are problems of mathematics/calculations per se, not errors of realization.

**The order of examination**:
1. Is the mathematical formula correct?
2. Whether the formula to code map is correct
3. API is correctly used
4. Whether data type matches
5. Numerical stability issues

### 2. Start with a small sample.

Test pyramid:
```
     ┌────────────┐
     │  Full Test   │  ← Final Authentication
     ├────────────┤
     │  Medium size   │  ← Expand Authentication
     ├────────────┤
     │  Small     │  ← Basic Authentication
     ├────────────┤
     │  Modular     │  ← First test.
     └────────────┘
```

### 3. Record everything.

Every change must be recorded:
- Modify content
- error Change
- Performance Impact

```markdown
### Debug Log

| Modify | Max Error | Mean Error | Performance |
|-----|-----------|------------|------|
| Initial | 1.2e-2 | 3.5e-4 | - |
| FP32 Composer | 2.1e-4 | 1.2e-6 | -5% |
| Numerical stabilization algorithm | 8.5e-6 | 2.3e-7 | -8% |
```

### 4. That's how you figure it out.

Try not blindly, understand the source of error:
- Why is there error?
- What step does error come from?
- How can the root causes be avoided?

## Validation Policy

### Test the pyramid

```
          ┌────────────┐
          │  Border tests   │  ← Small but important
          │  Special value     │
          ├────────────┤
          │  Random Test   │  ← Unknown problem detected
          │  Various sizes   │
          ├────────────┤
          │  Reunification Test   │  ← Ensuring that there is no backsliding
          │  Standard use examples   │
          └────────────┘
```

### Boundary overwrite

| Type | Test Value | Annotations |
|-----|-------|------|
| zero value | 0.0 | Test Zero Processing |
| Very small | 1e-10 | Surrounding border |
| Small value | 1e-6 | Normal small value |
| Normal value | 1.0 | Typical value |
| Great value | 1e6 | Normal high value |
| Extreme value | 1e10 | Surround the border |
| Negative value | -1.0 | Negative Number Processing |
| FP16 Saturation | 65504.0 | FP16 Max |

### Alignment Overwrite

| Test Type | Example: shape | Purpose |
|---------|---------|------|
| 32 Byte Alignment FP32 | (8, 16, 8) | Exclude alignment issues |
| 32 Byte Alignment FP16 | (8, 16, 16) | Exclude alignment issues |
| Inconsistent | (8, 17, 9) | Test non-matching |
| Hardware bound borders | (8, 8, 8) | Minimum number of elements |

## Preventive measures

### Design phase

1. **Choosing algorithms with stable values**
   - Avoid a-b when a≈b
   - Use log-sum-exp to avoid spilling
   - We'll do it in a single way.

2. **Planning for a mix of accuracy programmes**
   - FP16 Input/Output
   - Critical Intermediate Calculation FP32
   - Aggregation Operations FP32

3. **Understanding hardware constraints**
   - Data Alignment Requirements
   - Minimal number of elements bound
   - Single processing cap

### Encoding Phase

1. **Key path note**
```cpp
// accuracy Sensitivity: Use FP32 to avoid accumulation of error
float sum_fp32 = 0.0f;
for (int i = 0; i < size; ++i) {
    sum_fp32 += static_cast<float>(input[i]);
}

// accuracy Sensitivity: Decreasing max to avoid exp spilling
half max_val = ReduceMax(input);
half shifted = input[i] - max_val;
```

2. **Enter Authentication**
```cpp
// Check hardware constraints
if (cols < 8) {
    printf("Error: cols must be >= 8 (got %d)\n", cols);
    return;
}

// Check range of values
for (int i = 0; i < size; ++i) {
    if (isinf(static_cast<float>(input[i]))) {
        printf("Warning: input[%d] is Inf\n", i);
    }
}
```

3. **Debug support**
```cpp
#ifdef DEBUG_PRECISION
    printf("Step %d: value=%.6f\n", step, static_cast<float>(value));
#endif
```

## Portability Settings Guide

### Recommended tolerance

| scene | rtol | atol | Annotations |
|-----|------|------|------|
| **FP16 Standard** | 1e-3 | 1e-4 | Floating point accuracy limited |
| **FP32 Standard** | 1e-5 | 1e-6 | Standard accuracy |
| **Integer** | - | 0 | It has to match exactly. |

### It's a special scene.

| scene | rtol | atol | Reason |
|-----|------|------|------|
| Softmax FP16 | 1e-3 | 1e-4 | Probability output, accuracy limited |
| Softmax FP32 | 1e-5 | 1e-6 | Standard accuracy |
| Reduce FP16 | 5e-3 | 1e-4 | error is bigger. |
| Reduce FP32 | 1e-5 | 1e-6 | Standard accuracy |
| exp/log FP16 | 1e-3 | 1e-4 | Exceed function accuracy limited |
| Triangular function FP16 | 5e-3 | 1e-4 | Tyler's starting to look like one. |

### Discretion selection principle

1. **Based on the application scene**
   - Scientific calculations: strict requirements
   - In-depth learning: relative easing
   - Visualization: the most liberal

2. **Based on data type**
   - FP16: Easier tolerance
   - FP32: Standard tolerance
   - INT: Zero tolerance

3. **Based on operator properties**
   - Element-by-Element Operations: Standard Perceptions
   - Reduce Operation: Easier tolerance
   - Beyond function: Easier tolerance

## Common error mode

### Error 1: Overbug

**Symptoms**: repeated attempts at different methods, not systematic

**Correct practice**
1. Analyse error mode first.
2. Identification of possible causes
3. Targeted Authentication
4. Record every attempt

### Error 2: Ignore FP32

**Symptoms**: Directly debugging with FP16, difficult to distinguish from accuracy

**Correct practice**
1. Use FP32 first to verify the correctness of the algorithm
2. Retest FP16 accuracy
3. Comparison of differences in location issues

### Error 3: Blindly raise accuracy

**Symptoms**: FP32 is used for all calculations, performance declines

**Correct practice**
1. Identify the steps that really need high accuracy.
2. Use FP32 only at critical steps
3. Balance accuracy and performance

### Error 4: Ignore hardware constraints

**Symptoms**: abnormal results of specific input sizes

**Correct practice**
1. Access to hardware binding documents
2. Add Input Authentication
3. Description of limits in document

## Debug efficiency techniques

### 1. Diagnosis first.

```
Problem → errorAnalysis → Mode Recognition → Genesis. → Targeted repairs
```

### 2. Quick Validation

```
Modular → Small → Standard size → Large → Boundary value
```

### 3. Count rules

```
Quick method count < 7 → Go on.
Quick method count >= 7 → Toggle Debugging
```

### 4. Record template

```markdown
## Debug Record

### Problem
- Symptoms:
- Input size:
- data type:

### error analysis
- Maxerror:
- Averageerror:
- errorMode:

### Debug process
1. Try:
   - Methodology:
   - Results:
   - Count:

2. Try:
   - Methodology:
   - Results:
   - Count:

### Solutions
- GEN:
- Restoration:
- Authentication:
```

## Performance balanced with accuracy

### Mixed accuracy design

| Widget | accuracy Selection | Reason |
|-----|---------|------|
| Input | FP16 | Save bandwidth |
| Output | FP16 | Save Storage |
| Intermediate calculation | FP32 | Raise accuracy |
| Composer | FP32 | Avoid accumulation of error |
| Compare | FP16 Enough | Without prejudice to accuracy |

### Performance assessment

Assess the performance impact after repairing the accuracy problem:
- Performance loss < 5 per cent: acceptable
- Performance losses 5-10%: need to be assessed
- Performance loss > 10%: Consider optimisation options

## Documentation

### operator documents should contain

1. **Annotations to accuracy**
   - Supported data type
   - accuracy characteristics of each data type
   - Recommended tolerance settings

2. **Known limitations**
   - Hardware constraints (matching, minimum number of elements)
   - accuracy limit (FP16 valid number)
   - Numerical range limit

3. **Use of recommendations**
   - Recommended data type combination
   - Input mode to avoid
   - Performance optimization tips
