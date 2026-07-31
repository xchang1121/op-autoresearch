# Diagnosis workflow for accuracy problems

## Systematizing debugging process

```
┌─────────────────────────────────────────────────────┐
│ 1. Problem positioning: ConfirmedaccuracyQuestion or functional error?              │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ 2. errorAnalysis: analysiserrorDistribution and Mode                      │
│    - StatisticserrorCharacteristics (maximum, average, distribution)              │
│    - IdentificationerrorMode (systemic)/Random/Boundary)               │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ 3. Minimise recurrence: construct the smallesttest case                      │
│    - Single, frontier, special                          │
│    - Gradually increasing complexity                                  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ 4. Intermediate results check: step-by-step comparison of each calculation step                │
│    - Printf Output middle value                               │
│    - CPU vs NPU Contrast                                 │
│    - Positioning First Difference                                   │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ 5. Root analysis: understanding of differences                            │
│    - API accuracyRestrictions?                                   │
│    - Numerical stability problem?                                 │
│    - Achieving logical issues?                                   │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ 6. Solutions: implementation of targeted repairs                          │
│    - Adjust the order of calculation                                    │
│    - Raise Critical Pathaccuracy                                │
│    - Use stabilization algorithm variants                                │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ 7. Validation of restoration: comprehensive testing ensures resolution                        │
│    - Existing issuestest case                                │
│    - Extensiontest case                                    │
│    - Performance impact assessment                                    │
└─────────────────────────────────────────────────────┘
```

## Detailed step note

### Step 1: Problem positioning

**Target**: confirmation of accuracy problem or functional error

| Type | Features | Direction |
|-----|------|---------|
| Function Error | The result was completely wrong. The number was wrong. | Check formula realization, API call |
| System deviations | Oversize/minus, misproportion | Check constants, coefficients |
| accuracy Question | It's close, but there's error. | accuracy debugging process |
| Border issues | Error in specific range values | Check special value processing |

### Step 2: error analysis

**error mode recognition**:

```python
import numpy as np

# Basic error statistics
errors = abs(pred - truth)
print(f"Maxerror: {errors.max():.6f}")
print(f"Averageerror: {errors.mean():.6f}")
print(f"95Divisionerror: {np.percentile(errors, 95):.6f}")
print(f"Mediumerror: {np.median(errors):.6f}")

# Find the largest sample of error.
worst_idx = errors.argmax()
print(f"The worst sample.: idx={worst_idx}, pred={pred[worst_idx]}, truth={truth[worst_idx]}")

# error analysis
rel_errors = abs(pred - truth) / (abs(truth) + 1e-9)
print(f"Maximum relativeerror: {rel_errors.max():.2e}")
print(f"Average relativeerror: {rel_errors.mean():.2e}")

# error distribution analysis
print(f"error > 1e-3 Percentage: {(errors > 1e-3).sum() / len(errors) * 100:.2f}%")
print(f"error > 1e-4 Percentage: {(errors > 1e-4).sum() / len(errors) * 100:.2f}%")
```

**error model judgement**:
- **System deviation**: error Synonyms (all positive or all negative) → formulae/consistent problem
- **Random error**: error symbol randomly and evenly distributed → numeric accuracy question
- **Gather.error**: range of specified valueserrorLarge→Border conditions/Special value treatment issues
- **Slender-sized error**: Mostly correct, individual error → special input trigger

### Step 3: Minimize recurrence

**Sequence of tests**:

```python
# 1. Modular test (simplified)
test_input = np.array([1.5], dtype=np.float32)

# 2. Small-scale alignment tests (32 byte alignment + FP32)
test_input = np.random.rand(8, 16).astype(np.float32)  # Endaxis16=8*4Bytes

# 3. Boundary values test
boundary_cases = {
    "zero value": 0.0,
    "Very small": 1e-10,
    "Small value": 1e-6,
    "Normal value": 1.0,
    "Great value": 1e6,
    "Extreme value": 1e10,
    "Negative value": -1.0,
    "FP16Saturation": 65504.0,
}

# 4. Non-matching tests
test_input = np.random.rand(8, 17).astype(np.float32)

# 5. FP16 accuracy test
test_input = np.random.rand(8, 16).astype(np.float16)
```

### Step 4: Intermediate results check

See [binary-search-debug.md] (binary-search-debug.md) Debug Detailed Guide

### Step 5: Root analysis

**Common root cause classification**:

| Root Type | Typical performance | Check the direction. |
|---------|---------|---------|
| API accuracy Limit | FP16 error is clearly greater than FP32 | data type accuracy's not enough. |
| Numerical stability | Inappropriate handling of large/minus | Algorithmic numerical stability |
| Achieving logic | Error with specified input mode | Border conditions |
| Hardware constraints | Error on a specific scale | Alignment, minimum elements |

### Step 6: Solutions

See [common-traps.md] (common-traps.md)

### Step 7: Validation of repairs

**Validation list**
- [ ] Question test case adopted
- [ ] Border values tested.
- [ ] Multiple input scale tests passed
- [ ] FP16 and FP32 authentication
- [ ] Performance effects are acceptable
- [ ] The regression test passed.

## Debug count rules

```
Debug counter = 0

Quick Method List (Quick Method Counts per Attempt)+1):
1. errorDistribution analysis
2. Printf Organisation
3. Inspection FP16 accuracyA trap.
4. Inspection exp/log Spill
5. Check reduction offsets
6. Inspection Reduce accuracy
7. Check for zero.
8. Check hardware constraints

Time counter >= 7 Or all fast-track methods have been tried:
    → Switch to Half Debug
```
