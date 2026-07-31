---
name: performance-summary
description: To summarize and compare the results of multiple operator performance tests. Use this Skill when the user requests that multiple operator performance results be aggregated, compared operator benchmark, generated performance reports, and analysed multiple kernel acceleration ratios.
---

# Summary of performance

Summarize and compare multiple performance indicators for operator.

## Workflow

1. **Collecting performance data**
   - Read `verify_<op_name>/result.json` of each operator using `read_file`
   - Extract key indicators: latency_ms, throughput, memory_usage_mb, printup

2. **Generate comparative tables**
   ```
   | operator   | latency (ms) | throughput | Memory (MB) | Accelerated ratio |
   |--------|-----------|--------|-----------|--------|
   | relu   | 0.15      | 1000   | 128       | 2.3x   |
   ```

3. **Analysis**
   - Find Best / Worst operator
   - Mark performance anomalies
   - Optimization of objectives

4. **Generating report**
   Use template: ## test environment # → ## performance comparison # → ### analysis # → # optimization recommendation #

## Data Location

Performance results are stored in the authentication directory:
- `verify_<op_name>/result.json` - Main Results
- `verify_<op_name>/profiling.json` - Detailed profiling (if any)

## Key indicators

| Indicators | Annotations |
|------|------|
| latency_ms | Single execution of latency (ms) |
| throughput | Operation per second |
| speedup | Accelerated ratio of relative baseline (torch) |
| memory_usage_mb | Peak Memory Occupancy (MB) |

## Example:

**User**: "Consolidate the performance of relu and sigmoid operator for me."

**Agent Implementation**:
1. Call `read_file("verify_relu/result.json")`
2. Call `read_file("verify_sigmoid/result.json")`
3. Extract indicators, construct comparative tables
4. Generate summary and optimization recommendation

## Script Call

The `scripts/collect_metrics.py` batch can be used to collect and print a summary of performance:

```python
from collect_metrics import collect_metrics, print_results

# Print formatted performance summary results
print_results()
```

## Skill Authentication

Use `--verify` parameters to verify whether the current skill is available:

```bash
python scripts/collect_metrics.py --verify
```

Validation includes:
- SKILL.md files exist and are formatted correctly
- Core Functions Callable
- Reliance on the availability of modules
- Simulation Executive Tests

**Examples of output**:
```
============================================================
Summary of performance results
============================================================
| Operator | Latency (ms) | Throughput | Memory (MB) | Speedup | Correct |
|----------|--------------|------------|-------------|---------|---------|
| matmul   | 1.234        | 5000       | 256.0       | 1.80x   | ✅      |
| relu     | 0.150        | 10000      | 128.0       | 2.30x   | ✅      |
| sigmoid  | 0.180        | 8000       | 128.0       | 2.10x   | ✅      |
============================================================
```

