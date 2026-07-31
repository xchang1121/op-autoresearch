### tl.swizzle2d(i, j, size_i, size_j, group_size)
```python
task_i, task_j = tl.swizzle2d(block_i, block_j, NUM_BLOCKS_I, NUM_BLOCKS_J, GROUP_SIZE)
```
- **Parameters**:
  - `i`, `j`: Raw block index
  - `size_i`, `size_j`: Total number of blocks
  - `group_size`: Group Size (usually 2/4/8)
- **Return**: Renumbered block index (task_i, task_j)
- **Use**: 2D block reset to increase cache locality
- **Applicable scene**: matrix multiplication multi-dimensional block calculations to improve data reuse
- **Note: Priority (i) grouping is supported only, and priority is manually achieved
