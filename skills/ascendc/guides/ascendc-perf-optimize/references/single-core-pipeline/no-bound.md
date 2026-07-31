# No Bund Optimizing Policy

## Conditions for determination

- None of the hardware modules were fully utilized
- No visible single bottleneck
- Total throughput did not meet expectations

## Elements of a simulation map analysis

- Identify pipeline bubbles and waiting areas
- Check moving - calculate overlap degrees
- Mark UnitFlag Sync Point to see if it can be further reduced

---

## Policy 1: Start PingPong

| Operation | Annotations |
|------|------|
| Double Buffer | Move in and calculate alternately, hide move latency |
| Two-way water organization | Move with Set/Dst Double Queue - Count overlap |
| PingPong Particle Practising | Controls the amount of each move to ensure the time of calculation ≥ moves |

## Policy 2: UnitFlag between MMAD and Fixp

| Operation | Annotations |
|------|------|
| UnitFlag signal | MMAD output → Fixp input syncs with UnitFlag to avoid synchronized costs |
| Decrease Barrier | Substitute visible sync commands with UnitFlag |

## Policy 3: Reduction of single loads

| Operation | Annotations |
|------|------|
| Avoid full migration | Only move the data required for the current calculation |
| Gradual move | It's a matter of counting, no data backlogs. |

## Policy 4: Preload

| Operation | Annotations |
|------|------|
| Move early to the next round of data | Start the next data migration when the current calculation is not completed |
| Pre-command to launch ahead of time. | Use an empty MTE2 window |
| Resize Preload Window | Match calculation time-consuming to avoid moving completed but not calculated |

## Strategy 5: Command to launch early.

| Operation | Annotations |
|------|------|
| Move orders to launch early. | Do not wait for the current calculation to be fully completed |
| Compute command pipeline fill | Reduced pipeline bubbles |

## Policy 6: Vec command integration to reduce duplicate handling

| Operation | Annotations |
|------|------|
| Identification of duplicate move mode | The same data was moved to the scene several times. |
| Combining Vec Operations | Continuous processing in the repository to eliminate intermediate migration |
| One-time migration to multiple reuses | Data presence completes all consumption during UB |

## Strategy 7: Order of screening of low-utilization scenarios

There is no apparent base, which is often not a "no bottleneck", but rather a bottleneck that is dispersed by the start-up cost, synchronization, branching, or small-scale work. It is suggested that the following order be followed:

1. **kernel launch / host setup leads**: many small tensor,foreach or per-row calls should be combined once into Kernel.
2. **CopyIn/Compute/CopyOut does not overlap**:trace gives priority to water streaming instead of blindly increasing the file.
3. **Is there too much Copyout**: multi-line scalar output should be saved first to UB and then written back in bulk.
4. **Whether each file is a fixed branch**: dtype, rank, badcast mode, special value mode should be determined in host or `Init`.
5. **Is there an unused buffer or dead branch**: in the absence of a base scenario, clean-up dead helper, unused TQue, unused include sometimes improves the compilation schedule and memory pressure.
6. **tile is too small**: DMA, barrier and loop overhead will swallow up if each tile compute is too small.

Indication:

```cpp
// Discrepancies: each line is treated separately and the whole is not visible VEC/MTE base.
for (int32_t row = 0; row < rows; ++row) {
  CopyIn(row);
  ComputeSmallRow(row);
  CopyOut(row);
}

// Good: Multi-line approvals, allowing for sufficient workload per round.
for (int32_t rb = 0; rb < rows; rb += ROW_BATCH) {
  CopyInRows(rb, ROW_BATCH);
  ComputeRows(rb, ROW_BATCH);
  CopyOutRows(rb, ROW_BATCH);
}
```

## Policy 8: Do not fine-tune parameters as structural optimization

The common trap of the scene is the resonance of `TILE_LENGTH` or `blockDim`, but the real problem is structural:

| The phenomenon | More likely structural problems |
|---|---|
| Tile makes only a 1% change. | Sync or host setup lead |
| All hardware units have fragments. | Loop is too small or treated separately for each line |
| MTE/VEC is dissatisfied but always time-consuming | Branches, scalar index, queue too many round trips |
| Small case is much slower than the library | Missing speed path for launch number, copyout particle size or special value |

## Tiling Amendments

- Adjust the size of the tile to move -- calculate balance
- Optimizing PingPong Segment Size
- Resize Preload Window and Time Node
- In cases where multiple transfers yield less than noise, they are stopped and rechecked for batches, synchronization, indexing and dead codes.
