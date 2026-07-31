---
name: ascendc-cube-matmul-mix
description: Ascend C Cube(Matmul/Volume/Attention)operator MIXStart a fight with landing. Coverage: How to getCubeIt's really nuclear.NPUI've been executed.profilerCapture (rather than running)AIV/vector),MultiCoreMatmulTiling of baseK/SetSingleShape/Multi-nucleotide traps,L0A/L0B/L0CCapacity constraints,workspaceCombination,MTE2→VSynchronize,Release/DebugCompile differences. Trigger keyword:matmul,cube,AIC, mix nuclear,MIX,GetTensorC,Iterate,REGIST_MATMUL_OBJ,ml0/nl0/kl0=0,blockDimDeadlock.L0BSpills, all results.0/NaN,no_npuQuantified matrix multiplier,grouped_matmul,attention.
---

# Ascend C Cube operator MIX Start and Land

If the Cube (AIC) operator is pulled up with the "vector Nuclear Launch" mode, the AIC phase does not perform → at all, and the results are zero/not initialized, and profiler cannot capture NPU Kernel. The following is the eight key points where Cube operator is actually running, counting and captured, and any error is silently produced "0 / NN/ Deadlock/ accuracy error".

## 1. Start with MIX instead of vector-only
- Naked Chevron `<<<blockDim,l2Ctrl,stream>>>` is**AIV-only**, Cube phase is not running.
- Correct manner: `ascendc_library()` (`$ASCEND_HOME/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake`) compiles kernel, produces `aclrtlaunch_<kernel>.h`, pulls up (MIX) with `ACLRT_LAUNCH_KERNEL(<kernel>)(blockDim, stream, args...)`. Or moves the standard op framework (op_host+op_kernel+ops-info registration).
- PureCXXThe project will be sufficient:`include(ascendc.cmake); ascendc_library(k STATIC kernel.cpp)`.**Don't.**and ASC-language(`.asc`/`find_package(ASC)`)Mixed——Coexistence can be destroyed in silence.device-binaryRegisterkernelPull up but not executed.`set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH} FORCE)`.

## 2. Workspace / Tiling must be the last two parameters of Kernel
framework takes sys workspace from the "second last" parameter to `GetSysWorkSpacePtr()` (`REGIST_MATMUL_OBJ` uses it). L1/L0 status of → matmul is written in the wrong order into a small Tiling Buffer → MTE crosses the border. The end of the signature is fixed as `..., workspace, tiling`.

## 3. Tiling (MultiCoreMatmulTiling)
- `SetFixSplit(baseM, baseN, baseK)`'s**baseK must be visible**: Pass `-1` (auto) get `ml0/nl0/kl0=0` → under int8.
- `SetTraverse(MatrixTraverse::FIRSTM)`: There is no way that the Iterate is in the wrong position for → in the sequence with the Kernel handwritten `(cnt%roundM, cnt/roundM)`.
- `SetSingleShape(sM, sN, K)`: Each core is responsible for the `sM×sN` area. >1 GetTiling often needs it to succeed (otherwise `res=-1`, baseM/baseN=0).

## 4. BlockDim ≤ Physical AIC Number (otherwise MIX starts deadlock)
`blockDim > Physical AIC '**hangs**(not drops). tile cannot tile a core in many cases; instead, the `SetSingleShape` area is amplified by base increment until `area number ≤ Physical AIC ', with multiple base file (`roundM=ceil(singleCoreM/baseM)`) per core within the area. `get_usedCoreNum()` is blockdim.

## 5. L0 Capacity (int8, 64KB per block L0)
`baseM·baseK ≤ 64K` (L0A), `baseK·baseN ≤ 64K` (L0B), `baseM·baseN·4 ≤ 128K` (L0C, in32). `128×128×512` is perfect. Common pit: `baseN=256 + baseK=512` → L0B = 128K spills out → more than K-chunk shape**silently miscalculated**.

## 6. MTE2 → V Sync (load scale/bias followed by vector operation)
`DataCopy`(MTE2  Like a  per-channel scaleCopyUBI'll be right back.`Mul`(V),**I have to.**Use it.VECIN Que(`AllocTensor→DataCopy→EnQue→DeQue→Use it.→FreeTensor`)or `SetFlag/WaitFlag<HardEvent::MTE2_V>`Naked.`PipeBarrier<PIPE_V>`Lines onlyVOrders, no queuesMTE2→V → MulRun away!→SmallM shape NaN(Big)MAs a resultCastCover up.latencyI'm lucky, that's why.bugIt's hidden.

## 7. Cube Kernel compiled with Debug
`set(CMAKE_BUILD_TYPE Debug ... FORCE)`. Ascendc cube kernel meets under Release `-O2` (device.o merge; the official mlex example is also mandatory for Debug.

## 8. Torch_npu series of traps
`x[bi]` (several slices of tensor) is still accompanied by `storage_offset`, while `ConvertType`/`storage().data()`**ignores that offset**→ has been read in each batch to catch 0. Use `.clone()` (compulsory copy of fset=0) when pulling up in separate batch, not with `.contiguous()` (repeated as it is).

## Kernel Bones (the example of leakyrelu that can be copied)
`Matmul<A,B,C[,Bias]>` +Handwritten cycle:`SetTensorA/B; if bias SetBias; while(mm.Iterate<true>()){ GetTensorC<true>(local,false,true); /*vectorReprocessing*/ CopyOut; } mm.End();`;entry`REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling)`. Example:CANNIn the bag.`matmul_leakyrelu_custom`(fp16).int8Quantification:`Matmul<int8,int8,int32,int32-bias>` cubeOutint32Again.vectorInverse Quantificationcast→scale→pertoken→clamp→cast half).int8WeightsBReferences`CubeFormat::NZ`,but NDOr (internal)nd2nz).

Relevant: [[ascendc-hardware-tilling] [[ascendc-ub-budget] [[ascendc-crash-debug]](deadlock/total 0/NAN positioning)].
