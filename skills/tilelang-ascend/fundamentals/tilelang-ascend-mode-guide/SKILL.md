---
name: tilelang-ascend-mode-guide
description: "TileLang Ascend Development/Expert Mode Selection and pass_configs Configuration Guide. Triggers when programming mode, configuration_configs, or conversion between modes are needed. API details refer to tilelang-ascend-api skill."
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
---

# TileLang Ascend programming mode and pass_configs guide

 **API usage details**(RAM allocation, calculation of original language, sync original language, etc.) refer to TileLang Ascend API best practice chapter.

---

## 1. Mode Contrast

| Dimensions | Devloper Mode | Express Mode |
|------|---------------|-------------|
| **Memory distribution** | `T.alloc_shared` / `T.alloc_fragment` | `T.alloc_L1` / `T.alloc_ub` / `T.alloc_L0A/L0B/L0C` |
| **Calculated expression** | `T.Parallel` + Symbol Operations | `T.tile.xxx` Extension Original |
| **Range** | Autoseparate compilerCube/Vector | Manual `with T.Scope("C"/"V")` |
| **Sync** | compiler Autoinsertion | Manual `T.barrier_all` / `T.set_flag` / `T.wait_flag` |
| **pass_configs** | **All started** | **All closed or not set** |
| **Applicable scene** | Most of the operators are compatible across platforms | Extreme performance optimized, requiring bottom control |

**Mix mode**: Devloper main + small Expert / Ascend is exclusive to `T.tile.xxx`. Use Devloper's pass_configs without writing `T.Scope` and manually synchronized. Most actual operator uses hybrids.

---

## 2. Pass_configs Detail (core)

### 2.1 Four Ascend Special Switches

```python
import tilelang

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # ① Automatic Sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # ② AutoMemory Planning
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # ③ AutoCVSeparation
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # ④ AutoNunication
}
```

#### ① TL_ASCEND_AUTO_SYNC (Auto-InnerSync)

- **Key**:`"tl.ascend_auto_sync"`, default False
- **Function**: Automatically insert a sync command such as `T.barrier_all()` between data handling and calculation
- **On start**: no handwritten `T.barrier_all()`, `T.set_flag`/`T.wait_flag`
- **On close**: all sync points must be manually inserted

#### ② TL_ASCEND_MEMORY_PLANNING (Automated Memory Planning)

- **Key**:`"tl.ascend_memory_planning"`, default False
- **Function**: autoanalyse buffer life cycle, achieve memory reuse on film
- **On start**: Auto-reuse buffer space to reduce memory occupancy on film
- **On close**: plan memory address by hand via `T.annotate_address`


#### ③ TL_ASCEND_AUTO_CV_COMBINE (automatic CV separation)

- **Key**:`"tl.ascend_auto_cv_combine"`, default False
- **Function**: Auto-separate the Cube and Victor operations from Kernel to a different execution core
- **On start**: no handwritten `with T.Scope("C")` / `with T.Scope("V")`
- **On close**: `T.Scope` has to be manually marked as the execution field for each segment

#### ④ TL_ASCEND_AUTO_CV_SYNC (Autonuclear Synchronization)

- **Key**:`"tl.ascend_auto_cross_core_sync"`, default False
- **Function**: Automatically insert `T.set_cross_flag`/`T.wait_cross_flag` between CubeScope and VictorScope
- **On start**: no handwritten nuclei sync
- **On close**: the nuclear sync must be managed manually

### 2.2 Selection by scene


| scene | AUTO_SYNC | MEMORY_PLANNING | AUTO_CV_COMBINE | AUTO_CV_SYNC |
|------|-----------|-----------------|-----------------|--------------|
| **pure Victor operator**(elementwise, softmax) | ✅ | ✅ | I don't need it. | I don't need it. |
| **Developer GEMM** | ✅ | ✅ | ✅ | ✅ |
| **Developer Flash AttentionNUCLEARSpipeline)** | ✅ | Depending on the circumstances | ✅ | ✅ |
| **Expert Extreme Performance** | ❌ | ❌ | ❌ | ❌ |
| **Mixed mode** | ✅ | ✅ | ✅ | ✅ |

**Vector operator**:
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

**Developer GEMM**:
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

**Expert Full manual**:
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}
```

---

## 3. Mode conversion rules (Expert → Development)

### 3.1 Conversion steps

1. **Pass_configs**: Add full 4 True switches
2. **Memory distribution**: `T.alloc_L1` → `T.alloc_shared`, `T.alloc_L0C` → `T.alloc_fragment`, `T.alloc_ub` → `T.alloc_shared`
3. **Deleting Field**: Remove `with T.Scope("C")` / `with T.Scope("V")`
4. **Delete Synchronization**: Remove `T.barrier_all()`, `T.set_flag`/`T.wait_flag`, `T.set_cross_flag`/`T.wait_cross_flag`
5. **Calculate conversion**(optional): `T.tile.exp(dst, src)` → `for i,j in T.Parallel(...): dst[i,j] = T.exp(src[i,j])`
6. **Remove Manual Memory Plan**: Remove `T.annotate_address`


### 3.2 Conversion control tables

| Expert | Devloper |
|-------------|---------------|
| `T.alloc_L1(shape, dtype)` | `T.alloc_shared(shape, dtype)` |
| `T.alloc_ub(shape, dtype)` | `T.alloc_shared(shape, dtype)` |
| `T.alloc_L0A/L0B(shape, dtype)` | Delete (`gemm_v0` internal processing) |
| `T.alloc_L0C(shape, dtype)` | `T.alloc_fragment(shape, dtype)` |
| `with T.Scope("C"): ...` | Write direct code (compiler automatically separated) |
| `T.barrier_all()` | Delete (compiler auto-insert) |
| `T.set_flag/T.wait_flag(...)` | Delete |
| `T.set_cross_flag/T.wait_cross_flag(...)` | Delete |
| `T.tile.exp(dst, src)` | `for i,j in T.Parallel(...): dst[i,j] = T.exp(src[i,j])` or retain |
| `T.annotate_address({...})` | Delete (open MEMORY_PLANNING) |

---

## 4. Dveloper vs Expert Mode Code Comparison

---

### 4.1 Devloper Mode

**Features**:
- No `T.Scope`, no `T.barrier_all`, no `T.set_flag`
- Use `alloc_shared` / `alloc_fragment`
- All pass_configs auto-process sync and memory

---

### 4.2 Express Mode

**Features**:
- Manual `T.barrier_all()` Sync
- Specify storage level using `alloc_L1` / `alloc_L0C`
- No pass_configs

---

### 4.3 Express mode pass_configs

Expert Mode Extreme Performance scene,**All off**:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(...):
    ...
```

### 4.4 Developer Nuclear pipeline pass_configs

Nuclear pipeline scene,**All on**:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(...):
    ...
```

---

### 4.5 Mixing mode

Typical scenario for hybrid mode: Development pass_configs + Ascend `T.tile` original (`T.tile.fill/max/sub/exp/div`)

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

# Internal mix of kernelDeveloper and Express API
with T.Kernel(m_num, is_npu=True) as (cid, vid):
    # Expert API: T.tile.fill, T.tile.max, T.tile.sub, T.tile.exp, etc.
    T.tile.fill(acc_ub, 0.0)
    T.reduce_max(scores_ub, row_max_ub, dim=-1)
    T.tile.sub(scores_ub, scores_ub, row_max_ub)
    T.tile.exp(scores_ub, scores_ub)
    T.reduce_sum(scores_ub, row_sum_ub, dim=-1)
    T.tile.div(scores_ub, scores_ub, row_sum_ub)
    # Use Developer's pass_configs for autosync
```

**Key points**: `T.tile.xxx` and `T.reduce_*` can work under Development Pass_configs without handwritten synchronization.
