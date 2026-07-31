# Scalar High Performance Coding Principles

This document gives 9 coding principles for Scalar's high performance operator. The format of each principle is as follows:

- **Rationale**: Why does it make Scalar run faster?
- **Identifier**: Key signals of this reverse pattern are identified in existing codes
- **cross-referenced / good example**: a code template that can be directly cross-referenced (unreal operator code for mode recognition)

> Common objectives of all principles:**Helping compiler to better distribute the register, analyse aliases and disseminate constants, thereby reducing the Load/Store Directive**.

---

## P1 use arrays carefully in the structure

### Rationale

An array of members in the structure, if accessed through**dynamic subscript**(runtime variable), would have failed in compiler's**alias analysis: compiler cannot determine the specific value of the subscript, and must consider the worst case - the subscript may have been calculated to point precisely to the other members of the structure. To ensure correctness, compiler**would have abandoned the constant transmission of all members of the entire structure**and reread it from memory. The contamination would have spread upwards along inheritance/inclusion relationships.

### Identification points

- There are also array members and other members in the class/structure who need constant optimization
- Access to the members of the arrays by subscripting runtime variables (e. g. ping-pong index, loop variable)
- The phenomenon of the "continent parameters that should have folded but still walk Load" was observed in the programring

### Inversion

```cpp
class Block {
    event_t eventIds_[2] = {EVENT_ID0, EVENT_ID1};   // Members of arrays
    event_t eventIdsM_[2] = {EVENT_ID2, EVENT_ID3};
    uint16_t pingPongId_ = 0;                         // runtimeSubscript
    uint32_t tileM_;                                  // Contaminated with other members of the structure.
};

void Block::Step() {
    SetFlag<HardEvent::V_MTE2>(eventIds_[pingPongId_]);  // Dynamic Subscript
    pingPongId_ ^= 1;
    // Even if you can infer a constant during the compilation period, you'll go conservatively, Load.
}
```

### - Yes.

```cpp
class Block {
    event_t eventId_ = EVENT_ID0;                     // Split to Independencescalar
    event_t eventIdM_ = EVENT_ID2;
    uint16_t pingPongId_ = 0;
    uint32_t tileM_;
};

void Block::Step() {
    SetFlag<HardEvent::V_MTE2>(eventId_);
    eventId_   = (eventId_   == EVENT_ID0) ? EVENT_ID1 : EVENT_ID0;
    eventIdM_  = (eventIdM_  == EVENT_ID2) ? EVENT_ID3 : EVENT_ID2;
    pingPongId_ ^= 1;
}
```

---

## P2 Cycle main tail block separation

### Rationale

Thermal routing (e.g.)CubeCategoryoperatorof K-Reduce  If in a single  `for`Adopted`if`Handle first, periodic events, tails, then**It's done every turn of time.3Subdivision judgement**.ScalarThe branch predictor is simpler, yes."Most of them are.falseBut once in a while,true"The model predicts poor results; the cost of each failure in the heat path is amplified.

### Identification points

- `if (k == 0)`, `if (k == maxK - 1)`, `if (k % N == 0)` etc. in the heat path cycle
- Large number of cycles (tens-hundreds) with MTE2 periodic load or first-end special treatment

### Inversion

```cpp
for (uint64_t k = 0; k < maxK; ++k) {
    if (k == 0)               { SetInitialParams(); }
    if (k % loadStride == 0)  { MTE2_LoadAL1(); }
    if (k == maxK - 1)        { SetFinalParams(); }
    UpdateParams(k);
    Compute();
}
```

### - Yes.

```cpp
// Prologue: Handle header
SetInitialParams();
UpdateParams(0);
Compute();

// Hot Loop: zero branch, pure calculation and parameter update only, cycle events wrapped in the outer layer
uint64_t k = 1;
uint64_t segmentEnd = loadStride;
while (k < maxK - 1) {
    if unlikely(k == segmentEnd) {
        MTE2_LoadAL1();
        segmentEnd += loadStride;
    }
    uint64_t end = (segmentEnd < maxK - 1) ? segmentEnd : maxK - 1;
    for (; k < end; ++k) {           // Internal for Internal Zero Branch
        UpdateParams(k);
        Compute();
    }
}

// Epilogue: handling tail blocks
UpdateParams(maxK - 1);
SetFinalParams();
Compute();
```

> Application of the boundary: The three-part code inflation may trigger I-Cache miss when the number of cycles < 5 times is weighed.

---

## P3 Visible writing loop code

### Rationale

`for`'s cycle's back-up**Pilot cycle hardware**does not occur in flush without a branch predictor; and compiler can optimise the visible cycle more easily by revolving, non-cycle unrelated variables, etc.
On the contrary, pass.`while(Iterate())` state machineEvery time a hidden multi-dimensional loop is achievedTileThe switch triggers a lot of progress judgment.3–5Repeatedmispredict);state machineFunction to process dimensionsTailWith the boundary, the code size is usually large (hundreds of lines), easily triggered.I-Cache miss.

### Identification points

- Sees `while(Iterate(self))`, `while(self->Next())`, the hidden state machine drive cycle.
- `Iterate` / `Next` function multiple `if (counter == limit) { counter = 0; ...higher_dim++ }` digitals for internal cascades
- Function length (>200 row), multilayer embedded in-house `if` processing dimension switching

### Inversion

```cpp
while (IterateMFirstMMode(self)) {     // Invisiblestate machine
    LoadAL1(self);
    IterateK(self);
    FreeTensor();
}

// Iterate MFirstMMode inside:
template <class Intf>
inline bool IterateMFirstMMode(Intf* self) {
    if (IterateL0MFirstMMode(self)) return true;
    self->ctx.mAL1Iter++;
    if (self->ctx.mAL1Iter != self->ctx.loopM) { CalcMVar(self); return true; }
    self->ctx.mAL1Iter = 0;
    self->ctx.batchIter++;
    if (self->ctx.batchIter != self->ctx.loopBatch) return true;
    self->ctx.batchIter = 0;
    // It's a multi-level connection.
    return false;
}
```

### - Yes.

```cpp
for (uint64_t nBL1 = 0; nBL1 < self->ctx.loopN; ++nBL1) {
    for (uint64_t batch = 0; batch < self->ctx.loopBatch; ++batch) {
        for (uint64_t mAL1 = 0; mAL1 < self->ctx.loopM; ++mAL1) {
            self->ctx.nBL1Iter   = nBL1;
            self->ctx.batchIter  = batch;
            self->ctx.mAL1Iter   = mAL1;
            CalcMVar(self);

            for (uint64_t nL0 = 0; nL0 < self->ctx.l12l0LoopN; ++nL0) {
                for (uint64_t mL0 = 0; mL0 < self->ctx.l12l0LoopM; ++mL0) {
                    // L0 Layer Calculating
                }
            }
        }
    }
}
```

---

## P4 Use local variables as much as possible

### Rationale

There are three fundamental differences between local and member variables at the level of compiler optimization:

1. **Repositor distribution**: Local variables without an address may never enter memory; member variables must be accessed through `this`+offset at least once per reading and writing memory.
2. **Analysis of aliases**: Local variables without an address, compiler knows that no other pointer can point to it and can be used boldly as a sender; member variables may need to be re-readed for possible pointer aliases.
3. **Cross-functional call**: Local variables without an address can be called across functions and kept in the register; member variables are reloaded from memory when called by an external function.

### Identification points

- There are member variables that are frequently visited only in a single function (or a few interconnective functions) in a class
- The member variable does not actually need to be shared across functions during the Kernel life cycle
- Thermal routing expressions with `this->xxx` are particularly numerous

### Inversion

```cpp
template <typename T>
class FlashAttentionKernel {
protected:
    ConstParam constParam_;                          // Category members
public:
    inline void Init() {
        for (int i = 0; i < constParam_.iters; ++i) {       // Every round Load constParam_
            DoStep(constParam_.scale);                       // this->constParam_.scale
        }
    }
};
```

### - Yes.

```cpp
template <typename T>
class FlashAttentionKernel {
public:
    inline void Init() {
        ConstParam constParam;                       // Local variable, to stay in the repository for the entire journey
        // Initializing constParam...
        for (int i = 0; i < constParam.iters; ++i) {
            DoStep(constParam.scale);                 // None this Indirect, no one else's concerns.
        }
    }
};
```

---

## P5 Variable definition of proximity position

### Rationale

compiler maps the variable to the register. The longer the variable**active range (Live Range)**— from definition to last used code range — the longer the variable takes up the repository. The longer active range squeezes the memory space of other variables, forcing compiler to do Spill. The close location definition may allow the variable to use the temporary register directly, or even to be optimized.

### Identification points

- A large number of local variables are defined centrally at the top of the function
- When a variable is defined, it crosses the irrelevant code in large sections (tens of lines or more) before it is used
- Function segment with a large number of functions calling or recalculating (remister pressure high)

### Inversion

```cpp
void Foo() {
    int a = 1;              // Active range crosses the function
    int b = 2;
    int c = 3;

    DoHeavyWork();          // a/b/c Recalculated temporary repository
    CallSubKernel();
    DoMoreWork();

    Print(a, b, c);         // Really use at the end
}
```

### - Yes.

```cpp
void Foo() {
    DoHeavyWork();
    CallSubKernel();
    DoMoreWork();

    int a = 1;              // It's in close proximity. It's a very short active range.
    int b = 2;
    int c = 3;
    Print(a, b, c);
}
```

---

## P6 Avoid Multi-Level Pointing Reference

### Rationale

There are three kinds of questions:

1. **Visited latency string**: Each level of decitation is essentially a Load directive and there is data dependence among these Loads (the result of the previous Load is the address of the next Load).
2. **Disruption of locality of data**: The addresses of multiple levels of pointers may be scattered at different locations within the stack, and each level of decitation may trigger a D-Cache miss.
3. **compiler Optimizing**: The pointer symmetry problem makes compiler uncertain whether a certain level of content of the pointer will be modified by other writing operations, and every reference must be reloaded from memory.

### Identification points

- Class membership refers to the type of needle through which the target member is reached
- Access chain like `obj_->member->field` or `tiling_->params->Kb`
- Tiling/config type parameters are held permanently by pointer

### Inversion

```cpp
class Block {
    const TCubeTiling* tiling_;     // Pointer type members
public:
    bool IsTail(int kInner) {
        return kInner + stepK_ >= tiling_->Kb;   // 2 Numbers Load: First Load tiling_Again. Load tiling_->Kb
    }
};
```

### - Yes.

```cpp
class Block {
    AscendC::Shape<int64_t, int64_t, int64_t> problemShape_;   // Value type polymers (M, N, K)
public:
    bool IsTail(int kInner) {
        return kInner + stepK_ >= Get<MNK_K>(problemShape_);    // 1 Numbers Load
    }
};
```

---

## P7 Avoiding the use of superstructures

### Rationale

1. **Cache locality difference**: D-Cache Cacheline is 64 bytes. When the structure is too big, access to different members may fall on different cachelines, resulting in an increase in D-Cache miss. Suggested structure ≤ 64B (1 cacheline) does not exceed 128B.
2. **Depositor spill**: compiler is unable to fully place superstructures in the register and must be assigned to the vault memory and each visiting member generates a Load/Store command.
3. **Screen transfer introduces aliases**: to avoid copying costs, large structures often pass through the pin, but this introduces the question of aliases — compiler cannot determine whether the two points point to the same memory and cannot be radically optimized.

### Identification points

- Single struct/class with more than 10 members, estimated dimensions > 64B
- "God's Structure": stacking all fields in the same object at different stages of computation (L1/L0/incident/debug)
- interface signature `void f(BigParam* a, BigParam* b, ...)` (multiple pointers of the same type, aliases risk)

### Inversion

```cpp
struct GodParams {                  // ~200BOver and over. cacheline
    // L1 Phase
    uint64_t l1A, l1B, l1ScaleA, l1ScaleB;
    // L0 Phase
    uint64_t l0A, l0B, l0C;
    // Events
    event_t evV2MTE2, evV2MTE3, evMTE1_M, evM_FIX;
    // Tiling, debugging signs
    TilingParams tiling;
    uint32_t debugFlags;
    // ...
};

void Compute(GodParams* a, GodParams* b) {
    a->l1A = ...;
    b->l1A = ...;                   // if a == b, the previous value is overwritten
    int x = a->l1A;                 // compilerWe have to start over. Load
    int y = a->l0C;                 // and l1A Different. cacheline
}
```

### - Yes.

```cpp
struct L1Stage { uint64_t a, b, scaleA, scaleB; };           // 32B
struct L0Stage { uint64_t a, b, c; };                         // 24B
struct EventIds { event_t v2mte2, v2mte3, mte1_m, m_fix; };   // ≤ 32B

void Compute(L1Stage l1, L0Stage l0) {                        // By value, no other name.
    l1.a = ...;
    l0.c = ...;
}
```

---

## P8 Loads compilation period constants using constexpr/ template parameters

### Rationale

`const` modified member variables, or runtime constants given value in a construction function,**are still runtime values for compiler**- readable through Load and unable to participate in constant folding and revolving (Loop Unrolling).
When changed to `constexpr` or the template is not a type parameter, the value is embedded in the generation code during the compilation period, thus triggering a chain-based optimization of the P1-class constant (compiler is more willing to keep the variable in the register once it has been confirmed as a constant).

### Identification points

- Class consists of `const` member variables, values are determined at the time of construction but remain unchanged throughout the program life cycle
- Tiling parameters generated in the host side as templates/compilation periods are entered in Kernel as runtime input
- Queue like `for (int i = 0; i < kernel.tileM_; ++i)`, the maximum of which should have been known during the compilation period

### Inversion

```cpp
class Kernel {
    const uint32_t tileM_;               // const Members, yes.compilerStill. runtime Load
    const uint32_t tileK_;
public:
    Kernel(uint32_t m, uint32_t k) : tileM_(m), tileK_(k) {}
    void Run() {
        for (uint32_t i = 0; i < tileM_; ++i) {        // Unable to expand
            for (uint32_t j = 0; j < tileK_; ++j) {
                Compute(i, j);
            }
        }
    }
};
```

### - Yes.

```cpp
// Option A: Templates Nontype Parameters - Value solidified in Type
template <uint32_t TILE_M, uint32_t TILE_K>
class Kernel {
public:
    void Run() {
        for (uint32_t i = 0; i < TILE_M; ++i) {        // It can be fully expanded.
            for (uint32_t j = 0; j < TILE_K; ++j) {
                Compute(i, j);
            }
        }
    }
};

// Scheme B:constexpr status - Global compilation period constant
class Kernel {
    static constexpr uint32_t TILE_M = 128;
    static constexpr uint32_t TILE_K = 64;
public:
    void Run() {
        for (uint32_t i = 0; i < TILE_M; ++i) {
            for (uint32_t j = 0; j < TILE_K; ++j) {
                Compute(i, j);
            }
        }
    }
};
```

---

## P9 Hot Loop does not construct objects and does not remove addresses

### Rationale

1. **The construction/deconstruction is expanded to a large number of Store**: In the heat cycle, the object is constructed, and each of the rotations is initiated to the individual member, Store; objects with the semantic semantics are also inserted into the end of the iterative period.
2. **The taking of the address forces compiler Spill**: Taking the address for the local variable (`&x`) means that the address may be passed to an external code, and compiler must conservatively assume that the variable may be modified after the call, and therefore the variable must be placed on the stack (Spill) and not kept in the repository.
3. **Taking an address also hinders the analysis of aliases**: the non-alignment cannot be demonstrated between the variable from which the address is taken and the other pointer, contaminating the optimization of the surrounding code.

### Identification points

- In situ construction of unusual types (e.g., `LocalTensor`, temporary polymers) in the thermal cycle
- `Api(&x)` callable in cycle and `x` could have been passed by value
- Frequently retrieve addresses for intermediate variables in the cycle to pass to auxiliary functions

### Inversion

```cpp
// Example #1: LocalTensor inside cycle, triggers the initialization of multiple members per round
for (int i = 0; i < N; ++i) {
    LocalTensor<half> tmp = buf.Get<half>();      // Reconstructing per wheel
    Compute(tmp, i);
}

// Reverse #2: Take address to local variables in the cycle, force Spill to the inn
for (int i = 0; i < N; ++i) {
    uint32_t idx = i * 2;
    LegacyApi(&idx);                              // &idx Trigger Spill,idx Can not get folder: %s: %s
}
```

### - Yes.

```cpp
// Regular #1: Construct externals, recycle only
LocalTensor<half> tmp = buf.Get<half>();
for (int i = 0; i < N; ++i) {
    Compute(tmp, i);
}

// Example #2: Transferable by value; raise the variable to a circular non-variant when an address must be passed
for (int i = 0; i < N; ++i) {
    uint32_t idx = i * 2;
    NewApi(idx);                                  // It's worth it.idx It's all over the register.
}
```
