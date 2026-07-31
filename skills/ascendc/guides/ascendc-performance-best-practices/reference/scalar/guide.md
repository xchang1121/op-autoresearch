# Scalar Performance Optimization Policy Index

The Scalar module is responsible for scalar calculations, address calculations, command parameter construction and command distribution. When it becomes a bottleneck and cannot be sent to the Cube / Victor / MTE's Issue queue in a timely manner, other pipelines appear to be bubble and the whole operator is slow. This phenomenon is called**ScalarBound**.

The core conclusion of Scalar ' s optimization:**The primary reason is not the number of Scalar ' s computational instructions, but the number of Load/Store commands**(typically over 30%)**The root cause is the compiler repository, Spil. The common objective of all the principles under this heading is therefore:**To help compiler better complete the register allocation, aliases analysis and constant dissemination, thereby reducing the number of Load/Store**.

## 9 Encoding Principles Quick Check

| Numbering | Principles | The center of the sentence. |
|------|------|-----------|
| [P1](coding_principles.md#p1-Structural Practising Numerics) | We'll use the array carefully in the structure. | The dynamic subscript structure array will allow compiler to give up the constant transmission of the entire structure. |
| [P2](coding_principles.md#p2-recycling main tail block separated) | Loop main tail block separated | Prologue / Hot Loop (zero branch) / Epilogue three bands to avoid 3 branch judgements per round of thermal path |
| [P3](coding_principles.md#p3-Focus writing loop code) | Visible writing of loop code | Use embedded `for` instead of hidden state machine; `for` back-jumping special hardware, not branch prediction |
| [P4](coding_principles.md#p4 -- use of local variables to the extent possible) | Use local variables as much as possible | Member variables suppressed by aliases; cross-functional calls must be reset from memory |
| [P5](coding_principles.md#p5-View definition close to usage position) | Variable definition of proximity position | Shorten the active range = Shorten the register occupancy = Reduce the Spill probability |
| [P6](coding_principles.md#p6 - Avoid multi-level pointer decomposition references) | Avoid multi-level pointer decomposition references | Each level of decitation is dependent Load, replaced by a value type polymer |
| [P7](coding_principles.md#p7 - Avoiding the use of megastructures) | Avoiding the use of superstructures | > 64B cross carcheline; overstore capacity required Spill; pointer transfer includes aliases |
| [P8](coding_principles.md#p8-use-constexpr-format constant for template parameters) | Carrying translation period constants with `constexpr` / template parameters | `const`Members are right.compilerStill.runtime Load;entry constant folding to chain-ledP1Class optimization |
| [P9](coding_principles.md#p9-hot-loop-Do not construct objects without an address) | Hot Loop does not construct objects, does not remove addresses | Build/deconstruct large-scale initialization of Store; take an address so that compiler hypothetical variables can be modified externally and force Spil |

## Synergy with other operators

Scalar Optimization with operator Optimization**: operator Optimization determines "what to calculate and move" and Scalar Optimization determines "can these instructions be distributed in a timely manner." Both are to be done, but in the first place the operator Optimisation (tiling/ MTE strategy) is to be followed.

Each of the operator groups has a scalar-specific optimization strategy based on the 9 principles:

| operator | scene | Core thinking | Guide |
|--------|------|---------|------|
| FA | — | 📋 planning | — |
