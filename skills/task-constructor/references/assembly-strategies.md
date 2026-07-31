# Task assembly policy

## Reliance on tracking mechanisms

The `trace_dependencies` tool analyzes all dependences on automatic detection functions through AST:

### 1. Import Unnamed

Build `{alias: source module}'map from the top `import` statement and top grant value of the file:

```python
# import torch._prims_common as utils → {"utils": "torch._prims_common"}
# from torch._decomp import register_decomposition → {"register_decomposition": "torch._decomp"}
```

### 2. Reliance on classification

- **Reliance with file**: function/class defined in the same document, direct extraction
- **External call**: other module functions cited through Import alias
  - Public API (e. g. `torch.tensor()`) → Reserve Report
  - Internal API (with `_` prefix module) → needs inline

### 3. External Call Processing

For external calls requiring interconnectivity:
1. Position the original file via the source module path
2. Read complete with `read_function`
3. Check for function signatures to ensure consistency of parameters
4. Link function to output file

## Combination Policy

### Exclusion (large document)

```
General function of file - Unrequired Functions = Output
```

Applies to: When the target function relies on a large number of same-file functions

### Selective (accurate extraction)

```
Enter Functions + List of relying functions = Output
```

Applicable: when dependency relationships are clear and the number of functions is limited

### Full Dock (Small File)

```
Entire Document = Output
```

Applies to: small files, close inter-functional coupling

## Validation process

```
1. Syntax: → AST Parsable
2. Structure check → Model Category + get_inputs + get_init_inputs Existence
3. runtimeInspection → Empirical → forward → None NaN/Inf
4. References → Multigroup Input vs. Original Function Output
```
