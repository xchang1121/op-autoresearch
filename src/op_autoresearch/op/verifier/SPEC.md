# op/verifier/ — verifier

## Duties

Validates the correctness and performance of the resulting kernel code. Support multiplatforms through a combination of three types of adapters (backend, dsl, framework).

## Contents structure

```
verifier/
├── kernel_verifier.py         # KernelVerifier — Validation portal, group three adapters.
├── data_cache.py              # Verifier Data Cache — reference data / baseline Enduring Cache
├── sol_verifier.py            # SOL-ExecBench Special Generator for Format Validation
├── profiler.py                # NPU/CUDA Performance Collection
├── profiler_utils.py          # profile Script execution,msprof/nsys Parsing
├── roofline_utils.py          # Through installed solar Python Package Count roofline
├── l2_cache_clear.py          # Ascend L2 cache Cleaning
└── adapters/                  # Three types of adapters
    ├── factory.py             # get_backend_adapter / get_dsl_adapter / get_framework_adapter
    ├── backend/               # BackendAdapter and realization of the
    │   ├── base.py            #   BackendAdapter(ABC)
    │   ├── cuda.py            #   BackendAdapterCuda
    │   ├── ascend.py          #   BackendAdapterAscend
    │   └── cpu.py             #   BackendAdapterCpu
    ├── dsl/                   # DSLAdapter and realization of the
    │   ├── base.py            #   DSLAdapter(ABC) — Here's the extended hook.
    │   ├── triton_cuda.py     #   DSLAdapterTritonCuda
    │   ├── triton_ascend.py   #   DSLAdapterTritonAscend
    │   ├── cpp.py             #   DSLAdapterCpp
    │   ├── cuda_c.py          #   DSLAdapterCudaC
    │   ├── ascendc.py         #   DSLAdapterAscendC
    │   ├── ascendc_catlass.py #   DSLAdapterAscendC_Catlass
    │   ├── tilelang_cuda.py   #   DSLAdapterTilelangCuda
    │   ├── tilelang_npuir.py  #   DSLAdapterTilelangNpuir
    │   ├── torch.py           #   DSLAdapterTorch
    │   ├── pypto.py           #   DSLAdapterPypto
    │   └── swft.py            #   DSLAdapterSwft
    └── framework/             # FrameworkAdapter and realization of the
        ├── base.py            #   FrameworkAdapter(ABC)
        ├── torch.py           #   FrameworkAdapterTorch
        ├── mindspore.py       #   FrameworkAdapterMindSpore
        └── numpy.py           #   FrameworkAdapterNumpy
```

## Development engagements

### Add standard process for authentication adapters

1. Determine adapter type (backend / dsl / framework)
2. Create file in corresponding subdirectories, inherit `BackendAdapter(ABC)` / `DSLAdapter(ABC)` / `FrameworkAdapter(ABC)`
3. The abstract method to achieve ABC + the extension hook below as needed override
4. Register a name map in `adapters/factory.py`
5. Add DSL name to `op/utils/config_utils.py` 's `VALID_CONFIGS / valid_dsls` table

### DSLadapter Extension hook (base.py)

Each additional DSL should**only**change an adapter file, not at `kernel_verifier.py` / `local_worker.py`/
`workspace_autoresearch/scripts/scaffold.py` equal caller plus `if self.dsl == "xxx":` branch.
All per-DSL behaviour is expressed through the hook below; the caller calls the corresponding hook directly after he/she has taken the adapter instance.

**Abstract method (must be achieved)**:

| Methodology | Purpose |
|---|---|
| `get_import_statements(framework)` | Impl file top Import line |
| `get_impl_import(op_name, impl_func_name)` | Authentication of statements in the script impl |
| `call_impl(...)` | Authentication of code clips in scripts calling impl |
| `benchmark_impl(...)` | Benchmark Snippets in the profile template |

**Class Properties (Pure condition, override with `= value`, not written as `def f(self): return value`)**

| Properties | Default | Meaning |
|---|---|---|
| `needs_binary_io` | `False` | Whether binary I/O file transfer is required (swft = True) |
| `static_check_via_python_ast` | `True` | Whether LLM submitted source code is valid Python (cpp / cuda_c / swft = False, `CodeChecker` Skips the AST Layer) |
| `profile_via_python_script` | `False` | LocalWorker dispatch: True → Run Python script reading JSON; False → go msprof / nsys (triton_*/ pypto / catlass = True) |
| `benchmark_requires_l2_clear` | `True` | Base benchmark template clean-up L2(catlass = False) |
| `impl_func_name_template` | `"{op_name}_{dsl}_{framework}"` | Default impl function name template (ModelNew class DSL = `"ModelNew"`, AscendC = `"{op_name}_kernel"`) |
| `kernel_arg_is_directory` | `False` | kernel handoff for DSL is not a directory (ascendc_catlass / ascendc = True: the directory contains sibling Python wrapper+ subtrees) |
| `kernel_project_dir_name` | `None` | Project subdirectories when `kernel_arg_is_directory=True` (e. g. `"catlass_op"` / `"ascendc_op"`) |
| `kernel_project_files` | `[]` | The list of documents that make up the DSL Kernel project (relative to the Python wrapper peer directory), by which the driver layer (e. g. WA scaffold) determines what to copy / set to edit |

**Optional hook (default no-op, override, as required)**:

| The hook. | Caller | Purpose |
|---|---|---|
| `materialize_impl(impl_code, verify_dir, op_name, framework, dsl_name, task_info, config)` | `KernelVerifier.gen_verify_project` | Set LLM-generated code to verify_dir. Default to `<op>_<dsl>_impl.py` and prepend reports; Catlass to kernel.py + to copy Catlass_op tree |
| `expected_artifacts(verify_dir, op_name, framework, bench_type, dsl_filename_hint)` | `KernelVerifier._verify_impl_artifacts_ready` | Columns verify_dir must have a product path; default is a framework file + impl file |
| `prepare_config(config, task_info)` | Before `KernelVerifier.run / run_profile`, call `get_special_setup_code` | Config side effects (resolve CATLASS_ROOT et al.) |
| `get_special_setup_code(framework)` | impl file top | Injecting one-time setup clips (tilelang clear_cache, catlass cmake build etc.); Arch / catlass_root etc. do not sign, from `prepare_config` cache to self |
| `get_runtime_env_override_code(**kwargs)` | impl file top | Injecting runtime env Overwrite (pypto run mode/ debug position); default empty string |
| `post_iteration_cleanup(verify_dir)` | WA `eval_bridge._eval_async` finally | Clean up short-lived products (catlass delete `catlass_op/build`) at the end of each round |
| `read_kernel_source(kernel_arg, op_name=None)` | WA scaffold | Parsing the kernel handoff path to `(source_code, project_dir_or_None)`; read by default file, catlass read by directory + peer `kernel.py` / `<op>_kernel.py` |
| `materialize_project_tree(dst_dir, project_src, project_dir_name=None)` | WA scaffold | Cuff project sub-trees to dst_dir and do DSL-specific patches (catlass Tortures Directory + Patch CMakeLists) |

### Caller Agreement

| Caller | The ability to get through an adapter |
|---|---|
| `KernelVerifier.gen_verify_project` | `materialize_impl` + `get_import_statements` + `get_impl_import` + `get_special_setup_code` + `get_runtime_env_override_code` + `impl_func_name_template` + `needs_binary_io` |
| `KernelVerifier.run / run_profile` | `prepare_config` + `expected_artifacts` + `benchmark_requires_l2_clear` |
| `LocalWorker.profile`(`core/worker/local_worker.py`) | `profile_via_python_script` → chooses to follow the Python-script path or msprof / nsys |
| `CodeChecker.check`(`op/utils/code_checker.py`) | `static_check_via_python_ast` → Run the AST Layer Check |
| WA `scripts/scaffold.py` | `read_kernel_source` + `materialize_project_tree` + `kernel_project_files` (with which WA is derived task.yaml `editable_files`; "editable" is the WA policy, not on top of an adapter) |
| WA `scripts/batch/manifest.py` | `kernel_arg_is_directory` + `kernel_project_dir_name` (deciding the watch single file vs multifile resolution path) |
| WA `scripts/utils/eval_bridge.py` | `post_iteration_cleanup` |

### Kernel Verifier Core Logic

`KernelVerifier` obtained three examples of adapters by `get_*_adapter` plant method (DSL adapter in
Cache in `__init__` to `self.dsl_adapter` because `prepare_config` will make an example of an adapter
Upstash state, and then combine to generate validation scripts (Jinja2 templates), CMake configurations, etc., and eventually perform validation and profling.

Validation packages and returns data that cross the process/HTTP boundary: LocalWorker has to reject when unpacking
Absolute path, `..` across borders, links and device nodes; `sync_artifacts_to_directory` must be written before
Confirms that realpath is still in the current verify_dir and cannot trust the relative path of remote return.

### Verifier Data Cache

- Validation links only for `KernelBench` style
- Reference Data Cache: Reuse `generate_reference_data(save_inputs=True)` output `.pt`
- Baseline Cache: reuse `base_profile_result.json` / `avg_time_us`
- default close; enduring under `~/.op_autoresearch/verifier_data_cache/` after opening
- Enable scripts to directly reuse inputs / outputs, without duplicates
- Reference data Cache only overstatic Shape; dynamic shape automatically skips and avoids misinputation of single group input
- Reference data will verify `.pt` payload, remove old caches and regenerate when damaged or reusable fields are missing
- Directly inject `override_base_time_us` and skip base profile scripts when hit
- Cachekey contains `task_id` by default; use this stable identity when `data_cache.cache_key_id` is configured to support multiple verifier task reuses in the same workflow
- Baseline size key must also include DSL to avoid contamination of different time paths under the same framework/backend/arch

### Roofline Integration

- Roofline directly calls installed `solar` Python API (`graph/einsum/analysis/perf`) via `roofline_utils.py`
- OP_AUTORESEARCH**does not rely on**local `SOLAR` working tree path; runtime only requires `import solar` success
- Access logic (e. g. Solbench wrapper / Ascend arc config) not previously in the Solar official package is maintained by OP_AutoRESEARCH
- **Do not**change the `SOLAR` repository source code or beat it
- roofline failure can only be downgraded to "no roofline data",**cannot**affect original process / profile main process

## Autotune Double Mode Authentication

`KernelVerifier` supports two authentication modes for the Triton autotune code:

- **Direct validation mode**(default): Autotune code, like normal code, runs complete code validation at once.
- **Config Validation Mode**: Individualally validates config, running the whole code back once it's all passed. If config passes the complete code, but the complete code fails, the log prompts the addition of `restore_value`.

The config mode of authentication is open in two ways (either takes effect):
1. Environmental variable: `OP_AUTORESEARCH_VERIFY_PER_CONFIG=1`
2. Triton_config YaML Configuration: `verify_per_config: true` (default `false`)

Both modes require that `@triton.autotune` must contain `restore_value` parameters (guaranteed by `CodeChecker` static check).

## Nothing.

- **Not**Agent/Workflow logic in adapter
- **Do not**Hard Encoding backend/DSL Specific Behaviour to `kernel_verifier.py` / `local_worker.py` /
  `workspace_autoresearch/scripts/scaffold.py` - Expand the hook through the adaptor. Add `if
  Self.dsl = xx: `The branch review will be called back; the correct approach is to expand the hook in `DSLadapter ' Riga
  / Class Properties, caller only
- **Do not**write pure constants into `def f(self): return value` methods.
  Properties (`flag: bool = False`), method only left to a genuine side effect / calculated hook (e. g.
  `prepare_config`, `materialize_impl`, `post_iteration_cleanup`)
- **Do not**insert a DSL private parameter in the base `DSLAdapter` signature (e. g. `arch`)
  `catlass_root`. This is an "interface leak" -- should get an adapter from `prepare_config`
  config reads cache to `self.*`, `get_special_setup_code(framework)` to keep signature
  Cross DSL Consistency
