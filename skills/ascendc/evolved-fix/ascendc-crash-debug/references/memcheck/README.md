# Guidelines for the use of mssanitizer memory testing

The memory error (cross-border reading and writing, non-matching, multi-nuclei, etc.) used to detect memory errors in AscendC operator kernel is based on `mssanitizer --tool=memcheck`.

**This is `ascendc-crash-debug` Skill 's subfunctional module, which is triggered by `/ascendc-crash-debug`.**

## Preconditions

- NPU device available (`npu-smi info` normal)
- CANN environment installed and available (including `mssanitizer` tool)
- operator code repository containing `build.sh` compilation scripts
- There is a runable ST test case (pytest)

## Fast start.

### Step 1: Prepare configuration file

Copy the template to your work directory and adapt it to the actual situation:

```bash
cp scripts/memcheck_input.json.template ./memcheck_input.json
```

Edit `memcheck_input.json` to fill in your operator message:

```json
{
  "operator": {
    "name": "your_operator_name"
  },
  "paths": {
    "code_base_dir": "/path/to/your/training/ascendc"
  },
  "testing": {
    "test_script_dir": "/path/to/your/.../tests/st",
    "test_script_exe": "pytest test_npu_your_operator.py"
  },
  "environment": {
    "device_type": "ascend910b",
    "cann_env": "/path/to/cann/latest"
  }
}
```

The fields state:

| Fields | shall fill | Annotations |
|------|------|------|
| `operator.name` | Yes. | Name of operator |
| `paths.code_base_dir` | Yes. | Code silo directory (includes `build.sh`) |
| `testing.test_script_dir` | Yes. | ST Test script directory (absolute path) |
| `testing.test_script_exe` | Yes. | Test execution commands such as `pytest test_npu_xxx.py` |
| `environment.device_type` | Yes. | NPU device type, e.g. `ascend910b` |
| `environment.cann_env` | Yes. | CANN Environment Path |
| `compilation.sanitizer_options` | Yes | Compiler Options, Default `"-sanitizer;-g"` |
| `memcheck.timeout` | Yes | Timeout seconds, default `600` |
| `options.rebuild` | Yes | Whether to recompile, default `true` |

### Step 2: Copying scripts and implementing them

```bash
cp scripts/run_memcheck_pre.sh .
chmod +x run_memcheck_pre.sh
./run_memcheck_pre.sh
```

Script will be automatically completed: compile (with Sanitizer) → install operator package → run mssanitizer memcheck.

### Step 3: View results

Output saved under `<code_base_dir>/memcheck_output/`:

```
memcheck_output/
├── status.txt                          # Summary of status of implementation
├── build/build.log                     # Compile Log
├── install/install.log                 # Install Log
└── memcheck/
    ├── ascendc_memcheck_report_raw.txt # memcheck Original report (focus on this)
    └── mindstudio_sanitizer_log/       # mssanitizer Detailed Log
```

### Step 4: Let Claude Analyse

Enter `/ascendc-crash-debug` in Claude Code and specify the need for memory testing, and Claude will automatically:
1. Read memcheck report, extract ERRO and WARNING
2. Based on call stack, locate source code
3. Analysis of root causes and recommendations for rehabilitation
4. Generate `memcheck_detailed_report.md` detailed report

## Common Options

```bash
# Skip compile, run memcheck directly with the existing product
./run_memcheck_pre.sh --skip-build

# Specify other configuration file
./run_memcheck_pre.sh --config my_config.json

# Detailed Output
./run_memcheck_pre.sh --verbose
```

## Typical use of scene

- operator running crashes, occasional crashes cannot recur.
- Suspected of cross-border reading and writing or memory trampling leading to collapse
- Aic error or coredump stack is not clear
- The multi-nuclear scenario is going to collapse.
- Memory security check before code submission


## Contents structure

```
ascendc-crash-debug/
├── scripts/
│   ├── memcheck_input.json.template     # configuration fileTemplate (copy to work directory)
│   └── run_memcheck_pre.sh              # Automatically detect scripts
│   └── parse_plog.py                    # plog Log Resolution Script
└── references/
    └── memcheck/
        ├── automated_workflow.md        # Details of automated workflows
        ├── README.md                    # This document - User Use Guide
        └── mssanitizer_guide.md         # msSanitizer Tool Original Document
```

## common issue

**Q: Can't find `mssanitizer`**in the script?
Confirm CANN environment correctly loaded: `source <cann_env>/bin/setenv.bash`

**Q: Failed to compile**
Check `memcheck_output/build/build.log` to confirm that the `code_base_dir` path is correct and contains `build.sh`.

**: memcheck timeout**
Add `memcheck.timeout` (in seconds) to configuration file, and large operator suggests 1,200+.
