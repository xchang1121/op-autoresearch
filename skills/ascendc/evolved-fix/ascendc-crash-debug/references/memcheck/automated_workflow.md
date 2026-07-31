# AscendC Memcheck Automation Workflow Guide

## Overview

Step 1-3 of `run_memcheck_pre.sh` script automating AscendC memory detection workflow:

1. **Compiled**: operator compiled using the Sanitizer option
2. **Installation**: installation of operator packages to `code_base_dir/custom_output_dir` directory
3. **Run MemCheck**: perform mssanitizer memcheck testing

## Fast start.

### 1. Prepare configuration file.

Copy template from skill directory to current working directory:

```bash
cp scripts/memcheck_input.json.template memcheck_input.json
```

Edit `memcheck_input.json` to complete the required parameters:

```json
{
  "operator": {
    "name": "your_operator_name"
  },
  "paths": {
    "code_base_dir": "/path/to/your/training/ascendc"
  },
  "testing": {
    "test_script_dir": "/path/to/your/code/training/ascendc/src/.../tests/st",
    "test_script_exe": "pytest test_npu_xxx.py"
  },
  "environment": {
    "device_type": "ascend910b",
    "cann_env": "/path/to/cann"
  },
  "options": {
    "rebuild": true
  }
}
```

### 2. Copy running scripts

Copy script from the skill directory to the current path and execute:

```bash
cp scripts/run_memcheck_pre.sh .
./run_memcheck_pre.sh
```

### 3. View Results

Output results saved in `code_base_dir/memcheck_output/` directory:

```
code_base_dir/memcheck_output/
├── status.txt              # Summary of status of implementation
├── build/
│   ├── build.log           # Compile Log
│   ├── build_errors.log    # Compile Error Log
│   └── package_path.txt    # Path to Compile Product
├── install/
│   ├── install.log         # Install Log
│   └── install_errors.log  # Install Error Log
├── memcheck/
│   ├── memcheck.log        # Memcheck Log
│   ├── ascendc_memcheck_report_raw.txt  # Original report
│   └── mindstudio_sanitizer_log/        # mssanitizer Log
└── timestamp.txt           # Time stamp for execution
```

## Detailed description

### configuration file Parameters

| Field Path | Annotations | Whether necessary | Default value | Example: |
|---------|------|---------|--------|------|
| `operator.name` | Name of operator | Yes. | - | `"sparse_flash_attention_enhance"` |
| `paths.code_base_dir` | Code Library Directory (includes build.sh) | Yes. | - | `"/home/user/code/training/ascendc"` |
| `testing.test_script_dir` | ST Test Script Absolute Path | Yes | - | `"/home/user/code/training/ascendc/.../tests/st"` |
| `testing.test_script_exe` | ST Test Script Execution | Yes | - | `"pytest test_npu_xxx.py"` |
| `environment.device_type` | NPU device type | Yes. | - | `"ascend910b"` |
| `environment.cann_env` | CANN Environment Path | Yes. | - | `"/home/user/pkg/cann/latest"` |
| `compilation.sanitizer_options` | Compile Options | Yes | `"-sanitizer;-g"` | `"-sanitizer;-g"` |
| `memcheck.log_level` | Log Level | Yes | `"3"` | `"3"` |
| `memcheck.slog_print_to_stdout` | Standard Output Switches | Yes | `"true"` | `"true"` |
| `memcheck.timeout` | Timeout (sec) | Yes | `600` | `600` |
| `options.rebuild` | Renumbered or not | Yes | `true` | `true` |
| `installation.load_environment` | Whether to load operator environment | Yes | `true` | `true` |

### command line Parameters

```bash
./run_memcheck_pre.sh [options]
```

| Parameters | Annotations |
|------|------|
| `-h, --help` | Show help information |
| `-c, --config FILE` | configuration file path (default: `./memcheck_input.json`) |
| `--skip-build` | Skip Compiler Steps |
| `--keep-build` | Keep Build Directory |
| `--verbose` | Show Detailed Output |

### Use Example

#### Example 1: Use default configuration

```bash
./run_memcheck_pre.sh
```

#### Example 2: Specify configuration file

```bash
./run_memcheck_pre.sh --config /path/to/my_config.json
```

#### Example 3: Skip Compiled (Accompiled)

```bash
./run_memcheck_pre.sh --skip-build
```

#### Example 4: Detailed output mode

```bash
./run_memcheck_pre.sh --verbose
```

#### Example 5: Grouping Options

```bash
./run_memcheck_pre.sh \
  --config my_config.json \
  --verbose
```

## Workflow

### Full workflow (3 steps)

1. **Compiled operator**
   - Load CANN Environment: `source <cann_env>/bin/setenv.bash`
   - Enter code directory: `cd <code_base_dir>`
   - Execute Compiled: `bash build.sh -n <op_name> -c <device> -p <cann_env> --ops-compile-options "<sanitizer_opts>"`
   - Check compiler (.run files)
   - Save compiler path to `memcheck_output/build/package_path.txt`

2. **Installed operator**
   - Find compiler: Find.run files from `code_base_dir/output` directory
   - Create/cleanup installation directory: `code_base_dir/custom_output_dir/`
   - Execute installation: `./output/<package>.run --install-path=<code_base_dir/custom_output_dir>`
   - Load operator Environment (optional): `source <code_base_dir>/custom_output_dir/vendors/omni_training_custom_ops/bin/set_env.bash`

3. **Run MemCheck**
   - Set log environment variable:
     - `ASCEND_GLOBAL_LOG_LEVEL=<log_level>`
     - `ASCEND_SLOG_PRINT_TO_STDOUT=<slog_print>`
   - Enter ST directory (extract from `test_script_dir` absolute path)
   - Implementation: `mssanitizer --tool=memcheck <test_script_exe>`
   - Collect output and log file
   - Generate status report:
     - Statistics ERRO and WARNING Number
     - Show pre-5 errors/warnings
     - Check the test results.

### Skip compile workflow

When using `--skip-build` or `rebuild=false` in configuration file:

1. **Skip compile**: direct to existing compilers
2. **Install operator**: Search and install existing operator packages from `code_base_dir/output` to `code_base_dir/custom_output_dir`
3. **Run MemCheck**: perform memory testing

## Output file description

### status.txt

Summary of the status of implementation, including:
- Implementation time
- configuration file Path
- Implementation status of steps (success/failure/jump)
- List of Output File Paths

### build/

Compile related logs:
- `build.log`: Full compilation output
- `build_errors.log`: Compile error message (if compilation fails)
- `package_path.txt`: Absolute Path to Compiled Product

### install/

Install related logs:
- `install.log`: Full installation output
- `install_errors.log`: Install error message (if installation fails)

### memcheck/

Memcheck Related Document:
- `memcheck.log`: Full memcheck output (copy from raw_report)
- `ascendc_memcheck_report_raw.txt`: Original Memcheck Report
- Detailed logs generated by the `mindstudio_sanitizer_log/`:mssanitizer tool

## Error Handling

The script will exit and return the non-zero-state code in the following cases:

1. configuration file does not exist or format error
2. Cann Environment Script does not exist
3. No code catalogue or compiled script exists
4. Compiled failed
5. Installation Failed
6. Test directory does not exist
7. Memcheck failed (as appropriate)

Each failure saves the complete error message in the log.

## common issue

### Q: How do you change the compilation options?

A: Edit `compilation.sanitizer_options` fields from configuration file.

```json
{
  "compilation": {
    "sanitizer_options": "-sanitizer;-g"  // Use full debug information
  }
}
```

### Q: How can only run MemCheck, not recompilation?

A: Set `rebuild=false` using `--skip-build` parameters or in configuration file.

```bash
./run_memcheck_pre.sh --skip-build
```

or

```json
{
  "options": {
    "rebuild": false
  }
}
```

### Q: Where are the products kept?

A: All outputs are stored in the `code_base_dir/memcheck_output/` directory. The operator package is installed in the `code_base_dir/custom_output_dir/` directory.

### Q: How do you view Memcheck's results?

A: View `code_base_dir/memcheck_output/memcheck/ascendc_memcheck_report_raw.txt` files.

### Q: What depends on the script?

A: Scripts require:
- Bash 3.2+
- Python 3.6+ (for interpretation of JSON)
- CANN Environment
- mssanitizer tool (usually provided by CANN)

### Q: Why does testing scripts use absolute paths?

A: The use of absolute paths avoids a relative path calculation error, especially when performing scripts in different directories. configuration file uses `test_script_dir` to specify an absolute path to the test directory, and `test_script_exe` to specify an order (usually a pytest command) to be executed under this directory.

### Q: What is the use of `memcheck.timeout` parameters?

A: This parameter is defined in configuration file and is used to set the time limit (in seconds) for Memcheck execution, by default 600 seconds (10 minutes). The process may be terminated if the execution time exceeds the set value.

### Q: How does installation fail?

A: Script cleans up old installation directories and re-installs. If it fails:
1. Check `code_base_dir/memcheck_output/install/install.log` for detailed errors
2. Check if there's any remaining files in the catalogue.
3. Manually clear `code_base_dir/custom_output_dir/` directory
4. Rerun Script

### Q: How do you disable the automatic loading of operator environment?

A: Set `installation.load_environment` in configuration file as `false`:

```json
{
  "installation": {
    "load_environment": false
  }
}
```

## Distinction from manual execution

| Item | Manually execute | Automation Script |
|------|---------|-----------|
| Configure | Set command line parameters manually each time | Management through JSON configuration file |
| Compile | Manually execute build.sh | Automatically execute and check results |
| Install | Manually search for installation packages and specify paths | Automatically search and install to a fixed path |
| Environment | Manually load several times | Automatically load necessary environments according to configuration |
| Log | Dispersive Save | Centralize to `code_base_dir/memcheck_output` |
| Error Handling | Need manual check. | Automatically detect and report |
| Test script path | Manually calculate relative/absolute paths | Use absolute path to automatically enter the test directory |
| Install Path Management | Need to specify absolute paths manually | Fixed on `code_base_dir/custom_output_dir/` |
| Summary of results | Require manual statistics | Automatically generate ERRO/WARNING |

## Next steps

After the script is finished, continue step 4-7:

1. **Analysis error output**: View ERRO and WARDING in memcheck report
2. **Location source code**: based on call stack location problem code
3. **Root analysis**: analytical error
4. **Generating report**: Create detailed memory detection report

For further information, please refer to the main document [`SKILL.md`] (./SKILL.md).

## Skills and recommendations

### 1. Manage configuration file using version control

Create a different configuration file for a different operator or test scenario:

```bash
cp scripts/memcheck_input.json.template memcheck_input_op1.json
cp scripts/memcheck_input.json.template memcheck_input_op2.json
```

### 2. Custom Timeout

For large operator or complex test case, adjust `memcheck.timeout` parameters:

```json
{
  "memcheck": {
    "timeout": 1200  // 20 min
  }
}
```

### 3. Debug Mode

View detailed execution using `--verbose`:

```bash
./run_memcheck_pre.sh --verbose
```

### 4. Batch Test

Multioperator test (create test script) in conjunction with script:

```bash
#!/bin/bash
cp scripts/run_memcheck_pre.sh .

for config in memcheck_input_*.json; do
    ./run_memcheck_pre.sh -c "$config"
done
```

### 5. Environmental management

- Set `load_environment: false` if you do not want scripts to automatically load operator environment
- The compilation environment (CANN) and the operator environment are managed separately, without interference

### 6. Clear old version

If a version of the operator package is no longer needed:

```bash
# Manually delete specified custom_output_dir directory
rm -rf /path/to/code/base/custom_output_dir

# or modify the code_base_dir configuration to a different code directory
```

## Contents

### skill Directory Structure

```
ascendc-crash-debug/
├── scripts/
│   ├── memcheck_input.json.template    # configuration fileTemplates
│   ├── run_memcheck_pre.sh             # Automation Script
│   └── parse_plog.py                   # plog Log Resolution Script
└── references/
    └── memcheck/
        ├── automated_workflow.md       # This document - Automation workflow guide
        ├── README.md                   # User Use Guide
        └── mssanitizer_guide.md        # msSanitizer Tool Original Document
```

## Support and feedback

If you have a problem, please check:

1. Whether configuration file format is correct (JSON syntax)
2. All required parameters have been completed
3. CanN Environmental Path Correct
4. Whether NPU device is available
5. error message in log file
6. Is there another process to use the operator package installation path
7. Test whether Script Directory exists
8. Tests if script names match configuration

---

**Document version**: 2.4
**Final update**: 2026-05-16
**Applicable version**: ascendc-crash-debug skill (memcheck submodule)