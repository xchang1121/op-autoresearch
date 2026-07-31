#!/bin/bash

# ==============================================================================
# AscendC Memcheck Automation Script - Implement 3 Steps (Compilation, Installation, Memcheck)
# ==============================================================================
#
# Function:
#   1. Compile operator (with Sanitizer options)
#   2. Installation of operator packages
#   3. Run memcheck testing
#
# Usage:
#   ./scripts/run_memcheck_pre.sh [options]
#
# Parameters:
#   -h, --help display help information
#   -c, --config FILLE configuration file Path (default:./memcheck_input.json)
#   --skip-build Skip Compiler Step
#   --keep-build Keep Build Directory
#   Show detailed output for --verbose
#
# ==============================================================================

set -e

# ============================================================================
# Colour output definition
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Log Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# ============================================================================
# Default Configuration
# ============================================================================

CONFIG_FILE="./memcheck_input.json"
SKIP_BUILD=false
KEEP_BUILD=false
VERBOSE=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# Show help information
# ============================================================================

show_help() {
    cat << EOF
AscendC Memcheck Automation Script - Enforcement 1-3 Step (compilation, installation, installation)memcheck)

Usage:
  $(basename "$0") [options]

Parameters:
  -h, --help          Show help information
  -c, --config FILE   configuration filePath (default:./memcheck_input.json)
  -o, --output DIR    Output directory (default:./memcheck_output)
  --skip-build        Skip Compiler Steps
  --keep-build        Keep Build Directory
  --verbose           Show Detailed Output

Example:
  # Use default configuration
  $(basename "$0")

  # Specify configuration file
  $(basename "$0") --config /path/to/memcheck_input.json

  # Skip Compiler Steps
  $(basename "$0") --skip-build

  # Show Detailed Output
  $(basename "$0") --verbose

configuration fileTemplate:
  memcheck_input.json.template

Output directory structure:
  $OUTPUT_DIR/
  ├── status.txt              # Summary of status of implementation
  ├── build/                  # Build Related
  │   ├── build.log
  │   └── build_errors.log
  ├── install/                # Install Related
  │   ├── install.log
  │   └── install_errors.log
  ├── memcheck/               # Memcheck Relevant
  │   ├── memcheck.log
  │   ├── ascendc_memcheck_report.txt
  │   └── mindstudio_sanitizer_log/
  └── timestamp.txt           # Time stamp for execution

EOF
}

# ============================================================================
# Parsing command line Parameters
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --keep-build)
                KEEP_BUILD=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            *)
                log_error "Unknown Arguments: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# Parsing JSON configuration file
# ============================================================================

parse_config() {
    local config_file="$1"

    if [ ! -f "$config_file" ]; then
        log_error "configuration filedoes not exist: $config_file"
        exit 1
    fi

    log_info "Readconfiguration file: $config_file"

    # Parsing JSON with Python (more reliable)
    if ! command -v python3 &> /dev/null; then
        log_error "Yes. Python 3 To parse it.configuration file"
        exit 1
    fi

    # Extract Configuration Value
    OP_NAME=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('operator', {}).get('name', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo operator.name"; exit 1; }

    CODE_BASE_DIR=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('paths', {}).get('code_base_dir', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo paths.code_base_dir"; exit 1; }
    TEST_SCRIPT_DIR=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('testing', {}).get('test_script_dir', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo testing.test_script_dir"; exit 1; }


    TEST_SCRIPT_EXE=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('testing', {}).get('test_script_exe', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo testing.test_script_exe"; exit 1; }

    DEVICE_TYPE=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('environment', {}).get('device_type', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo environment.device_type"; exit 1; }

    CANN_ENV=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('environment', {}).get('cann_env', ''))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo environment.cann_env"; exit 1; }
    LOAD_ENV=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    val = config.get('installation', {}).get('load_environment', True)
    print('true' if val else 'false')
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo installation.load_environment"; exit 1; }

    SANITIZER_OPTS=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('compilation', {}).get('sanitizer_options', '-sanitizer;-g'))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo compilation.sanitizer_options"; exit 1; }

    LOG_LEVEL=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('memcheck', {}).get('log_level', '3'))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo memcheck.log_level"; exit 1; }

    SLOG_PRINT=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('memcheck', {}).get('slog_print_to_stdout', 'true'))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo memcheck.slog_print_to_stdout"; exit 1; }

    TIMEOUT=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    print(config.get('memcheck', {}).get('timeout', 600))
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo memcheck.timeout"; exit 1; }

    REBUILD=$(python3 -c "
import json
import sys
try:
    with open('$config_file', 'r') as f:
        config = json.load(f)
    val = config.get('options', {}).get('rebuild', True)
    print('true' if val else 'false')
except Exception as e:
    sys.exit(1)
" 2>/dev/null) || { log_error "Could not initialise Bonobo options.rebuild"; exit 1; }

    # Validate Required Fields
    if [ -z "$OP_NAME" ]; then
        log_error "configuration fileRequired Fields Missing: operator.name"
        exit 1
    fi
    if [ -z "$CODE_BASE_DIR" ]; then
        log_error "configuration fileRequired Fields Missing: paths.code_base_dir"
        exit 1
    fi
    if [ -z "$DEVICE_TYPE" ]; then
        log_error "configuration fileRequired Fields Missing: environment.device_type"
        exit 1
    fi
    if [ -z "$CANN_ENV" ]; then
        log_error "configuration fileRequired Fields Missing: environment.cann_env"
        exit 1
    fi
    # Build a custom installation path (code_base_dir/kustom_output)
    INSTALL_PATH="$CODE_BASE_DIR/custom_output_dir"
    log_verbose "Configure parsing complete:"
    log_verbose "  OP_NAME: $OP_NAME"
    log_verbose "  CODE_BASE_DIR: $CODE_BASE_DIR"
    log_verbose "  TEST_SCRIPT_EXE: $TEST_SCRIPT_EXE"
    log_verbose "  DEVICE_TYPE: $DEVICE_TYPE"
    log_verbose "  CANN_ENV: $CANN_ENV"
    log_verbose "  INSTALL_PATH: $INSTALL_PATH"
    log_verbose "  SANITIZER_OPTS: $SANITIZER_OPTS"
    log_verbose "  TEST_SCRIPT_DIR: $TEST_SCRIPT_DIR"
    log_verbose "  REBUILD: $REBUILD"
}

# ============================================================================
# Environmental validation
# ============================================================================

check_environment() {
    log_info "===== Validate the operating environment ====="
    echo "CANN_ENV:$CANN_ENV"

    # Check CANN environment
    if [ ! -f "$CANN_ENV/bin/setenv.bash" ]; then
        log_error "CANN Environment script does not exist: $CANN_ENV/bin/setenv.bash"
        exit 1
    fi
    log_success "CANN Environment script: $CANN_ENV"

    # Check the code directory
    if [ ! -d "$CODE_BASE_DIR" ]; then
        log_error "Code directory does not exist: $CODE_BASE_DIR"
        exit 1
    fi

    BUILD_SCRIPT="$CODE_BASE_DIR/build.sh"
    if [ ! -f "$BUILD_SCRIPT" ]; then
        log_error "Script does not exist: $BUILD_SCRIPT"
        exit 1
    fi
    log_success "Code Directory: $CODE_BASE_DIR"
    log_success "Compile Script: $BUILD_SCRIPT"

    # Check mssanitizer
    if ! command -v mssanitizer &> /dev/null; then
        log_warning "mssanitizer Not here. PATH , will load CANN Retry after Environment"
    else
        log_success "mssanitizer: $(command -v mssanitizer)"
    fi

    # Create Output Directory
    OUTPUT_DIR="$CODE_BASE_DIR/memcheck_output"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/build"
    mkdir -p "$OUTPUT_DIR/install"
    mkdir -p "$OUTPUT_DIR/memcheck"
    log_success "Output Directory: $OUTPUT_DIR"
}

# ============================================================================
# Step 1: Compile operator
# ============================================================================

build_operator() {
    log_info "===== I don't think so. 1 Step: Compileoperator ====="

    # Check to skip compile
    if [ "$SKIP_BUILD" = true ] || [ "$REBUILD" = false ]; then
        log_warning "Skip Compiler Steps"
        return 0
    fi

    # Load CANN Environment
    source "$CANN_ENV/bin/setenv.bash"
    log_info "Load CANN Environment: source $CANN_ENV/bin/setenv.bash"

    # Enter Code Directory
    log_info "Enter Code Directory: $CODE_BASE_DIR"
    cd "$CODE_BASE_DIR" || exit 1

    # Execute Compile
    local build_cmd="bash build.sh -n $OP_NAME -c $DEVICE_TYPE -p $CANN_ENV --ops-compile-options \"$SANITIZER_OPTS\""
    log_info "Execute Compile: $build_cmd"
    echo "outdir: $OUTPUT_DIR/build/build.log"

    if eval "$build_cmd" > "$OUTPUT_DIR/build/build.log" 2>&1; then
        log_success "Compiled successfully"
    else
        log_error "Compiled failed"
        # Extract error message
        tail -50 "$OUTPUT_DIR/build/build.log" | tee "$OUTPUT_DIR/build/build_errors.log"
        exit 1
    fi

    if [ "$VERBOSE" = true ]; then
        log_verbose "Compile log saved to: $OUTPUT_DIR/build/build.log"
    fi

    # Check the compiler.
    local package_dir="$CODE_BASE_DIR/output"
    local install_package=$(find "$package_dir" -name "*.run" | head -1)

    if [ -z "$install_package" ]; then
        log_error "No compiler found (%2).run Documentation): $package_dir"
        exit 1
    fi

    log_success "Compiled product: $install_package"

    # Save compiler path
    echo "$install_package" > "$OUTPUT_DIR/build/package_path.txt"
}

# ============================================================================
# Step 2: Installation of operator
# ============================================================================

install_operator() {
    log_info "===== I don't think so. 2 Step: Installationoperator ====="

    # Find Compilers
    local package_dir="$CODE_BASE_DIR/output"
    local install_package=$(find "$package_dir" -name "*.run" | head -1)

    if [ -z "$install_package" ]; then
        log_error "No compiler found (%2).run Documentation): $package_dir"
        log_error "Please proceed with the process of compilation."
        exit 1
    fi

    log_info "Install package: $install_package"

    # Create installation directory
    if [ ! -d "$INSTALL_PATH" ]; then
        log_info "Create installation directory: $INSTALL_PATH"
        mkdir -p "$INSTALL_PATH"
    else
        log_info "Clear old installation: $INSTALL_PATH"
        rm -rf "$INSTALL_PATH"
        log_info "Recreate Installation Directory: $INSTALL_PATH"
        mkdir -p "$INSTALL_PATH"
    fi

    # Execute installation
    log_info "Installed to: $INSTALL_PATH"
    if "$install_package" --install-path="$INSTALL_PATH" > "$OUTPUT_DIR/install/install.log" 2>&1; then
        log_success "Installed successfully"
    else
        log_error "Installation Failed"
        tail -50 "$OUTPUT_DIR/install/install.log" | tee "$OUTPUT_DIR/install/install_errors.log"
        exit 1
    fi

    if [ "$VERBOSE" = true ]; then
        log_verbose "Installation log saved to: $OUTPUT_DIR/install/install.log"
    fi

    # Load operator environment
    if [ "$LOAD_ENV" = true ]; then
        local env_script=$(find "$INSTALL_PATH/vendors" -path "*/bin/set_env.bash" -print -quit 2>/dev/null)
        if [ -f "$env_script" ]; then
            log_info "LoadoperatorEnvironment: $env_script"
            source "$env_script"
            log_success "operatorEnvironment loaded source $env_script"
        else
            log_warning "operatorEnvironment script not found: $env_script"
            log_warning "It may require manual loading environment"
        fi
    fi
}

# ============================================================================
# Step 3: Run MemCheck
# ============================================================================

run_memcheck() {
    log_info "===== I don't think so. 3 Step: Run MemCheck ====="

    # Use absolute path to get a test script directory and name
    local test_script_dir="$TEST_SCRIPT_DIR"

    # Enter the test directory
    if [ ! -d "$test_script_dir" ]; then
        log_error "Test directory does not exist: $test_script_dir"
        exit 1
    fi

    log_info "Enter the test directory: $test_script_dir"
    cd "$test_script_dir" || exit 1

    # Set Environmental Variables
    log_info "Set Environmental Variables:"
    log_info "  ASCEND_GLOBAL_LOG_LEVEL=$LOG_LEVEL"
    log_info "  ASCEND_SLOG_PRINT_TO_STDOUT=$SLOG_PRINT"

    export ASCEND_GLOBAL_LOG_LEVEL="$LOG_LEVEL"
    export ASCEND_SLOG_PRINT_TO_STDOUT="$SLOG_PRINT"

    # Run memcheck
    local memcheck_cmd="mssanitizer --tool=memcheck $TEST_SCRIPT_EXE"
    log_info "Implementation Memcheck: $memcheck_cmd"

    # Save original report
    local raw_report="$OUTPUT_DIR/memcheck/ascendc_memcheck_report_raw.txt"

    if eval "$memcheck_cmd" > "$raw_report" 2>&1; then
        log_success "Memcheck Implementation complete"
    else
        log_warning "Memcheck Execute completed (possibly detected errors)"
    fi

    # Save full log
    cp "$raw_report" "$OUTPUT_DIR/memcheck/memcheck.log"

    # Find and copy mendstudio_sanitizer_log directory
    if [ -d "mindstudio_sanitizer_log" ]; then
        cp -r mindstudio_sanitizer_log "$OUTPUT_DIR/memcheck/"
        log_success "Sanitizer Log copied to: $OUTPUT_DIR/memcheck/mindstudio_sanitizer_log"
    fi

    # Extracting the summary of the report
    log_info "===== Memcheck Summary of results ====="

    local error_count=$(grep -c "====== ERROR:" "$raw_report" 2>/dev/null || echo "0")
    local warning_count=$(grep -c "====== WARNING:" "$raw_report" 2>/dev/null || echo "0")

    echo "ERROR Number: $error_count"
    echo "WARNING Number: $warning_count"

    if [ "$error_count" -gt 0 ]; then
        grep "====== ERROR:" "$raw_report" | head -5
    fi

    if [ "$warning_count" -gt 0 ]; then
        grep "====== WARNING:" "$raw_report" | head -5
    fi

    # Check the test results.
    grep -E "collected|passed|FAILED" "$raw_report" | tail -10
}

# ============================================================================
# Generate Status Report
# ============================================================================

generate_status_report() {
    log_info "===== Generate Status Report ====="

    local status_file="$OUTPUT_DIR/status.txt"

    cat > "$status_file" << EOF
AscendC Memcheck Status of implementation report
================================

Implementation time: $(date)
configuration file: $CONFIG_FILE

Implementation steps:
EOF

    # Compile Status
    if [ "$SKIP_BUILD" = true ] || [ "$REBUILD" = false ]; then
        echo "  I don't think so. 1 Step (compilation): Skip" >> "$status_file"
    else
        if [ -f "$OUTPUT_DIR/build/build.log" ]; then
            echo "  I don't think so. 1 Step (compilation): Success" >> "$status_file"
        else
            echo "  I don't think so. 1 Step (compilation): Failed" >> "$status_file"
        fi
    fi

    # Install Status
    if [ -f "$OUTPUT_DIR/install/install.log" ]; then
        echo "  I don't think so. 2 Step (installation): Success" >> "$status_file"
    else
        echo "  I don't think so. 2 Step (installation): Failed" >> "$status_file"
    fi

    # Memcheck Status
    if [ -f "$OUTPUT_DIR/memcheck/memcheck.log" ]; then
        echo "  I don't think so. 3 StepMemcheck): Success" >> "$status_file"
    else
        echo "  I don't think so. 3 StepMemcheck): Failed" >> "$status_file"
    fi

    echo "" >> "$status_file"
    echo "Output File:" >> "$status_file"
    echo "  Build Log: $OUTPUT_DIR/build/build.log" >> "$status_file"
    echo "  Install Log: $OUTPUT_DIR/install/install.log" >> "$status_file"
    echo "  Memcheck Log: $OUTPUT_DIR/memcheck/memcheck.log" >> "$status_file"
    echo "  Original report: $OUTPUT_DIR/memcheck/ascendc_memcheck_report_raw.txt"
    echo "" >> "$status_file"

    # Save Timetamp
    date > "$OUTPUT_DIR/timestamp.txt"

    log_success "Status report generated: $status_file"
    cat "$status_file"
}

# ============================================================================
# Clear Build Directory
# ============================================================================

cleanup_build() {
    if [ "$KEEP_BUILD" = false ]; then
        log_info "Clear Build Directory (optional)"
        # Here you can add cleanup logic.
        # For example: cd "$CODE_BASE_DIR" & rm-ref built/
    fi
}

# ============================================================================
# Main Functions
# ============================================================================

main() {
    log_info "===== AscendC Memcheck Automation Script ====="
    log_info "Start Time: $(date)"
    log_info ""

    # Parsing Parameters
    parse_arguments "$@"

    # Parsing Configuration
    parse_config "$CONFIG_FILE"

    # Validation Environment
    check_environment

    # Implementation of the steps
    build_operator
    install_operator
    run_memcheck

    # Generate Report
    generate_status_report

    # Cleaning
    cleanup_build

    log_info ""
    log_info "===== Implementation complete ====="
    log_info "End Time: $(date)"
    log_info "Output Directory: $OUTPUT_DIR"
    log_success "All steps are completed"
}

# ============================================================================
# script entrance
# ============================================================================

main "$@"
