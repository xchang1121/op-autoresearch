"""CANN-Bench format verification project generator.

Responsibilities:
1. Generate CANN-Bench verification project (verify script + data files)
2. Generate CANN-Bench profile project (profile scripts)

Pattern follows sol_verifier.py: sys.path imports CANN-Bench source repo
for DataGenerator, ParamBuilder, etc., instead of reimplementing.
"""

import os
import shutil
import logging
import json
import yaml
from jinja2 import Template
from op_autoresearch import get_project_root
from op_autoresearch.core.worker.eval_config import (
    resolve_run_times,
    resolve_warmup_times,
)
from op_autoresearch.op.verifier.adapters.factory import (
    get_framework_adapter, get_dsl_adapter, get_backend_adapter,
)

logger = logging.getLogger(__name__)

# Package-relative — templates and core.py travel with this package, not the
# op/resources tree, so the cannbench eval-standard stays self-contained.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.join(_PKG_DIR, "templates")
_CORE_PY_PATH = os.path.join(_PKG_DIR, "core.py")

PROF_CANN_BASE_TEMPLATE_PATH = os.path.join(_TEMPLATES_DIR, "prof_cann_base.j2")
PROF_CANN_GENERATION_TEMPLATE_PATH = os.path.join(_TEMPLATES_DIR, "prof_cann_generation.j2")

# CANN-Bench source tree — provides the `kernel_eval` package the generated
# verify/profile scaffolds import via sys.path. Defaults to the vendored
# thirdparty checkout; override with OP_AUTORESEARCH_CANN_BENCH_SRC when cann-bench lives
# elsewhere on the host (same pattern as OP_AUTORESEARCH_AR_SKILLS_ROOT). Single
# owner — baseline_profiler imports this.
CANN_BENCH_SRC_DIR = os.environ.get(
    "OP_AUTORESEARCH_CANN_BENCH_SRC",
    os.path.abspath(
        os.path.join(get_project_root(), "..", "..", "thirdparty", "cann-bench", "src")
    ),
)

CANN_DATA_FILES = ["proto.yaml", "golden.py", "cases.yaml"]


def generate_cann_verify_project(verifier, impl_code: str, verify_dir: str, device_id: int = 0):
    """Generate CANN-Bench verification project files into verify_dir."""
    logger.info(
        f"[{verifier.op_name}] Start Generating CANN-Bench Validation of projects,"
        f"Contents: {verify_dir}, device_id={device_id}"
    )

    cann_problem_dir = verifier.config.get("cann_problem_dir")
    if not cann_problem_dir or not os.path.exists(cann_problem_dir):
        raise ValueError(f"cann_problem_dir is missing or does not exist: {cann_problem_dir}")

    # 1. Copy CANN-Bench data files
    for file_name in CANN_DATA_FILES:
        src_file = os.path.join(cann_problem_dir, file_name)
        dst_file = os.path.join(verify_dir, file_name)
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Missing required CANN file: {src_file}")
        shutil.copy2(src_file, dst_file)

    # Copy desc.md if exists
    desc_src = os.path.join(cann_problem_dir, "desc.md")
    if os.path.exists(desc_src):
        shutil.copy2(desc_src, os.path.join(verify_dir, "desc.md"))

    # 2. Copy core.py in as cann_correctness.py (the name the scaffold imports)
    if not os.path.exists(_CORE_PY_PATH):
        raise FileNotFoundError(f"Missing core.py: {_CORE_PY_PATH}")
    shutil.copy2(_CORE_PY_PATH, os.path.join(verify_dir, "cann_correctness.py"))

    # 3. Create implementation file
    dsl_adapter = get_dsl_adapter(verifier.dsl)
    if getattr(dsl_adapter, "kernel_arg_is_directory", False):
        dsl_adapter.prepare_config(verifier.config, task_info=None)
        dsl_adapter.materialize_impl(
            impl_code=impl_code,
            verify_dir=verify_dir,
            op_name=verifier.op_name,
            framework=verifier.framework,
            dsl_name=verifier.dsl,
            task_info=None,
            config=verifier.config,
        )
    else:
        file_name = f"{verifier.op_name}_{verifier.dsl}_impl.py"
        impl_file = os.path.join(verify_dir, file_name)

        try:
            dsl_adapter = get_dsl_adapter(verifier.dsl)
            import_statements = dsl_adapter.get_import_statements(verifier.framework)
        except Exception as e:
            logger.error(f"[{verifier.op_name}] DSL importstatement generation failed: {e}")
            raise

        try:
            with open(impl_file, "w", encoding="utf-8") as f:
                f.write(import_statements + impl_code)
        except Exception as e:
            logger.error(f"[{verifier.op_name}] Failed to achieve file creation: {impl_file}, Error: {e}")
            raise

    # 4. Load proto.yaml for template rendering
    proto_path = os.path.join(verify_dir, "proto.yaml")
    with open(proto_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)

    op_spec = proto.get("operator", {})
    precision_thresholds = op_spec.get("precision_thresholds")
    outputs = op_spec.get("outputs", [])
    ignore_output_indices = [
        i for i, out in enumerate(outputs) if out.get("compare", True) is False
    ]

    # 5. Generate verify script
    verify_file = os.path.join(verify_dir, f"verify_{verifier.op_name}.py")
    template_path = os.path.join(_TEMPLATES_DIR, "verify_cann.j2")

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = Template(f.read())
    except Exception as e:
        logger.error(f"[{verifier.op_name}] Failed to load template file: {template_path}, Error: {e}")
        raise

    try:
        framework_adapter = get_framework_adapter(verifier.framework)
        dsl_adapter = get_dsl_adapter(verifier.dsl)
        backend_adapter = get_backend_adapter(verifier.backend)
    except Exception as e:
        logger.error(f"[{verifier.op_name}] AdaptersInitialization failed: {e}")
        raise

    try:
        dsl_imports = dsl_adapter.get_import_statements(verifier.framework)
        dsl_impl_import = dsl_adapter.get_impl_import(verifier.op_name, verifier.impl_func_name).strip()
        if dsl_impl_import.startswith("from ") and dsl_impl_import.split(" ")[1][0].isdigit():
            module_name = dsl_impl_import.split(" ")[1]
            import_name = dsl_impl_import.split(" ")[3].strip()
            dsl_impl_import = (
                f"import importlib.util\nimport sys\n"
                f"spec = importlib.util.spec_from_file_location('{module_name}', '{module_name}.py')\n"
                f"module = importlib.util.module_from_spec(spec)\n"
                f"sys.modules['{module_name}'] = module\n"
                f"spec.loader.exec_module(module)\n"
                f"{import_name} = getattr(module, '{import_name}')"
            )
        dsl_adapter.prepare_config(verifier.config, task_info=None)
        special_setup = dsl_adapter.get_special_setup_code(
            framework=verifier.framework
        )
        dsl_imports += "\n" + dsl_impl_import
        if special_setup:
            dsl_imports += "\n" + special_setup

        backend_adapter.setup_environment(device_id, verifier.arch)
        create_impl_code = verifier._prepare_code_lines(
            dsl_adapter.create_impl_module(verifier.framework, framework_adapter)
        )
        device_setup_code = verifier._prepare_code_lines(
            framework_adapter.get_device_setup_code(verifier.backend, verifier.arch, device_id)
        )

        # Serialize precision_thresholds for template injection
        precision_thresholds_yaml = json.dumps(precision_thresholds) if precision_thresholds else "None"

        verify_script = template.render(
            op_name=verifier.op_name,
            framework=verifier.framework,
            backend=verifier.backend,
            arch=verifier.arch,
            dsl=verifier.dsl,
            device_id=device_id,
            dsl_imports=dsl_imports,
            device_setup_code=device_setup_code,
            create_impl_code=create_impl_code,
            precision_thresholds_yaml=precision_thresholds_yaml,
            ignore_output_indices=ignore_output_indices,
            schema=op_spec.get("schema", ""),
            cann_bench_src_dir=CANN_BENCH_SRC_DIR,
        )

        with open(verify_file, "w", encoding="utf-8") as f:
            f.write(verify_script)

    except Exception as e:
        logger.error(f"[{verifier.op_name}] Authentication script generation failed: {e}")
        raise


def generate_cann_profile_project(verifier, verify_dir: str, device_id: int = 0,
                                   warmup_times: int | None = None,
                                   run_times: int | None = None,
                                   skip_base: bool = False):
    """Generate CANN-Bench profile project files into verify_dir.

    Produces two profile scripts:
    - profile_{op_name}_base.py: measure golden.py performance
    - profile_{op_name}_generation.py: measure generated implementation performance
    """
    warmup_times = resolve_warmup_times(warmup_times)
    run_times = resolve_run_times(run_times)
    logger.info(
        f"[{verifier.op_name}] Start Generating CANN-Bench The performance test project,"
        f"Contents: {verify_dir}, device_id={device_id}"
    )

    cann_problem_dir = verifier.config.get("cann_problem_dir")
    if not cann_problem_dir or not os.path.exists(cann_problem_dir):
        raise ValueError(f"cann_problem_dir is missing or does not exist: {cann_problem_dir}")

    # Ensure CANN data files exist (may already be copied by gen_verify_project)
    for file_name in CANN_DATA_FILES:
        dst_file = os.path.join(verify_dir, file_name)
        if not os.path.exists(dst_file):
            src_file = os.path.join(cann_problem_dir, file_name)
            if not os.path.exists(src_file):
                raise FileNotFoundError(f"Missing required CANN file: {src_file}")
            shutil.copy2(src_file, dst_file)

    desc_src = os.path.join(cann_problem_dir, "desc.md")
    if os.path.exists(desc_src):
        dst = os.path.join(verify_dir, "desc.md")
        if not os.path.exists(dst):
            shutil.copy2(desc_src, dst)

    # Ensure cann_correctness.py exists (not strictly needed for profile, but consistency)
    cann_correctness_dst = os.path.join(verify_dir, "cann_correctness.py")
    if not os.path.exists(cann_correctness_dst):
        shutil.copy2(_CORE_PY_PATH, cann_correctness_dst)

    # Load proto.yaml for template
    proto_path = os.path.join(verify_dir, "proto.yaml")
    with open(proto_path, "r", encoding="utf-8") as f:
        proto = yaml.safe_load(f)
    op_spec = proto.get("operator", {})
    schema = op_spec.get("schema", "")

    profile_generation_enabled = getattr(
        verifier, "_profile_generation_enabled", True)

    framework_adapter = get_framework_adapter(verifier.framework)
    backend_adapter = get_backend_adapter(verifier.backend)
    backend_adapter.setup_environment(device_id, verifier.arch)
    base_device_setup_code = verifier._prepare_code_lines(
        framework_adapter.get_device_setup_code(
            verifier.backend, verifier.arch, device_id)
    )

    # Generate base profile script (measure golden.py)
    if not skip_base:
        try:
            with open(PROF_CANN_BASE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                base_template = Template(f.read())
            base_script = base_template.render(
                op_name=verifier.op_name,
                backend=verifier.backend,
                arch=verifier.arch,
                dsl=verifier.dsl,
                device_id=device_id,
                warmup_times=warmup_times,
                run_times=run_times,
                device_setup_code=base_device_setup_code,
                schema=schema,
                cann_bench_src_dir=CANN_BENCH_SRC_DIR,
            )
            base_path = os.path.join(verify_dir, f"profile_{verifier.op_name}_base.py")
            with open(base_path, "w", encoding="utf-8") as f:
                f.write(base_script)
            logger.info(f"[{verifier.op_name}] CANN base profile Script written: {base_path}")
        except Exception as e:
            logger.error(f"[{verifier.op_name}] CANN base profile Script Generation Failed: {e}")
            raise
    else:
        logger.info(f"[{verifier.op_name}] Skip CANN base profile Generateskip_base=True)")

    if not profile_generation_enabled:
        logger.info(f"[{verifier.op_name}] Skip CANN generation profile Generate (Previous round) verify Not adopted)")
        return

    common_vars = _get_cann_common_template_vars(verifier, device_id)

    # Generate generation profile script
    try:
        with open(PROF_CANN_GENERATION_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            gen_template = Template(f.read())
        gen_script = gen_template.render(
            op_name=verifier.op_name,
            backend=verifier.backend,
            arch=verifier.arch,
            dsl=verifier.dsl,
            device_id=device_id,
            warmup_times=warmup_times,
            run_times=run_times,
            device_setup_code=common_vars["device_setup_code"],
            dsl_imports=common_vars["dsl_imports"],
            create_impl_code=common_vars["create_impl_code"],
            schema=schema,
            cann_bench_src_dir=common_vars["cann_bench_src_dir"],
        )
        gen_path = os.path.join(verify_dir, f"profile_{verifier.op_name}_generation.py")
        with open(gen_path, "w", encoding="utf-8") as f:
            f.write(gen_script)
        logger.info(f"[{verifier.op_name}] CANN generation profile Script written: {gen_path}")
    except Exception as e:
        logger.error(f"[{verifier.op_name}] CANN generation profile Script Generation Failed: {e}")
        raise


def _get_cann_common_template_vars(verifier, device_id: int):
    """Get common template variables for CANN profile scripts."""
    framework_adapter = get_framework_adapter(verifier.framework)
    dsl_adapter = get_dsl_adapter(verifier.dsl)
    backend_adapter = get_backend_adapter(verifier.backend)

    backend_adapter.setup_environment(device_id, verifier.arch)
    device_setup_code = verifier._prepare_code_lines(
        framework_adapter.get_device_setup_code(verifier.backend, verifier.arch, device_id)
    )

    dsl_imports = dsl_adapter.get_import_statements(verifier.framework)
    dsl_impl_import = dsl_adapter.get_impl_import(verifier.op_name, verifier.impl_func_name).strip()
    if dsl_impl_import.startswith("from ") and dsl_impl_import.split(" ")[1][0].isdigit():
        module_name = dsl_impl_import.split(" ")[1]
        import_name = dsl_impl_import.split(" ")[3].strip()
        dsl_impl_import = (
            f"import importlib.util\nimport sys\n"
            f"spec = importlib.util.spec_from_file_location('{module_name}', '{module_name}.py')\n"
            f"module = importlib.util.module_from_spec(spec)\n"
            f"sys.modules['{module_name}'] = module\n"
            f"spec.loader.exec_module(module)\n"
            f"{import_name} = getattr(module, '{import_name}')"
        )
    dsl_adapter.prepare_config(verifier.config, task_info=None)
    special_setup = dsl_adapter.get_special_setup_code(
        framework=verifier.framework
    )
    dsl_imports += "\n" + dsl_impl_import
    if special_setup:
        dsl_imports += "\n" + special_setup

    create_impl_code = verifier._prepare_code_lines(
        dsl_adapter.create_impl_module(verifier.framework, framework_adapter)
    )

    return {
        "device_setup_code": device_setup_code,
        "dsl_imports": dsl_imports,
        "create_impl_code": create_impl_code,
        "cann_bench_src_dir": CANN_BENCH_SRC_DIR,
    }
