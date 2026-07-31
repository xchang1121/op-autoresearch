import os
import shutil
import logging
from jinja2 import Template
from op_autoresearch import get_project_root
from op_autoresearch.core.worker.eval_config import (
    resolve_run_times,
    resolve_warmup_times,
)
from op_autoresearch.op.verifier.adapters.factory import (
    get_framework_adapter, get_dsl_adapter, get_backend_adapter
)

logger = logging.getLogger(__name__)

PROF_SOL_BASE_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "prof_sol_base_template.j2"
)
PROF_SOL_GENERATION_TEMPLATE_PATH = os.path.join(
    get_project_root(), "op", "resources", "templates", "prof_sol_generation_template.j2"
)

def generate_sol_verify_project(verifier, impl_code: str, verify_dir: str, device_id: int = 0):
    """Generate SOL-ExecBench to the specified directory"""
    logger.info(f"[{verifier.op_name}] Start Generating SOL-ExecBench Validation of projects, directories: {verify_dir}, device_id={device_id}")

    sol_problem_dir = verifier.config.get("sol_problem_dir")
    if not sol_problem_dir or not os.path.exists(sol_problem_dir):
        raise ValueError(f"sol_problem_dir is missing or does not exist: {sol_problem_dir}")

    # 1. Copies of the SOL Core Document
    for file_name in ["definition.json", "workload.jsonl", "reference.py"]:
        src_file = os.path.join(sol_problem_dir, file_name)
        dst_file = os.path.join(verify_dir, file_name)
        if not os.path.exists(src_file):
            raise FileNotFoundError(f"Missing required SOL file: {src_file}")
        shutil.copy2(src_file, dst_file)

    # 2. Copy sol_correctness.py
    sol_correctness_src = os.path.join(get_project_root(), "op", "resources", "utils", "sol_correctness.py")
    sol_correctness_dst = os.path.join(verify_dir, "sol_correctness.py")
    if not os.path.exists(sol_correctness_src):
        raise FileNotFoundError(f"Missing sol_correctness.py: {sol_correctness_src}")
    shutil.copy2(sol_correctness_src, sol_correctness_dst)

    # 3. Creation of a specific implementation document
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

    # 4. Generate authentication scripts
    verify_file = os.path.join(verify_dir, f"verify_{verifier.op_name}.py")
    template_path = os.path.join(get_project_root(), "op", "resources", "templates", "verify_sol_template.j2")

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
        # Fix for Python module names starting with numbers
        if dsl_impl_import.startswith("from ") and dsl_impl_import.split(" ")[1][0].isdigit():
            module_name = dsl_impl_import.split(" ")[1]
            import_name = dsl_impl_import.split(" ")[3].strip()
            dsl_impl_import = f"import importlib.util\nimport sys\nspec = importlib.util.spec_from_file_location('{module_name}', '{module_name}.py')\nmodule = importlib.util.module_from_spec(spec)\nsys.modules['{module_name}'] = module\nspec.loader.exec_module(module)\n{import_name} = getattr(module, '{import_name}')"

        dsl_adapter.prepare_config(verifier.config, task_info=None)
        special_setup = dsl_adapter.get_special_setup_code(
            framework=verifier.framework
        )
        dsl_imports += "\n" + dsl_impl_import
        if special_setup:
            dsl_imports += "\n" + special_setup

        backend_adapter.setup_environment(device_id, verifier.arch)
        create_impl_code = verifier._prepare_code_lines(dsl_adapter.create_impl_module(verifier.framework, framework_adapter))
        device_setup_code = verifier._prepare_code_lines(framework_adapter.get_device_setup_code(verifier.backend, verifier.arch, device_id))

        sol_execbench_src_dir = os.path.abspath(os.path.join(get_project_root(), "..", "..", "thirdparty", "sol-execbench", "src"))

        # Render Template
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
            sol_execbench_src_dir=sol_execbench_src_dir
        )

        with open(verify_file, "w", encoding="utf-8") as f:
            f.write(verify_script)

    except Exception as e:
        logger.error(f"[{verifier.op_name}] Authentication script generation failed: {e}")
        raise


def _get_sol_common_template_vars(verifier, device_id: int):
    """Retrieving public variables for the SOL profile template (shared with base and source template)"""
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

    sol_execbench_src_dir = os.path.abspath(
        os.path.join(get_project_root(), "..", "..", "thirdparty", "sol-execbench", "src")
    )

    return {
        "device_setup_code": device_setup_code,
        "dsl_imports": dsl_imports,
        "create_impl_code": create_impl_code,
        "sol_execbench_src_dir": sol_execbench_src_dir,
    }


def generate_sol_profile_project(verifier, verify_dir: str, device_id: int = 0,
                                  warmup_times: int | None = None,
                                  run_times: int | None = None,
                                  skip_base: bool = False):
    """Generate SOL-ExecBench Performance Test Project Files to a specified directory

    CorrelBench's gen_profile_project generated two profile scripts:
    -Profile_{op_name}_base.py to measure access.run
    -profile_{op_name}_generation.py to measure the performance achieved by generation

    Both scripts output JSON files compatible with KernelBench
    (base_profile_result.json / generation_profile_result.json),
    Downstream profiler_utils.run_profile_scripts_and_collect_results can be used directly.
    """
    warmup_times = resolve_warmup_times(warmup_times)
    run_times = resolve_run_times(run_times)
    logger.info(
        f"[{verifier.op_name}] Start Generating SOL-ExecBench The performance test project,"
        f"Contents: {verify_dir}, device_id={device_id}"
    )

    sol_problem_dir = verifier.config.get("sol_problem_dir")
    if not sol_problem_dir or not os.path.exists(sol_problem_dir):
        raise ValueError(f"sol_problem_dir is missing or does not exist: {sol_problem_dir}")

    # Ensure that SOL data files exist (gen_verify_project is usually copied)
    for file_name in ["definition.json", "workload.jsonl", "reference.py"]:
        dst_file = os.path.join(verify_dir, file_name)
        if not os.path.exists(dst_file):
            src_file = os.path.join(sol_problem_dir, file_name)
            if not os.path.exists(src_file):
                raise FileNotFoundError(f"Missing required SOL file: {src_file}")
            shutil.copy2(src_file, dst_file)

    # Ensure that sol_correctness.py exists
    sol_correctness_dst = os.path.join(verify_dir, "sol_correctness.py")
    if not os.path.exists(sol_correctness_dst):
        sol_correctness_src = os.path.join(
            get_project_root(), "op", "resources", "utils", "sol_correctness.py"
        )
        shutil.copy2(sol_correctness_src, sol_correctness_dst)

    profile_generation_enabled = getattr(
        verifier, "_profile_generation_enabled", True)

    framework_adapter = get_framework_adapter(verifier.framework)
    backend_adapter = get_backend_adapter(verifier.backend)
    backend_adapter.setup_environment(device_id, verifier.arch)
    base_device_setup_code = verifier._prepare_code_lines(
        framework_adapter.get_device_setup_code(
            verifier.backend, verifier.arch, device_id)
    )
    sol_execbench_src_dir = os.path.abspath(
        os.path.join(get_project_root(), "..", "..",
                     "thirdparty", "sol-execbench", "src")
    )

    # Generate base profile script (measurement of reference.run)
    if not skip_base:
        try:
            with open(PROF_SOL_BASE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                base_template = Template(f.read())
            base_script = base_template.render(
                op_name=verifier.op_name,
                backend=verifier.backend,
                arch=verifier.arch,
                device_id=device_id,
                warmup_times=warmup_times,
                run_times=run_times,
                device_setup_code=base_device_setup_code,
                sol_execbench_src_dir=sol_execbench_src_dir,
            )
            base_path = os.path.join(verify_dir, f"profile_{verifier.op_name}_base.py")
            with open(base_path, "w", encoding="utf-8") as f:
                f.write(base_script)
            logger.info(f"[{verifier.op_name}] SOL base profile Script written: {base_path}")
        except Exception as e:
            logger.error(f"[{verifier.op_name}] SOL base profile Script Generation Failed: {e}")
            raise
    else:
        logger.info(f"[{verifier.op_name}] Skip SOL base profile Generateskip_base=True)")

    if not profile_generation_enabled:
        logger.info(f"[{verifier.op_name}] Skip SOL generation profile Generate (Previous round) verify Not adopted)")
        return

    common_vars = _get_sol_common_template_vars(verifier, device_id)

    # Generate Generation profile scripts (measure the performance of generation)
    try:
        with open(PROF_SOL_GENERATION_TEMPLATE_PATH, "r", encoding="utf-8") as f:
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
            sol_execbench_src_dir=common_vars["sol_execbench_src_dir"],
        )
        gen_path = os.path.join(verify_dir, f"profile_{verifier.op_name}_generation.py")
        with open(gen_path, "w", encoding="utf-8") as f:
            f.write(gen_script)
        logger.info(f"[{verifier.op_name}] SOL generation profile Script written: {gen_path}")
    except Exception as e:
        logger.error(f"[{verifier.op_name}] SOL generation profile Script Generation Failed: {e}")
        raise
