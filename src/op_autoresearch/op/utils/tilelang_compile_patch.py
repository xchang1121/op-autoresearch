import subprocess
from pathlib import Path


def patch_tilelang_compiler():
    """Dynamic patch tilelang compiler, add detailed error message display"""
    try:
        from tilelang.jit import jit_npu
    except ImportError:
        return False

    if not hasattr(jit_npu, 'compiler_npu'):
        return False

    compiler_class = jit_npu.compiler_npu

    # Check if you've been patched.
    if hasattr(compiler_class._npuir_to_bin_enable_npu_compile, '_op_autoresearch_patched'):
        return True

    original_compile_method = compiler_class._npuir_to_bin_enable_npu_compile

    def patched_npuir_to_bin_enable_npu_compile(self):
        """Compiler after patch, providing detailed error message"""
        import tempfile
        import os
        from pathlib import Path

        linalg = self.mlir_content
        metadata = self.metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            ttadapter_path = os.path.join(tmpdir, "kernel.npuir")
            Path(ttadapter_path).write_text(linalg)
            bin_file = os.path.join(tmpdir, "kernel")
            bin_path = os.path.join(tmpdir, "kernel.o")

            npu_compiler_path = jit_npu._get_npucompiler_path()
            _compile_option_list = [
                "--enable-auto-multi-buffer=true", "--enable-triton-kernel-compile=true",
                "--enable-hivm-compile=true", "--disable-hivm-tensor-compile=true"
            ]
            cmd_list = ([npu_compiler_path, ttadapter_path] + _compile_option_list +
                        ["-o", bin_file])

            try:
                ret = subprocess.run(cmd_list, capture_output=True, check=True, text=True)
            except subprocess.CalledProcessError as e:
                # Show complete compilation error message
                error_msg = f"\n{'='*60}\n"
                error_msg += f"NPU Compiler Error (exit code {e.returncode})\n"
                error_msg += f"Command: {' '.join(cmd_list)}\n"
                if e.stdout:
                    error_msg += f"\nSTDOUT:\n{e.stdout}\n"
                if e.stderr:
                    error_msg += f"\nSTDERR:\n{e.stderr}\n"
                raise RuntimeError(error_msg) from e

            return Path(bin_path).read_bytes()

    # Apply Patch
    try:
        compiler_class._npuir_to_bin_enable_npu_compile = patched_npuir_to_bin_enable_npu_compile
        # The tags have been patched.
        compiler_class._npuir_to_bin_enable_npu_compile._op_autoresearch_patched = True
        return True
    except (AttributeError, TypeError) as e:
        print(f"Warning: Failed to patch TileLang compiler: {e}")
        return False


def apply_tilelang_patches():
    """Apply all tilelang patches"""
    success = patch_tilelang_compiler()
    return success


# Automatically apply patches (when modules are imported)
if __name__ != "__main__":
    apply_tilelang_patches()

# Test Code
if __name__ == "__main__":
    print("Testing TileLang patches...")
    success = patch_tilelang_compiler()

    if success:
        print("✓ TileLang compiler patch applied successfully!")
    else:
        print("✗ Failed to apply TileLang compiler patch (tilelang may not be installed)")
