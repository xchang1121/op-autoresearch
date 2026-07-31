import os
import sys
import logging
from typing import Annotated, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import uvicorn

from op_autoresearch.core.worker.local_worker import LocalWorker
from op_autoresearch.core.async_pool.device_pool import DevicePool
from op_autoresearch.core.worker.eval_config import (
    resolve_eval_timeout,
    resolve_reference_timeout,
)
from op_autoresearch.cli.service.worker_config import worker_timing
from op_autoresearch.op.utils.config_utils import check_backend_arch
from op_autoresearch.op.utils.json_safe import sanitize_floats
from op_autoresearch.utils.process_utils import reap_orphaned_process_groups

# Configure logging. stream=sys.stdout keeps logs chronological under
# capture frameworks (see op_autoresearch/__init__.py).
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# Global worker instance
worker: Optional[LocalWorker] = None

def get_worker_config():
    """Get worker configuration from environment variables."""
    names = ("WORKER_BACKEND", "WORKER_ARCH", "WORKER_DEVICES")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(
            f"worker configuration missing: {', '.join(missing)}; "
            "start the daemon through `op-autoresearch worker --start`")

    backend, arch, devices_str = (os.environ[name].strip() for name in names)
    check_backend_arch(backend, arch)
    try:
        devices = [int(value.strip()) for value in devices_str.split(",")]
        if not devices or any(device < 0 for device in devices):
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            f"WORKER_DEVICES must be comma-separated non-negative integers, "
            f"got {devices_str!r}") from exc

    return backend, arch, devices

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize worker resources on startup."""
    global worker
    backend, arch, devices = get_worker_config()

    reaped = reap_orphaned_process_groups()
    if reaped:
        logger.warning("Reaped orphan eval process groups from predecessor: %s",
                       reaped)

    logger.info(f"Initializing Worker Service: Backend={backend}, Arch={arch}, Devices={devices}")

    timing = worker_timing()
    device_pool = DevicePool(devices, lease_ttl_s=timing.lease_ttl)
    device_pool.start_reaper(timing.lease_reap_interval)
    worker = LocalWorker(device_pool, backend=backend)

    yield

    await device_pool.stop_reaper()
    logger.info("Shutting down Worker Service")

app = FastAPI(title="OP_AUTORESEARCH Worker Service", lifespan=lifespan)


def _require_worker() -> None:
    """503 if the worker isn't initialized yet (startup race)."""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")


@asynccontextmanager
async def _guarded(task_id: str, *, device_id: Optional[int] = None,
                   lease_id: Optional[int] = None):
    """Shared eval-endpoint guard: renew the device lease for the whole
    request, and map any unhandled error to a 500 (HTTPExceptions — e.g. the
    400 from a bad profile_settings parse done before entering — pass
    through). Single owner of the keepalive + error-mapping boilerplate."""
    try:
        async with worker.device_pool.keepalive(
                task_id, device_id=device_id, lease_id=lease_id):
            yield
    except LookupError as e:
        # The script contains a device id acquired under this exact token.
        # If the token is stale, that device may already have a new owner.
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{task_id}] request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/verify")
async def verify(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    timeout: Optional[int] = Form(None),
    device_id: Optional[int] = Form(None),
    lease_id: Optional[int] = Form(None),
):
    """
    Execute verification task.

    Returns:
        - Access: Verify success
        -log: Execute Log
        -artifices: the content of the JSON file generated during execution
    """
    _require_worker()
    logger.info(f"[{task_id}] Received verification request for {op_name}")
    async with _guarded(task_id, device_id=device_id, lease_id=lease_id):
        package_data = await package.read()
        success, log, artifacts = await worker.verify(
            package_data, task_id, op_name, resolve_eval_timeout(timeout))
        return sanitize_floats({
            "success": success,
            "log": log,
            "artifacts": artifacts,
        })

@app.post("/api/v1/profile")
async def profile(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    profile_settings: str = Form("{}"),
    device_id: Optional[int] = Form(None),
    lease_id: Optional[int] = Form(None),
):
    """
    Execute profiling task.
    """
    _require_worker()
    import json
    try:
        settings = json.loads(profile_settings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for profile_settings")

    async with _guarded(task_id, device_id=device_id, lease_id=lease_id):
        package_data = await package.read()
        result = await worker.profile(package_data, task_id, op_name, settings)
        return sanitize_floats(result)

@app.post("/api/v1/generate_reference")
async def generate_reference(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    timeout: Optional[int] = Form(None),
    device_id: Optional[int] = Form(None),
    lease_id: Optional[int] = Form(None),
):
    """
    Execute task_desc and generate reference data.

    For the CUDA-to-Asend conversion scenario: execute the Triton-CUDA code,
    Saves the output as a reference data (.pt file) and returns with the base64 code.

    Returns:
        -success: Successfully generate reference data
        -log: Execute Log
        -Reference_data: base64 encoded.pt file content
    """
    import base64

    _require_worker()
    logger.info(f"[{task_id}] Received generate_reference request for {op_name}")
    async with _guarded(task_id, device_id=device_id, lease_id=lease_id):
        package_data = await package.read()
        success, log, ref_bytes = await worker.generate_reference(
            package_data, task_id, op_name,
            resolve_reference_timeout(timeout)
        )
        # When successful, base64 encoding returns binary data; fail-time string.
        return {
            "success": success,
            "log": log,
            "reference_data": (base64.b64encode(ref_bytes).decode('utf-8')
                               if success else ""),
        }

@app.post("/api/v1/profile_single_task")
async def profile_single_task(
    package: UploadFile = File(...),
    task_id: str = Form(...),
    op_name: str = Form(...),
    profile_settings: str = Form("{}"),
    device_id: Optional[int] = Form(None),
    lease_id: Optional[int] = Form(None),
):
    """
    Execute single task profiling (only measure task_desc performance, no base comparison).

    A separate measure of the performance of a certain section of the code does not allow comparison.

    Returns:
        - time_us: execution time (microseconds)
        - Success.
        -log: Execute Log
    """
    _require_worker()
    import json
    try:
        settings = json.loads(profile_settings)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for profile_settings")

    logger.info(f"[{task_id}] Received profile_single_task request for {op_name}")
    async with _guarded(task_id, device_id=device_id, lease_id=lease_id):
        package_data = await package.read()
        result = await worker.profile_single_task(package_data, task_id, op_name, settings)
        return sanitize_floats(result)

@app.get("/api/v1/docs/{doc_name}")
async def get_doc(
    doc_name: str,
):
    """
    Fetchs the contents of the document in the current environment of the worker.

    Typical scenario:
    - Server/agent does not have triton_ascend, but far away, worker does.
    - Need to return filtered API documents based on remote real runtime
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    try:
        content = await worker.get_doc(doc_name)
        return {
            "doc_name": doc_name,
            "content": content,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Get doc request failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/acquire_device")
async def acquire_device(
    task_id: str = Form(...),
    timeout: Optional[float] = Form(None),
):
    """
    Acquire a device from the device pool.
    Client should call this before generating verification scripts.

    ``timeout`` bounds the server-side wait for a free device. The client
    sends its own wait budget and uses a slightly larger HTTP read timeout, so
    the server gives up first and returns 503 — no orphaned waiter survives to
    grab a freed device for a client that already timed out.
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    try:
        wait_timeout = (
            float(timeout) if timeout is not None
            else worker_timing().acquire_timeout
        )
        # renewable: the lease carries a TTL and is kept alive by the
        # subsequent verify/profile requests (renew). If this client dies
        # before /release_device, the reaper reclaims the device.
        device_id, lease_id = await worker.device_pool.acquire_device(
            owner=task_id, timeout=wait_timeout, renewable=True
        )
        logger.info(f"[{task_id}] Acquired device {device_id} (lease {lease_id})")
        return {"device_id": device_id, "lease_id": lease_id}
    except TimeoutError as e:
        logger.warning(f"[{task_id}] No device free within {wait_timeout}s: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"[{task_id}] Failed to acquire device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/release_device")
async def release_device(
    task_id: str = Form(...),
    device_id: int = Form(...),
    lease_id: int = Form(...)
):
    """
    Release a device back to the device pool.
    Client should call this after task completion.
    """
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    try:
        await worker.device_pool.release_device(device_id, lease_id)
        logger.info(f"[{task_id}] Released device {device_id} (lease {lease_id})")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"[{task_id}] Failed to release device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def status():
    """Daemon liveness + identity. ``log_file`` echoes the daemon's stdout
    log path (set by worker_service via ``OP_AUTORESEARCH_WORKER_LOG_FILE`` env) so
    op-autoresearch's remote probe tails the actual file instead of guessing."""
    log_file = os.environ.get("OP_AUTORESEARCH_WORKER_LOG_FILE") or ""
    if worker is None:
        return {"status": "initializing", "log_file": log_file}

    backend, arch, devices = get_worker_config()
    return {
        "status": "ready",
        "backend": backend,
        "arch": arch,
        "devices": devices,
        "log_file": log_file,
    }


@app.get("/api/v1/health")
async def health():
    """Non-construction of health expeditions - \"Daemon's requested path was still alive\"
    Do not rob device:

      - Try it with ``asyncio.Queue.get_nowait()``. Device, as soon as you can.
        Put it back; empty queue (load) as healthy, unreported
      - The whole handler press the worker. health_timeout timeout; timeout is only when the event cycle itself is stuck

    /status verify only HTTP server online; /health walk through real Que operation
    Paths that capture \"event loop\" or \"queue lock competition\".**
    Blocking and waiting for device, so full-loading worker won't be miscalculated."""
    import asyncio
    if worker is None:
        return {"status": "initializing", "healthy": False, "free": 0}

    timing = worker_timing()
    backend, arch, devices = get_worker_config()
    device_pool = worker.device_pool
    pool = device_pool.available_devices
    base = {
        "status": "ready",
        "backend": backend,
        "arch": arch,
        "devices": devices,
        "free": pool.qsize(),
        "healthy": False,
    }

    async def _probe():
        # Exercise the real Queue path (get + immediate put) to catch a
        # wedged event loop / queue. The pool's free set is a plain
        # asyncio.Queue, so put_nowait wakes any pending getter on its own —
        # no Condition to coordinate. A real acquirer racing this only waits
        # the instant between get and put.
        try:
            device_id = pool.get_nowait()
        except asyncio.QueueEmpty:
            # All devices busy — daemon is fine, just at capacity.
            return None
        pool.put_nowait(device_id)
        return device_id

    try:
        device_id = await asyncio.wait_for(_probe(), timeout=timing.health_timeout)
        base["healthy"] = True
        if device_id is not None:
            base["probed_device"] = device_id
        else:
            base["note"] = "all devices busy (healthy, just at capacity)"
        return base
    except asyncio.TimeoutError:
        base["error"] = (
            f"event loop unresponsive (>{timing.health_timeout}s) "
            "— The cycle of events may be blocked"
        )
        logger.warning(
            "Health probe timed out: event loop %s did not respond in seconds",
            timing.health_timeout,
        )
        return base
    except Exception as e:
        base["error"] = f"Health detection is unusual:{type(e).__name__}: {e}"
        logger.warning(f"Health detection is unusual:{e}")
        return base


def start_server(host: Optional[str] = None, port: Optional[int] = None):
    """
    Start OP_AUTORESEARCH WORker Service.

    Args:
        host: The listening address. You can use the WOrkER_HOST settings for environment variables.
              - IPv4: \"0.0.0.0\" (all interfaces), \"127.0.0.1\" (local)
              - IPv6: \"::\" (all interfaces, double bar), \": 1\" (local)
              Default: \"0.0.0.0\"
        port: Listen port. You can use the environment variable WORKER_PORT settings.
              Default: 901
    """
    # Read configurations from environmental variables, parameters first
    if host is None:
        host = os.environ.get("WORKER_HOST", "0.0.0.0")
    if port is None:
        port = int(os.environ.get("WORKER_PORT", "9001"))

    logger.info(f"Starting Worker Service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
